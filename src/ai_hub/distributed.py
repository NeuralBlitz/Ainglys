"""
Ray Distributed Training Backend

Provides distributed training across multiple GPUs/nodes using Ray.
"""

from typing import Dict, List, Optional, Callable, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dataclasses import dataclass
import time
from datetime import datetime

# Try to import Ray
try:
    import ray
    from ray import train
    from ray.train.torch import TorchTrainer
    from ray.air.config import ScalingConfig, RunConfig

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("⚠️ Ray not installed. Install with: pip install ray[default]")


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    num_workers: int = 2
    use_gpu: bool = True
    resources_per_worker: Dict[str, float] = None
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001

    def __post_init__(self):
        if self.resources_per_worker is None:
            self.resources_per_worker = {"CPU": 2, "GPU": 1 if self.use_gpu else 0}


class DistributedTrainer:
    """
    Distributed training manager using Ray.

    Handles multi-GPU and multi-node training automatically.
    """

    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or DistributedConfig()
        self.is_initialized = False

        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray is required for distributed training. Install with: pip install ray[default]"
            )

        self._initialize_ray()

    def _initialize_ray(self):
        """Initialize Ray runtime."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        self.is_initialized = True

    def shutdown(self):
        """Shutdown Ray runtime."""
        if ray.is_initialized():
            ray.shutdown()
        self.is_initialized = False

    def train_distributed(
        self,
        model_fn: Callable[[], nn.Module],
        dataset_fn: Callable[[], Tuple[Dataset, Dataset]],
        loss_fn: Optional[nn.Module] = None,
        optimizer_fn: Optional[Callable] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """
        Train model in distributed fashion.

        Args:
            model_fn: Function that returns a model instance
            dataset_fn: Function that returns (train_dataset, val_dataset)
            loss_fn: Loss function (default: CrossEntropyLoss)
            optimizer_fn: Optimizer function (default: Adam)
            callbacks: List of callback functions

        Returns:
            Training results and metrics
        """

        def train_loop_per_worker(config: Dict):
            """Training loop executed on each worker."""
            # Get worker info
            rank = train.get_context().get_world_rank()
            world_size = train.get_context().get_world_size()

            # Create model
            model = model_fn()
            model = train.torch.prepare_model(model)

            # Create datasets
            train_dataset, val_dataset = dataset_fn()

            # Create dataloaders with distributed sampler
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=2,
            )
            train_loader = train.torch.prepare_data_loader(train_loader)

            val_loader = None
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset, batch_size=config["batch_size"], num_workers=2
                )
                val_loader = train.torch.prepare_data_loader(val_loader)

            # Setup training
            criterion = loss_fn or nn.CrossEntropyLoss()
            optimizer = (
                optimizer_fn(model.parameters())
                if optimizer_fn
                else torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            )

            # Training history
            history = {
                "train_loss": [],
                "val_loss": [],
                "train_acc": [],
                "val_acc": [],
                "epoch_times": [],
            }

            # Training loop
            for epoch in range(config["epochs"]):
                epoch_start = time.time()

                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = output.max(1)
                    train_total += target.size(0)
                    train_correct += predicted.eq(target).sum().item()

                    # Report intermediate results
                    if batch_idx % 10 == 0 and rank == 0:
                        train.report(
                            {
                                "epoch": epoch,
                                "batch": batch_idx,
                                "train_loss": loss.item(),
                                "rank": rank,
                            }
                        )

                # Validation phase
                if val_loader:
                    model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0

                    with torch.no_grad():
                        for data, target in val_loader:
                            output = model(data)
                            loss = criterion(output, target)

                            val_loss += loss.item()
                            _, predicted = output.max(1)
                            val_total += target.size(0)
                            val_correct += predicted.eq(target).sum().item()

                    val_accuracy = 100.0 * val_correct / val_total
                    history["val_loss"].append(val_loss / len(val_loader))
                    history["val_acc"].append(val_accuracy)

                # Calculate metrics
                train_accuracy = 100.0 * train_correct / train_total
                epoch_time = time.time() - epoch_start

                history["train_loss"].append(train_loss / len(train_loader))
                history["train_acc"].append(train_accuracy)
                history["epoch_times"].append(epoch_time)

                # Only rank 0 reports final epoch results
                if rank == 0:
                    train.report(
                        {
                            "epoch": epoch,
                            "train_loss": history["train_loss"][-1],
                            "train_acc": history["train_acc"][-1],
                            "val_loss": history["val_loss"][-1] if val_loader else None,
                            "val_acc": history["val_acc"][-1] if val_loader else None,
                            "epoch_time": epoch_time,
                        }
                    )

                    print(
                        f"Rank {rank} - Epoch {epoch + 1}/{config['epochs']}: "
                        f"train_loss={history['train_loss'][-1]:.4f}, "
                        f"train_acc={history['train_acc'][-1]:.2f}%, "
                        f"time={epoch_time:.2f}s"
                    )

                # Execute callbacks
                if callbacks:
                    for callback in callbacks:
                        callback(epoch, history)

            # Save model checkpoint
            if rank == 0:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "config": config,
                }
                train.save_checkpoint(checkpoint)

            return history

        # Configure scaling
        scaling_config = ScalingConfig(
            num_workers=self.config.num_workers,
            use_gpu=self.config.use_gpu,
            resources_per_worker=self.config.resources_per_worker,
        )

        # Configure run
        run_config = RunConfig(
            name=f"distributed_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            callbacks=None,
        )

        # Create trainer
        trainer = TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config={
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
            },
            scaling_config=scaling_config,
            run_config=run_config,
        )

        # Run training
        result = trainer.fit()

        return {
            "metrics": result.metrics,
            "checkpoint": result.checkpoint,
            "config": self.config,
        }

    def tune_hyperparameters(
        self,
        model_fn: Callable,
        dataset_fn: Callable,
        param_space: Dict[str, Any],
        num_samples: int = 10,
        metric: str = "val_acc",
        mode: str = "max",
    ) -> Dict[str, Any]:
        """
        Distributed hyperparameter tuning using Ray Tune.

        Args:
            model_fn: Function returning model
            dataset_fn: Function returning datasets
            param_space: Dictionary of hyperparameter search spaces
            num_samples: Number of configurations to try
            metric: Metric to optimize
            mode: "min" or "max"

        Returns:
            Best hyperparameters and results
        """
        try:
            from ray import tune
            from ray.tune import Tuner, TuneConfig
            from ray.air.config import RunConfig
        except ImportError:
            raise ImportError("Ray Tune required. Install with: pip install ray[tune]")

        def train_fn(config):
            """Training function for tuning."""
            # Merge tune config with base config
            merged_config = {**self.config.__dict__, **config}

            # Create temporary trainer
            temp_config = DistributedConfig(**merged_config)
            temp_config.num_workers = 1  # Single worker for tuning

            trainer = DistributedTrainer(temp_config)
            result = trainer.train_distributed(model_fn, dataset_fn)

            # Report final metrics
            tune.report(**result["metrics"])

        # Create tuner
        tuner = Tuner(
            train_fn,
            param_space=param_space,
            tune_config=TuneConfig(num_samples=num_samples, metric=metric, mode=mode),
            run_config=RunConfig(
                name=f"hyperparameter_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ),
        )

        # Run tuning
        results = tuner.fit()

        best_result = results.get_best_result()

        return {
            "best_config": best_result.config,
            "best_metrics": best_result.metrics,
            "all_results": [r.metrics for r in results],
        }


class MultiGPUTraining:
    """Utility for multi-GPU training on single node."""

    def __init__(self, device_ids: Optional[List[int]] = None):
        """
        Args:
            device_ids: List of GPU device IDs (None for all available)
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.num_gpus = len(self.device_ids)

        print(f"Initialized multi-GPU training on devices: {self.device_ids}")

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
    ) -> Dict[str, List]:
        """Train model on multiple GPUs using DataParallel."""

        # Wrap model in DataParallel
        if self.num_gpus > 1:
            model = nn.DataParallel(model, device_ids=self.device_ids)

        model = model.cuda()

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for data, target in train_loader:
                data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()

            # Validation
            if val_loader:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.cuda(), target.cuda()
                        output = model(data)
                        loss = criterion(output, target)

                        val_loss += loss.item()
                        _, predicted = output.max(1)
                        val_total += target.size(0)
                        val_correct += predicted.eq(target).sum().item()

                history["val_loss"].append(val_loss / len(val_loader))
                history["val_acc"].append(100.0 * val_correct / val_total)

            history["train_loss"].append(train_loss / len(train_loader))
            history["train_acc"].append(100.0 * train_correct / train_total)

            print(
                f"Epoch {epoch + 1}/{epochs}: "
                f"train_loss={history['train_loss'][-1]:.4f}, "
                f"train_acc={history['train_acc'][-1]:.2f}%"
                + (
                    f", val_loss={history['val_loss'][-1]:.4f}, "
                    f"val_acc={history['val_acc'][-1]:.2f}%"
                    if val_loader
                    else ""
                )
            )

        return history


# Export
__all__ = [
    "DistributedTrainer",
    "DistributedConfig",
    "MultiGPUTraining",
    "RAY_AVAILABLE",
]
