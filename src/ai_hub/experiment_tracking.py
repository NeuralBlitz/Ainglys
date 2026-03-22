"""
Experiment Tracking Integration for AI Hub

Supports MLflow and Weights & Biases for comprehensive experiment management.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn


@dataclass
class ExperimentRun:
    """Represents a single experiment run."""

    run_id: str
    experiment_name: str
    start_time: str
    end_time: Optional[str] = None
    params: Dict[str, Any] = None
    metrics: Dict[str, List[float]] = None
    artifacts: List[str] = None
    tags: List[str] = None
    status: str = "running"

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if self.metrics is None:
            self.metrics = {}
        if self.artifacts is None:
            self.artifacts = []
        if self.tags is None:
            self.tags = []


class ExperimentTracker:
    """
    Universal experiment tracker supporting multiple backends.

    Supports: MLflow, Weights & Biases, and local file-based tracking
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        use_mlflow: bool = True,
        use_wandb: bool = False,
        local_dir: str = "experiments",
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.active_run: Optional[ExperimentRun] = None
        self.runs: List[ExperimentRun] = []

        self._initialize_backends()

    def _initialize_backends(self):
        """Initialize tracking backends."""
        self.mlflow_client = None
        self.wandb_run = None

        # Try MLflow
        if self.use_mlflow:
            try:
                import mlflow

                if self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                self.mlflow_available = True
            except ImportError:
                self.mlflow_available = False
                print("⚠️ MLflow not installed. Install with: pip install mlflow")

        # Try W&B
        if self.use_wandb:
            try:
                import wandb

                self.wandb_available = True
            except ImportError:
                self.wandb_available = False
                print("⚠️ W&B not installed. Install with: pip install wandb")

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[List[str]] = None
    ):
        """Start a new experiment run."""
        run_id = f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.active_run = ExperimentRun(
            run_id=run_id,
            experiment_name=self.experiment_name,
            start_time=datetime.now().isoformat(),
            tags=tags or [],
        )

        # Start MLflow run
        if self.use_mlflow and self.mlflow_available:
            import mlflow

            mlflow.start_run(run_name=run_name)
            if tags:
                for tag in tags:
                    mlflow.set_tag(tag, "true")

        # Start W&B run
        if self.use_wandb and self.wandb_available:
            import wandb

            self.wandb_run = wandb.init(
                project=self.experiment_name, name=run_name or run_id, tags=tags
            )

        return self.active_run

    def log_param(self, key: str, value: Any):
        """Log a parameter."""
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self.active_run.params[key] = value

        # Log to MLflow
        if self.use_mlflow and self.mlflow_available:
            import mlflow

            mlflow.log_param(key, value)

        # Log to W&B
        if self.use_wandb and self.wandb_available and self.wandb_run:
            import wandb

            wandb.config.update({key: value})

    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        for key, value in params.items():
            self.log_param(key, value)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        if key not in self.active_run.metrics:
            self.active_run.metrics[key] = []

        self.active_run.metrics[key].append(
            (step, value) if step is not None else value
        )

        # Log to MLflow
        if self.use_mlflow and self.mlflow_available:
            import mlflow

            mlflow.log_metric(key, value, step=step)

        # Log to W&B
        if self.use_wandb and self.wandb_available and self.wandb_run:
            import wandb

            log_dict = {key: value}
            if step is not None:
                wandb.log(log_dict, step=step)
            else:
                wandb.log(log_dict)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    def log_artifact(self, filepath: str, artifact_path: Optional[str] = None):
        """Log an artifact file."""
        if self.active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self.active_run.artifacts.append(filepath)

        # Log to MLflow
        if self.use_mlflow and self.mlflow_available:
            import mlflow

            mlflow.log_artifact(filepath, artifact_path)

        # Log to W&B
        if self.use_wandb and self.wandb_available and self.wandb_run:
            import wandb

            wandb.save(filepath)

    def log_model(self, model: nn.Module, model_name: str):
        """Log a PyTorch model."""
        # Save locally
        model_dir = self.local_dir / self.active_run.run_id / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{model_name}.pt"
        torch.save(model.state_dict(), model_path)

        # Log to MLflow
        if self.use_mlflow and self.mlflow_available:
            import mlflow

            mlflow.pytorch.log_model(model, model_name)

        # Log to W&B
        if self.use_wandb and self.wandb_available and self.wandb_run:
            import wandb

            wandb.save(str(model_path))

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters in organized format."""
        # Organize by category
        organized = {"model": {}, "optimizer": {}, "training": {}, "other": {}}

        for key, value in hparams.items():
            if any(
                x in key.lower()
                for x in [
                    "lr",
                    "learning_rate",
                    "optimizer",
                    "momentum",
                    "weight_decay",
                ]
            ):
                organized["optimizer"][key] = value
            elif any(
                x in key.lower() for x in ["batch_size", "epochs", "steps", "warmup"]
            ):
                organized["training"][key] = value
            elif any(
                x in key.lower()
                for x in ["hidden_dim", "num_layers", "dropout", "architecture"]
            ):
                organized["model"][key] = value
            else:
                organized["other"][key] = value

        for category, params in organized.items():
            if params:
                self.log_params({f"{category}/{k}": v for k, v in params.items()})

    def end_run(self, status: str = "completed"):
        """End the current run."""
        if self.active_run is None:
            return

        self.active_run.end_time = datetime.now().isoformat()
        self.active_run.status = status

        # Save locally
        self._save_run_locally(self.active_run)
        self.runs.append(self.active_run)

        # End MLflow run
        if self.use_mlflow and self.mlflow_available:
            import mlflow

            mlflow.end_run(status=status)

        # End W&B run
        if self.use_wandb and self.wandb_available and self.wandb_run:
            import wandb

            wandb.finish()
            self.wandb_run = None

        self.active_run = None

    def _save_run_locally(self, run: ExperimentRun):
        """Save run to local storage."""
        run_dir = self.local_dir / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        run_file = run_dir / "run.json"
        with open(run_file, "w") as f:
            json.dump(asdict(run), f, indent=2)

    def get_run_history(self) -> List[ExperimentRun]:
        """Get history of all runs."""
        return self.runs

    def compare_runs(self, run_ids: List[str]) -> Dict:
        """Compare multiple runs."""
        runs = [r for r in self.runs if r.run_id in run_ids]

        comparison = {
            "runs": [asdict(r) for r in runs],
            "param_diff": {},
            "metric_comparison": {},
        }

        # Compare final metrics
        all_metrics = set()
        for run in runs:
            all_metrics.update(run.metrics.keys())

        for metric in all_metrics:
            comparison["metric_comparison"][metric] = {}
            for run in runs:
                if metric in run.metrics and run.metrics[metric]:
                    values = run.metrics[metric]
                    if isinstance(values[-1], tuple):
                        final_value = values[-1][1]
                    else:
                        final_value = values[-1]
                    comparison["metric_comparison"][metric][run.run_id] = final_value

        return comparison

    def get_best_run(self, metric: str, mode: str = "max") -> Optional[ExperimentRun]:
        """Get best run based on metric."""
        if not self.runs:
            return None

        valid_runs = [r for r in self.runs if metric in r.metrics and r.metrics[metric]]

        if not valid_runs:
            return None

        def get_final_value(run):
            values = run.metrics[metric]
            if isinstance(values[-1], tuple):
                return values[-1][1]
            return values[-1]

        if mode == "max":
            return max(valid_runs, key=get_final_value)
        else:
            return min(valid_runs, key=get_final_value)


class AutoTracker:
    """
    Automatic experiment tracking that hooks into PyTorch training loops.

    Usage:
        tracker = AutoTracker("my_experiment")
        model, optimizer = tracker.setup(model, optimizer)

        for epoch in range(epochs):
            train(...)
            tracker.log_epoch(epoch, loss, accuracy)
    """

    def __init__(self, experiment_name: str, log_every_n_steps: int = 10):
        self.tracker = ExperimentTracker(experiment_name)
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
        self.epoch_metrics = {}

    def setup(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        hparams: Optional[Dict] = None,
    ):
        """Setup tracking for model and optimizer."""
        self.tracker.start_run()

        # Log model architecture
        self.tracker.log_param("model_class", type(model).__name__)
        num_params = sum(p.numel() for p in model.parameters())
        self.tracker.log_param("num_parameters", num_params)

        # Log optimizer
        self.tracker.log_param("optimizer", type(optimizer).__name__)
        self.tracker.log_param("learning_rate", optimizer.defaults["lr"])

        # Log hyperparameters
        if hparams:
            self.tracker.log_hyperparameters(hparams)

        return model, optimizer

    def log_step(self, loss: float, step: Optional[int] = None):
        """Log a single training step."""
        self.step_count += 1

        if self.step_count % self.log_every_n_steps == 0:
            self.tracker.log_metric("train_loss", loss, step=step or self.step_count)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Log end of epoch metrics."""
        self.tracker.log_metric("epoch_train_loss", train_loss, step=epoch)

        if val_loss is not None:
            self.tracker.log_metric("epoch_val_loss", val_loss, step=epoch)

        if metrics:
            for name, value in metrics.items():
                self.tracker.log_metric(f"epoch_{name}", value, step=epoch)

        self.epoch_metrics[epoch] = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            **(metrics or {}),
        }

    def finish(self):
        """Finish tracking."""
        self.tracker.end_run()


# CLI utilities
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment Tracker")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--compare", nargs="+", help="Compare run IDs")
    parser.add_argument("--best", type=str, help="Get best run for metric")

    args = parser.parse_args()

    tracker = ExperimentTracker("default")

    if args.list:
        runs = tracker.get_run_history()
        print(f"\nFound {len(runs)} runs:")
        for run in runs:
            print(f"  {run.run_id}: {run.status}")

    if args.compare:
        comparison = tracker.compare_runs(args.compare)
        print(json.dumps(comparison, indent=2))

    if args.best:
        best = tracker.get_best_run(args.best)
        if best:
            print(f"\nBest run: {best.run_id}")
            print(f"  Metric '{args.best}': {best.metrics.get(args.best, [])[-1]}")
