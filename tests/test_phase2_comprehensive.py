"""
Comprehensive Phase 2 Test Suite

Tests all Phase 2 advanced modules:
1. Async API with WebSockets
2. Ray Distributed Training
3. AutoML Hyperparameter Optimization
4. Model Serving & Deployment
5. Integration between all components
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import asyncio
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import json


# ============================================================================
# Test Async API & WebSockets
# ============================================================================


class TestAsyncAPI:
    """Test Async API with WebSocket support."""

    @pytest.fixture
    def job_manager(self):
        """Create job manager for testing."""
        from ai_hub.async_api import TrainingJobManager

        manager = TrainingJobManager()
        yield manager
        manager.running = False

    def test_job_creation(self, job_manager):
        """Test training job creation."""
        job_id = job_manager.create_job(
            model_config={"input_dim": 128, "output_dim": 10}, total_epochs=10
        )

        assert job_id is not None
        assert len(job_id) > 0

        job = job_manager.get_job(job_id)
        assert job is not None
        assert job.status == "pending"
        assert job.progress == 0.0

    def test_job_progress_tracking(self, job_manager):
        """Test job progress updates."""
        job_id = job_manager.create_job(
            model_config={"input_dim": 10, "output_dim": 2}, total_epochs=5
        )

        job = job_manager.get_job(job_id)
        job.status = "running"
        job.progress = 50.0
        job.current_epoch = 2
        job.current_loss = 0.5

        retrieved = job_manager.get_job(job_id)
        assert retrieved.progress == 50.0
        assert retrieved.current_epoch == 2

    def test_job_cancellation(self, job_manager):
        """Test job cancellation."""
        job_id = job_manager.create_job(
            model_config={"input_dim": 10, "output_dim": 2}, total_epochs=10
        )

        # Cancel non-running job (should return False)
        result = job_manager.cancel_job(job_id)
        assert result == False

    def test_job_listing(self, job_manager):
        """Test listing all jobs."""
        job1 = job_manager.create_job(
            model_config={"input_dim": 10, "output_dim": 2}, total_epochs=5
        )
        job2 = job_manager.create_job(
            model_config={"input_dim": 20, "output_dim": 5}, total_epochs=10
        )

        jobs = job_manager.list_jobs()
        assert len(jobs) == 2

        job_ids = [j.job_id for j in jobs]
        assert job1 in job_ids
        assert job2 in job_ids


# ============================================================================
# Test Distributed Training
# ============================================================================


class TestDistributedTraining:
    """Test Ray Distributed Training."""

    @pytest.fixture
    def dist_config(self):
        """Create distributed config for testing."""
        from ai_hub.distributed import DistributedConfig

        return DistributedConfig(
            num_workers=1,  # Single worker for testing
            use_gpu=False,
            batch_size=32,
            epochs=2,
        )

    def test_config_initialization(self, dist_config):
        """Test distributed config."""
        assert dist_config.num_workers == 1
        assert dist_config.use_gpu == False
        assert dist_config.batch_size == 32
        assert dist_config.epochs == 2

    def test_multi_gpu_initialization(self):
        """Test MultiGPU training initialization."""
        from ai_hub.distributed import MultiGPUTraining

        # Should work even without GPU in test environment
        try:
            trainer = MultiGPUTraining(device_ids=[0])
            assert trainer.num_gpus == 1
            assert trainer.device_ids == [0]
        except RuntimeError as e:
            # Expected if CUDA not available
            assert "CUDA not available" in str(e)


# ============================================================================
# Test AutoML
# ============================================================================


class TestAutoML:
    """Test AutoML Hyperparameter Optimization."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer for testing."""
        from ai_hub.automl import HyperparameterOptimizer, OptimizationConfig

        config = OptimizationConfig(
            strategy="random",
            max_iterations=5,
            objective_metric="loss",
            objective_mode="min",
        )
        return HyperparameterOptimizer(config)

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.config.strategy == "random"
        assert optimizer.config.max_iterations == 5
        assert optimizer.best_params is None
        assert len(optimizer.history) == 0

    def test_search_space_definition(self, optimizer):
        """Test search space definition."""
        from ai_hub.automl import HyperparameterSpace

        optimizer.define_search_space(
            learning_rate=HyperparameterSpace(
                name="lr", param_type="log_uniform", bounds=(1e-5, 1e-1)
            ),
            batch_size=HyperparameterSpace(
                name="batch", param_type="discrete", bounds=(16, 128)
            ),
        )

        assert "learning_rate" in optimizer.search_space
        assert "batch_size" in optimizer.search_space

    def test_random_search(self, optimizer):
        """Test random search optimization."""
        from ai_hub.automl import HyperparameterSpace

        optimizer.define_search_space(
            lr=HyperparameterSpace("lr", "continuous", (0.001, 0.1))
        )

        # Simple model function
        def model_fn(params):
            return nn.Linear(10, 2)

        # Simple train function
        def train_fn(model, params):
            return np.random.random()  # Random loss

        results = optimizer.optimize(model_fn, train_fn)

        assert "best_params" in results
        assert "best_score" in results
        assert "history" in results
        assert len(results["history"]) == 5

    def test_is_better_function(self, optimizer):
        """Test comparison function."""
        # Test minimization
        assert optimizer._is_better(0.5, 1.0) == True
        assert optimizer._is_better(1.0, 0.5) == False

        # Test with maximization
        optimizer.config.objective_mode = "max"
        assert optimizer._is_better(1.0, 0.5) == True
        assert optimizer._is_better(0.5, 1.0) == False


# ============================================================================
# Test Model Serving
# ============================================================================


class TestModelServing:
    """Test Model Serving & Deployment."""

    @pytest.fixture
    def server(self):
        """Create model server for testing."""
        from ai_hub.serving import ModelServer, ServingConfig

        config = ServingConfig(
            batch_size=4,
            max_latency_ms=100.0,
            enable_gpu=False,
            enable_batching=False,  # Disable for testing
        )
        server = ModelServer(config)
        yield server
        server.shutdown()

    def test_server_initialization(self, server):
        """Test server initialization."""
        assert server.config.batch_size == 4
        assert server.config.max_latency_ms == 100.0
        assert server.config.enable_gpu == False

    def test_model_loading(self, server):
        """Test model loading."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        server.load_model(model, "test_model", "v1.0")

        assert "test_model" in server.models
        assert "v1.0" in server.models["test_model"]
        assert server.active_versions["test_model"] == "v1.0"

    def test_single_prediction(self, server):
        """Test single prediction."""
        model = nn.Linear(10, 5)
        server.load_model(model, "predict_model", "v1.0")

        data = [0.1] * 10
        response = server.predict(data, "predict_model", "v1.0")

        assert response is not None
        assert response.model_version == "v1.0"
        assert len(response.predictions) == 5
        assert response.latency_ms >= 0

    def test_ab_test_setup(self, server):
        """Test A/B test configuration."""
        model = nn.Linear(10, 5)

        server.load_model(model, "ab_model", "v1.0")
        server.load_model(model, "ab_model", "v2.0")

        server.setup_ab_test("ab_model", "v1.0", "v2.0", traffic_split=0.3)

        assert "ab_model" in server.ab_test_configs
        config = server.ab_test_configs["ab_model"]
        assert config["version_a"] == "v1.0"
        assert config["version_b"] == "v2.0"
        assert config["traffic_split"] == 0.3

    def test_metrics_collection(self, server):
        """Test metrics collection."""
        model = nn.Linear(10, 5)
        server.load_model(model, "metrics_model", "v1.0")

        # Make some predictions
        for i in range(3):
            server.predict([0.1] * 10, "metrics_model")

        metrics = server.get_metrics()

        assert metrics["total_requests"] == 3
        assert metrics["total_latency_ms"] > 0
        assert "avg_latency_ms" in metrics


# ============================================================================
# Test Integration Between Phase 2 Components
# ============================================================================


class TestPhase2Integration:
    """Integration tests for Phase 2 components."""

    def test_async_training_with_tracking(self):
        """Test async training with experiment tracking."""
        from ai_hub.async_api import TrainingJobManager
        from ai_hub.experiment_tracking import ExperimentTracker

        # Create components
        job_manager = TrainingJobManager()
        tracker = ExperimentTracker("async_test", use_mlflow=False, use_wandb=False)

        # Start experiment
        tracker.start_run()
        tracker.log_param("model_type", "async")

        # Create async job
        job_id = job_manager.create_job(
            model_config={"input_dim": 10, "output_dim": 2}, total_epochs=3
        )

        assert job_id is not None
        tracker.log_param("job_id", job_id)

        # End experiment
        tracker.end_run()

        assert len(tracker.runs) == 1
        assert tracker.runs[0].params["job_id"] == job_id

    def test_distributed_training_with_serving(self):
        """Test distributed training to serving pipeline."""
        from ai_hub.distributed import DistributedConfig
        from ai_hub.serving import ModelServer

        # Create distributed config
        dist_config = DistributedConfig(num_workers=1, use_gpu=False, epochs=1)

        # Create model
        model = nn.Linear(10, 5)

        # Serve the model
        server = ModelServer(ServingConfig(enable_batching=False))
        server.load_model(model, "distributed_model", "v1.0")

        # Test prediction
        response = server.predict([0.1] * 10, "distributed_model")

        assert response is not None
        assert response.model_version == "v1.0"

        server.shutdown()

    def test_automl_to_registry_pipeline(self):
        """Test AutoML optimization to model registry."""
        from ai_hub.automl import (
            HyperparameterOptimizer,
            OptimizationConfig,
            HyperparameterSpace,
        )
        from ai_hub.model_registry import ModelRegistry
        import tempfile

        temp_dir = tempfile.mkdtemp()
        try:
            # Optimize hyperparameters
            optimizer = HyperparameterOptimizer(
                OptimizationConfig(strategy="random", max_iterations=3)
            )

            optimizer.define_search_space(
                lr=HyperparameterSpace("lr", "continuous", (0.001, 0.01))
            )

            def model_fn(params):
                return nn.Linear(10, 2)

            def train_fn(model, params):
                return np.random.random()

            results = optimizer.optimize(model_fn, train_fn)

            # Register best model
            registry = ModelRegistry(registry_dir=temp_dir)
            best_model = model_fn(results["best_params"])

            version = registry.register_model(
                model=best_model,
                model_name="automl_optimized",
                metrics={"loss": results["best_score"]},
                parameters=results["best_params"],
            )

            assert version is not None
            assert version.params == results["best_params"]

        finally:
            shutil.rmtree(temp_dir)


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance tests for Phase 2 components."""

    def test_serving_latency(self):
        """Test model serving latency."""
        from ai_hub.serving import ModelServer, ServingConfig
        import time

        server = ModelServer(ServingConfig(enable_batching=False))
        model = nn.Linear(100, 10)
        server.load_model(model, "perf_model", "v1.0")

        # Warm up
        server.predict([0.1] * 100, "perf_model")

        # Measure latency
        latencies = []
        for _ in range(10):
            start = time.time()
            server.predict([0.1] * 100, "perf_model")
            latencies.append((time.time() - start) * 1000)

        avg_latency = np.mean(latencies)
        assert avg_latency < 100  # Should be under 100ms

        server.shutdown()

    def test_job_manager_scaling(self):
        """Test job manager with multiple jobs."""
        from ai_hub.async_api import TrainingJobManager

        manager = TrainingJobManager()

        # Create many jobs
        job_ids = []
        for i in range(100):
            job_id = manager.create_job(
                model_config={"input_dim": 10, "output_dim": 2}, total_epochs=10
            )
            job_ids.append(job_id)

        assert len(manager.list_jobs()) == 100
        assert len(set(job_ids)) == 100  # All unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
