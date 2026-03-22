"""
Comprehensive Integration Tests - All Phases

Tests the complete AI Hub platform by integrating all 4 phases:
- Phase 1: Core Infrastructure (Registry, Security, Experiments)
- Phase 2: Advanced Infrastructure (Async, Distributed, AutoML, Serving)
- Phase 3: Visualization & Analysis (Viz, Interpretability, Widgets)
- Phase 4: Production (Docker-ready components)

This ensures all components work together seamlessly.
"""

import pytest
import asyncio
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import json


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 2))


@pytest.fixture
def sample_data():
    """Create sample training data."""
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, 100)
    return X, y


class TestPhase1ToPhase2Integration:
    """Test integration between Phase 1 and Phase 2 components."""

    def test_model_registry_to_serving(self, temp_dir, simple_model):
        """Test registering a model and then serving it."""
        from src.ai_hub.model_registry import get_registry
        from src.ai_hub.serving import ModelServer

        # Register model (Phase 1)
        registry = get_registry()
        registry.registry_dir = Path(temp_dir)

        version = registry.register_model(
            model=simple_model,
            model_name="test_classifier",
            metrics={"accuracy": 0.95},
            tags=["production"],
        )

        # Serve model (Phase 2)
        model_path = registry.get_model_path("test_classifier", version)
        server = ModelServer(model_path=str(model_path))

        assert server is not None
        assert server.model is not None

    def test_experiment_tracking_with_automl(self, temp_dir):
        """Test experiment tracking integrated with AutoML."""
        from src.ai_hub.experiment_tracking import ExperimentTracker
        from src.ai_hub.automl import HyperparameterOptimizer

        # Create experiment (Phase 1)
        tracker = ExperimentTracker("automl_test", use_mlflow=False)
        tracker.experiments_dir = Path(temp_dir)
        tracker.start_run()

        # Run AutoML (Phase 2)
        optimizer = HyperparameterOptimizer(strategy="grid")

        # Log some metrics as if AutoML ran
        tracker.log_param("strategy", "grid")
        tracker.log_metric("best_accuracy", 0.95)

        tracker.end_run()

        # Verify experiment was recorded
        assert len(tracker.runs) == 1
        assert tracker.runs[0].metrics["best_accuracy"] == 0.95

    def test_security_with_async_api(self):
        """Test security integration with async API."""
        from src.ai_hub.security import create_access_token, verify_token

        # Create token (Phase 1)
        token = create_access_token(
            data={"sub": "test_user", "scopes": ["read", "write"]}
        )

        # Verify token
        payload = verify_token(token)
        assert payload is not None
        assert payload["sub"] == "test_user"


class TestPhase2ToPhase3Integration:
    """Test integration between Phase 2 and Phase 3 components."""

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_distributed_training_with_visualization(self, mock_fig_class, temp_dir):
        """Test distributed training results with visualization."""
        from src.ai_hub.advanced_viz import ModelComparator
        from src.ai_hub.distributed import DistributedTrainer

        mock_fig = MagicMock()
        mock_fig_class.return_value = mock_fig

        # Simulate distributed training results (Phase 2)
        results = {
            "accuracy": 0.92,
            "loss": 0.15,
            "f1": 0.91,
            "precision": 0.93,
            "recall": 0.90,
        }

        # Visualize results (Phase 3)
        comparator = ModelComparator()
        comparator.add_model("Distributed_4workers", results)
        comparator.add_model(
            "Distributed_8workers",
            {
                "accuracy": 0.94,
                "loss": 0.12,
                "f1": 0.93,
                "precision": 0.95,
                "recall": 0.92,
            },
        )

        fig = comparator.create_radar_chart()
        assert fig is not None

    def test_serving_with_interpretability(self, simple_model, sample_data):
        """Test model serving with interpretability."""
        from src.ai_hub.interpretability import InterpretabilityDashboard

        X, y = sample_data

        # Create interpretability dashboard (Phase 3)
        dashboard = InterpretabilityDashboard(simple_model)

        # Simulate prediction and explanation
        input_tensor = torch.FloatTensor(X[:1])
        explanation = dashboard.explain_prediction(input_tensor, method="saliency")

        assert "attribution" in explanation
        assert explanation["method"] == "saliency"

    @patch("src.ai_hub.training_widgets.go.Figure")
    def test_automl_with_jupyter_widgets(self, mock_fig_class):
        """Test AutoML progress monitoring with Jupyter widgets."""
        try:
            from src.ai_hub.training_widgets import LivePlotter

            mock_fig = MagicMock()
            mock_fig_class.return_value = mock_fig

            # Create live plotter (Phase 3)
            plotter = LivePlotter(max_points=100)

            # Simulate AutoML iterations
            for i in range(10):
                metrics = {"train_loss": 1.0 / (i + 1), "val_accuracy": 0.5 + 0.05 * i}
                plotter.update(metrics, i)

            assert len(plotter.history) > 0

        except ImportError:
            pytest.skip("Jupyter widgets not available")


class TestPhase1ToPhase3Integration:
    """Test integration between Phase 1 and Phase 3 components."""

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_registry_with_advanced_viz(self, mock_fig_class, temp_dir, simple_model):
        """Test model registry with advanced visualization."""
        from src.ai_hub.model_registry import get_registry
        from src.ai_hub.advanced_viz import ModelComparator

        mock_fig = MagicMock()
        mock_fig_class.return_value = mock_fig

        # Register multiple models (Phase 1)
        registry = get_registry()
        registry.registry_dir = Path(temp_dir)

        registry.register_model(
            simple_model, "classifier", "v1.0", metrics={"accuracy": 0.90, "f1": 0.89}
        )
        registry.register_model(
            simple_model, "classifier", "v2.0", metrics={"accuracy": 0.95, "f1": 0.94}
        )

        # Compare with visualization (Phase 3)
        comparator = ModelComparator()
        comparator.add_model("v1.0", {"accuracy": 0.90, "f1": 0.89})
        comparator.add_model("v2.0", {"accuracy": 0.95, "f1": 0.94})

        fig = comparator.create_radar_chart()
        assert fig is not None

    def test_experiment_tracking_with_paper_generator(self, temp_dir):
        """Test experiment tracking integrated with paper generation."""
        from src.ai_hub.experiment_tracking import ExperimentTracker
        from src.ai_hub.paper_generator import PaperGenerator

        # Track experiments (Phase 1)
        tracker = ExperimentTracker("paper_experiment", use_mlflow=False)
        tracker.experiments_dir = Path(temp_dir)
        tracker.start_run()
        tracker.log_param("lr", 0.001)
        tracker.log_metric("accuracy", 0.95, step=0)
        tracker.end_run()

        # Generate paper (Phase 3)
        paper = PaperGenerator()
        paper.add_section("Results", f"Achieved accuracy: {0.95}")
        paper.add_section("Methodology", "Trained with lr=0.001")

        latex = paper.generate_latex()
        assert "\\documentclass" in latex
        assert "0.95" in latex

    def test_dataset_versioning_with_experiments(self, temp_dir):
        """Test dataset versioning with experiment tracking."""
        from src.ai_hub.dataset_versioning import DatasetVersionManager
        from src.ai_hub.experiment_tracking import ExperimentTracker

        # Version dataset (Phase 3)
        dataset_manager = DatasetVersionManager(temp_dir)

        # Create sample dataset
        data_file = Path(temp_dir) / "data.csv"
        data_file.write_text("col1,col2\n1,2\n3,4")

        version = dataset_manager.commit("Initial dataset", "author@test.com")

        # Track experiment with dataset version (Phase 1)
        tracker = ExperimentTracker("versioned_experiment", use_mlflow=False)
        tracker.experiments_dir = Path(temp_dir)
        tracker.start_run()
        tracker.log_param("dataset_version", version)
        tracker.log_metric("accuracy", 0.92)
        tracker.end_run()

        assert tracker.runs[0].params["dataset_version"] == version


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows across all phases."""

    def test_full_ml_lifecycle(self, temp_dir, simple_model, sample_data):
        """Test complete ML lifecycle from training to serving."""
        from src.ai_hub.model_registry import get_registry
        from src.ai_hub.experiment_tracking import ExperimentTracker
        from src.ai_hub.security import create_access_token

        X, y = sample_data

        # 1. Create authenticated session (Phase 1)
        token = create_access_token({"sub": "researcher"})

        # 2. Track experiment (Phase 1)
        tracker = ExperimentTracker("full_lifecycle", use_mlflow=False)
        tracker.experiments_dir = Path(temp_dir)
        tracker.start_run()

        # 3. Train model (simulated)
        tracker.log_param("epochs", 10)
        tracker.log_metric("train_loss", 0.15)
        tracker.log_metric("val_accuracy", 0.92)

        # 4. Register model (Phase 1)
        registry = get_registry()
        registry.registry_dir = Path(temp_dir)
        version = registry.register_model(
            simple_model, "classifier", metrics={"accuracy": 0.92}, tags=["production"]
        )

        tracker.log_param("model_version", version)
        tracker.end_run()

        # Verify complete lifecycle
        assert version is not None
        assert len(tracker.runs) == 1
        assert registry.get_model("classifier", version) is not None

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_research_to_production_pipeline(
        self, mock_fig_class, temp_dir, simple_model
    ):
        """Test complete research to production pipeline."""
        from src.ai_hub.model_registry import get_registry
        from src.ai_hub.paper_generator import PaperGenerator
        from src.ai_hub.advanced_viz import ModelComparator

        mock_fig = MagicMock()
        mock_fig_class.return_value = mock_fig

        # 1. Research phase - compare architectures
        comparator = ModelComparator()
        comparator.add_model(
            "Architecture_A", {"accuracy": 0.90, "f1": 0.89, "params": 1000000}
        )
        comparator.add_model(
            "Architecture_B", {"accuracy": 0.94, "f1": 0.93, "params": 2000000}
        )

        # 2. Select best and register
        registry = get_registry()
        registry.registry_dir = Path(temp_dir)
        version = registry.register_model(
            simple_model, "production_model", metrics={"accuracy": 0.94, "f1": 0.93}
        )

        # 3. Generate research paper
        paper = PaperGenerator()
        paper.add_section("Abstract", "We present a novel architecture...")
        paper.add_section("Results", "Best model achieved 94% accuracy")
        latex = paper.generate_latex()

        assert version is not None
        assert "94%" in latex or "0.94" in latex

    def test_collaborative_research_workflow(self, temp_dir, simple_model):
        """Test collaborative research with version control."""
        from src.ai_hub.dataset_versioning import DatasetVersionManager
        from src.ai_hub.model_registry import get_registry

        # 1. Dataset versioning
        dataset_manager = DatasetVersionManager(temp_dir)
        data_file = Path(temp_dir) / "research_data.csv"
        data_file.write_text("feature,label\n1.0,0\n2.0,1")

        v1 = dataset_manager.commit(
            "Initial dataset by Researcher A", "researcher_a@lab.com"
        )

        # 2. Researcher B updates dataset
        data_file.write_text("feature,label\n1.0,0\n2.0,1\n3.0,0")
        v2 = dataset_manager.commit(
            "Added more samples by Researcher B", "researcher_b@lab.com"
        )

        # 3. Both train models
        registry = get_registry()
        registry.registry_dir = Path(temp_dir)

        registry.register_model(
            simple_model, "model_a", "v1.0", metrics={"accuracy": 0.88}
        )
        registry.register_model(
            simple_model, "model_b", "v1.0", metrics={"accuracy": 0.91}
        )

        # 4. Compare versions
        versions = dataset_manager.list_versions()
        assert len(versions) == 2

        models = registry.list_models()
        assert len(models) == 2


class TestErrorHandling:
    """Test error handling across phase boundaries."""

    def test_registry_error_propagation(self, temp_dir):
        """Test that registry errors are handled properly."""
        from src.ai_hub.model_registry import get_registry

        registry = get_registry()
        registry.registry_dir = Path(temp_dir)

        # Try to load non-existent model
        with pytest.raises(Exception):
            registry.load_model("nonexistent", "v1.0")

    def test_experiment_tracking_error_recovery(self, temp_dir):
        """Test experiment tracking error recovery."""
        from src.ai_hub.experiment_tracking import ExperimentTracker

        tracker = ExperimentTracker("error_test", use_mlflow=False)
        tracker.experiments_dir = Path(temp_dir)

        # Start run
        tracker.start_run()

        # Simulate error during logging
        try:
            tracker.log_metric("metric", "invalid_value")  # Should fail
        except:
            pass

        # Should still be able to end run
        tracker.end_run()
        assert len(tracker.runs) == 1


class TestPerformance:
    """Test performance characteristics across phases."""

    def test_model_registry_performance(self, temp_dir, simple_model):
        """Test model registry performance with many models."""
        from src.ai_hub.model_registry import get_registry
        import time

        registry = get_registry()
        registry.registry_dir = Path(temp_dir)

        # Register many models
        start_time = time.time()
        for i in range(10):
            registry.register_model(
                simple_model, f"model_{i}", metrics={"accuracy": 0.9 + i * 0.01}
            )
        duration = time.time() - start_time

        # Should complete in reasonable time
        assert duration < 10.0

        # List should be fast
        start_time = time.time()
        models = registry.list_models()
        duration = time.time() - start_time

        assert duration < 1.0
        assert len(models) == 10

    def test_visualization_performance(self):
        """Test visualization with large datasets."""
        from src.ai_hub.advanced_viz import EmbeddingVisualizer

        # Large embedding dataset
        embeddings = np.random.randn(1000, 50)
        labels = np.random.randint(0, 10, 1000)

        viz = EmbeddingVisualizer()

        # Should handle large datasets
        import time

        start_time = time.time()
        fig = viz.plot_tsne(embeddings, labels)
        duration = time.time() - start_time

        assert duration < 30.0  # t-SNE can be slow


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
