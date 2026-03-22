"""
Comprehensive tests for Phase 3: Visualization & Analysis

Tests cover:
- Advanced visualization (training animations, attention, architecture diagrams)
- Model comparison tools
- Feature importance and confusion matrices
- Embedding visualization (t-SNE, PCA)
- Experiment timeline and hyperparameter analysis
- Basic visualization
- Paper generation
- Dataset versioning

All tests use mocking to avoid GUI dependencies in CI/CD
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil


@pytest.fixture
def mock_figure():
    """Create a mock plotly figure."""
    fig = MagicMock()
    fig.write_html = MagicMock()
    fig.show = MagicMock()
    fig.to_image = MagicMock(return_value=b"image_data")
    fig.update_layout = MagicMock(return_value=fig)
    return fig


@pytest.fixture
def sample_training_history():
    """Create sample training history data."""
    return {
        "train_loss": [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28],
        "val_loss": [1.1, 0.9, 0.7, 0.6, 0.55, 0.52, 0.51, 0.5],
        "train_acc": [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.88, 0.9],
        "val_acc": [0.45, 0.55, 0.65, 0.7, 0.75, 0.78, 0.79, 0.8],
        "epochs": list(range(8)),
    }


@pytest.fixture
def sample_attention_weights():
    """Create sample attention weights for transformer."""
    return torch.randn(2, 4, 10, 10)  # batch, heads, seq, seq


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for visualization."""
    return np.random.randn(100, 50)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


class TestAdvancedViz:
    """Test suite for advanced visualization module."""

    def test_module_imports(self):
        """Test that all advanced viz modules can be imported."""
        try:
            from src.ai_hub.advanced_viz import (
                TrainingAnimator,
                AttentionVisualizer,
                ArchitectureVisualizer,
                ModelComparator,
                FeatureImportanceViz,
                ConfusionMatrixViz,
                EmbeddingVisualizer,
                ExperimentTimeline,
                HyperparameterVisualizer,
            )
            from src.ai_hub.visualization import DashboardBuilder, MetricsPlotter
            from src.ai_hub.paper_generator import PaperGenerator
            from src.ai_hub.dataset_versioning import DatasetVersionManager

            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import visualization modules: {e}")

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_training_animator_init(self, mock_fig_class, sample_training_history):
        """Test TrainingAnimator initialization."""
        from src.ai_hub.advanced_viz import TrainingAnimator

        animator = TrainingAnimator()
        assert animator is not None
        assert hasattr(animator, "frames")

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_attention_visualizer_init(self, mock_fig_class, sample_attention_weights):
        """Test AttentionVisualizer initialization."""
        from src.ai_hub.advanced_viz import AttentionVisualizer

        viz = AttentionVisualizer()
        assert viz is not None
        assert hasattr(viz, "colormap")

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_architecture_visualizer_init(self, mock_fig_class):
        """Test ArchitectureVisualizer initialization."""
        from src.ai_hub.advanced_viz import ArchitectureVisualizer

        viz = ArchitectureVisualizer()
        assert viz is not None

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_model_comparator_init(self, mock_fig_class):
        """Test ModelComparator initialization."""
        from src.ai_hub.advanced_viz import ModelComparator

        comparator = ModelComparator()
        assert comparator is not None
        assert hasattr(comparator, "models")

    def test_model_comparator_add_model(self):
        """Test adding models to comparator."""
        from src.ai_hub.advanced_viz import ModelComparator

        comparator = ModelComparator()
        metrics = {"accuracy": 0.85, "precision": 0.84, "recall": 0.83, "f1": 0.835}
        comparator.add_model("Model_A", metrics)

        assert "Model_A" in comparator.models
        assert comparator.models["Model_A"] == metrics

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_model_comparator_radar_chart(self, mock_fig_class):
        """Test radar chart creation."""
        from src.ai_hub.advanced_viz import ModelComparator

        mock_fig = MagicMock()
        mock_fig_class.return_value = mock_fig

        comparator = ModelComparator()
        comparator.add_model(
            "Model_A",
            {"accuracy": 0.85, "precision": 0.84, "recall": 0.83, "f1": 0.835},
        )

        fig = comparator.create_radar_chart()
        assert fig is not None

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_feature_importance_init(self, mock_fig_class):
        """Test FeatureImportanceViz initialization."""
        from src.ai_hub.advanced_viz import FeatureImportanceViz

        viz = FeatureImportanceViz()
        assert viz is not None

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_feature_importance_plot(self, mock_fig_class):
        """Test feature importance plotting."""
        from src.ai_hub.advanced_viz import FeatureImportanceViz

        mock_fig = MagicMock()
        mock_fig_class.return_value = mock_fig

        viz = FeatureImportanceViz()
        features = ["feat1", "feat2", "feat3", "feat4", "feat5"]
        importance = [0.3, 0.25, 0.2, 0.15, 0.1]

        fig = viz.plot_feature_importance(features, importance)
        assert fig is not None

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_confusion_matrix_init(self, mock_fig_class):
        """Test ConfusionMatrixViz initialization."""
        from src.ai_hub.advanced_viz import ConfusionMatrixViz

        viz = ConfusionMatrixViz()
        assert viz is not None

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_confusion_matrix_plot(self, mock_fig_class):
        """Test confusion matrix plotting."""
        from src.ai_hub.advanced_viz import ConfusionMatrixViz

        mock_fig = MagicMock()
        mock_fig_class.return_value = mock_fig

        viz = ConfusionMatrixViz()
        cm = np.array([[45, 5], [3, 47]])
        classes = ["Class_A", "Class_B"]

        fig = viz.plot_confusion_matrix(cm, classes)
        assert fig is not None

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_confusion_matrix_normalized(self, mock_fig_class):
        """Test normalized confusion matrix."""
        from src.ai_hub.advanced_viz import ConfusionMatrixViz

        mock_fig = MagicMock()
        mock_fig_class.return_value = mock_fig

        viz = ConfusionMatrixViz()
        cm = np.array([[45, 5], [3, 47]])
        classes = ["Class_A", "Class_B"]

        fig = viz.plot_confusion_matrix(cm, classes, normalize=True)
        assert fig is not None

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_embedding_visualizer_init(self, mock_fig_class):
        """Test EmbeddingVisualizer initialization."""
        from src.ai_hub.advanced_viz import EmbeddingVisualizer

        viz = EmbeddingVisualizer()
        assert viz is not None

    @patch("src.ai_hub.advanced_viz.go.Figure")
    @patch("src.ai_hub.advanced_viz.TSNE")
    def test_embedding_tsne(self, mock_tsne, mock_fig_class, sample_embeddings):
        """Test t-SNE embedding visualization."""
        from src.ai_hub.advanced_viz import EmbeddingVisualizer

        mock_fig = MagicMock()
        mock_fig_class.return_value = mock_fig

        mock_tsne_instance = MagicMock()
        mock_tsne_instance.fit_transform.return_value = np.random.randn(100, 2)
        mock_tsne.return_value = mock_tsne_instance

        viz = EmbeddingVisualizer()
        labels = np.random.randint(0, 3, 100)

        fig = viz.plot_tsne(sample_embeddings, labels)
        assert fig is not None

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_experiment_timeline_init(self, mock_fig_class):
        """Test ExperimentTimeline initialization."""
        from src.ai_hub.advanced_viz import ExperimentTimeline

        timeline = ExperimentTimeline()
        assert timeline is not None
        assert hasattr(timeline, "experiments")

    def test_experiment_timeline_add(self):
        """Test adding experiments to timeline."""
        from src.ai_hub.advanced_viz import ExperimentTimeline
        from datetime import datetime

        timeline = ExperimentTimeline()
        timeline.add_experiment(
            "exp_001", datetime.now(), datetime.now(), "Test experiment"
        )

        assert len(timeline.experiments) == 1
        assert timeline.experiments[0]["name"] == "exp_001"

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_experiment_timeline_plot(self, mock_fig_class):
        """Test experiment timeline plotting."""
        from src.ai_hub.advanced_viz import ExperimentTimeline
        from datetime import datetime

        mock_fig = MagicMock()
        mock_fig_class.return_value = mock_fig

        timeline = ExperimentTimeline()
        now = datetime.now()
        timeline.add_experiment("exp_001", now, now, "Test 1")
        timeline.add_experiment("exp_002", now, now, "Test 2")

        fig = timeline.plot_timeline()
        assert fig is not None

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_hyperparameter_visualizer_init(self, mock_fig_class):
        """Test HyperparameterVisualizer initialization."""
        from src.ai_hub.advanced_viz import HyperparameterVisualizer

        viz = HyperparameterVisualizer()
        assert viz is not None

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_hyperparameter_parallel_coords(self, mock_fig_class):
        """Test parallel coordinates for hyperparameters."""
        from src.ai_hub.advanced_viz import HyperparameterVisualizer

        mock_fig = MagicMock()
        mock_fig_class.return_value = mock_fig

        viz = HyperparameterVisualizer()
        hparams = [
            {"lr": 0.01, "batch_size": 32, "accuracy": 0.85},
            {"lr": 0.001, "batch_size": 64, "accuracy": 0.87},
            {"lr": 0.0001, "batch_size": 128, "accuracy": 0.86},
        ]

        fig = viz.plot_parallel_coordinates(hparams)
        assert fig is not None


class TestBasicViz:
    """Test suite for basic visualization module."""

    def test_metrics_plotter_init(self, mock_figure):
        """Test MetricsPlotter initialization."""
        from src.ai_hub.visualization import MetricsPlotter

        plotter = MetricsPlotter()
        assert plotter is not None

    @patch("src.ai_hub.visualization.go.Figure")
    def test_metrics_plotter_training(self, mock_fig_class, sample_training_history):
        """Test training metrics plotting."""
        from src.ai_hub.visualization import MetricsPlotter

        mock_fig = MagicMock()
        mock_fig_class.return_value = mock_fig

        plotter = MetricsPlotter()
        fig = plotter.plot_training_history(
            sample_training_history["train_loss"],
            sample_training_history["val_loss"],
            sample_training_history["train_acc"],
            sample_training_history["val_acc"],
        )
        assert fig is not None

    def test_dashboard_builder_init(self, mock_figure):
        """Test DashboardBuilder initialization."""
        from src.ai_hub.visualization import DashboardBuilder

        builder = DashboardBuilder("Test Dashboard")
        assert builder is not None
        assert builder.title == "Test Dashboard"


class TestPaperGenerator:
    """Test suite for paper generation module."""

    def test_paper_generator_init(self):
        """Test PaperGenerator initialization."""
        from src.ai_hub.paper_generator import PaperGenerator

        generator = PaperGenerator()
        assert generator is not None
        assert hasattr(generator, "sections")

    def test_paper_add_section(self):
        """Test adding sections to paper."""
        from src.ai_hub.paper_generator import PaperGenerator

        generator = PaperGenerator()
        generator.add_section("Introduction", "This is the intro.")

        assert "Introduction" in generator.sections
        assert generator.sections["Introduction"] == "This is the intro."

    def test_paper_generate_latex(self):
        """Test LaTeX generation."""
        from src.ai_hub.paper_generator import PaperGenerator

        generator = PaperGenerator()
        generator.add_section("Abstract", "Test abstract.")
        generator.add_section("Introduction", "Test intro.")

        latex = generator.generate_latex()
        assert isinstance(latex, str)
        assert "\\documentclass" in latex
        assert "Abstract" in latex

    def test_paper_add_figure(self):
        """Test adding figures to paper."""
        from src.ai_hub.paper_generator import PaperGenerator

        generator = PaperGenerator()
        generator.add_figure("figure1.png", "Test figure caption.")

        assert len(generator.figures) == 1
        assert generator.figures[0]["caption"] == "Test figure caption."

    def test_paper_add_table(self):
        """Test adding tables to paper."""
        from src.ai_hub.paper_generator import PaperGenerator

        generator = PaperGenerator()
        data = {"Metric": ["Acc", "Loss"], "Value": [0.9, 0.1]}
        df = pd.DataFrame(data)
        generator.add_table(df, "Test table caption.")

        assert len(generator.tables) == 1


class TestDatasetVersioning:
    """Test suite for dataset versioning module."""

    def test_version_manager_init(self, temp_dir):
        """Test DatasetVersionManager initialization."""
        from src.ai_hub.dataset_versioning import DatasetVersionManager

        manager = DatasetVersionManager(temp_dir)
        assert manager is not None
        assert manager.repo_path == temp_dir

    def test_version_manager_commit(self, temp_dir):
        """Test dataset version committing."""
        from src.ai_hub.dataset_versioning import DatasetVersionManager

        manager = DatasetVersionManager(temp_dir)

        # Create a dummy file
        test_file = Path(temp_dir) / "data.csv"
        test_file.write_text("col1,col2\n1,2\n3,4")

        version = manager.commit("Initial commit", "author@test.com")
        assert version is not None
        assert isinstance(version, str)

    def test_version_manager_list_versions(self, temp_dir):
        """Test listing dataset versions."""
        from src.ai_hub.dataset_versioning import DatasetVersionManager

        manager = DatasetVersionManager(temp_dir)

        test_file = Path(temp_dir) / "data.csv"
        test_file.write_text("col1,col2\n1,2")

        manager.commit("Commit 1", "author@test.com")

        versions = manager.list_versions()
        assert isinstance(versions, list)

    def test_version_manager_checkout(self, temp_dir):
        """Test checking out a specific version."""
        from src.ai_hub.dataset_versioning import DatasetVersionManager

        manager = DatasetVersionManager(temp_dir)

        test_file = Path(temp_dir) / "data.csv"
        test_file.write_text("col1,col2\n1,2")

        version = manager.commit("Commit 1", "author@test.com")

        # Update file
        test_file.write_text("col1,col2\n5,6")
        version2 = manager.commit("Commit 2", "author@test.com")

        # Checkout first version
        manager.checkout(version)
        content = test_file.read_text()
        assert "1,2" in content

    def test_version_manager_diff(self, temp_dir):
        """Test diff between versions."""
        from src.ai_hub.dataset_versioning import DatasetVersionManager

        manager = DatasetVersionManager(temp_dir)

        test_file = Path(temp_dir) / "data.csv"
        test_file.write_text("col1,col2\n1,2")

        v1 = manager.commit("Commit 1", "author@test.com")
        test_file.write_text("col1,col2\n5,6")
        v2 = manager.commit("Commit 2", "author@test.com")

        diff = manager.diff(v1, v2)
        assert isinstance(diff, str)


class TestIntegration:
    """Integration tests for Phase 3 modules working together."""

    @patch("src.ai_hub.advanced_viz.go.Figure")
    def test_full_visualization_pipeline(
        self, mock_fig_class, sample_training_history, temp_dir
    ):
        """Test full visualization pipeline from training to export."""
        from src.ai_hub.advanced_viz import (
            ModelComparator,
            ConfusionMatrixViz,
            FeatureImportanceViz,
        )
        from src.ai_hub.paper_generator import PaperGenerator

        mock_fig = MagicMock()
        mock_fig_class.return_value = mock_fig

        # Create model comparison
        comparator = ModelComparator()
        comparator.add_model(
            "Model_A",
            {"accuracy": 0.85, "precision": 0.84, "recall": 0.83, "f1": 0.835},
        )
        comparator.add_model(
            "Model_B",
            {"accuracy": 0.88, "precision": 0.87, "recall": 0.86, "f1": 0.865},
        )

        fig1 = comparator.create_radar_chart()
        assert fig1 is not None

        # Create confusion matrix
        cm_viz = ConfusionMatrixViz()
        cm = np.array([[45, 5], [3, 47]])
        fig2 = cm_viz.plot_confusion_matrix(cm, ["Class_A", "Class_B"])
        assert fig2 is not None

        # Create feature importance
        feat_viz = FeatureImportanceViz()
        fig3 = feat_viz.plot_feature_importance(
            ["feat1", "feat2", "feat3"], [0.5, 0.3, 0.2]
        )
        assert fig3 is not None

        # Generate paper
        paper = PaperGenerator()
        paper.add_section("Results", "Model comparison results.")
        paper.add_section("Analysis", "Detailed analysis.")

        latex = paper.generate_latex()
        assert "\\documentclass" in latex

    def test_experiment_to_visualization_flow(self, sample_training_history, temp_dir):
        """Test flow from experiment to visualization."""
        from src.ai_hub.advanced_viz import ExperimentTimeline
        from datetime import datetime

        timeline = ExperimentTimeline()
        now = datetime.now()

        # Simulate multiple experiments
        timeline.add_experiment("preprocessing", now, now, "Data preprocessing")
        timeline.add_experiment("training", now, now, "Model training")
        timeline.add_experiment("evaluation", now, now, "Model evaluation")

        assert len(timeline.experiments) == 3

        # Verify experiments are in order
        names = [e["name"] for e in timeline.experiments]
        assert names == ["preprocessing", "training", "evaluation"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_model_comparator(self):
        """Test ModelComparator with no models."""
        from src.ai_hub.advanced_viz import ModelComparator

        comparator = ModelComparator()

        with pytest.raises(ValueError):
            comparator.create_radar_chart()

    def test_confusion_matrix_empty_classes(self):
        """Test ConfusionMatrixViz with empty classes."""
        from src.ai_hub.advanced_viz import ConfusionMatrixViz

        viz = ConfusionMatrixViz()

        with pytest.raises(ValueError):
            viz.plot_confusion_matrix(np.array([[1, 2], [3, 4]]), [])

    def test_feature_importance_mismatch(self):
        """Test FeatureImportanceViz with mismatched lengths."""
        from src.ai_hub.advanced_viz import FeatureImportanceViz

        viz = FeatureImportanceViz()

        with pytest.raises(ValueError):
            viz.plot_feature_importance(["feat1", "feat2"], [0.5, 0.3, 0.2])

    def test_paper_generator_empty_sections(self):
        """Test PaperGenerator with no sections."""
        from src.ai_hub.paper_generator import PaperGenerator

        generator = PaperGenerator()

        with pytest.raises(ValueError):
            generator.generate_latex()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
