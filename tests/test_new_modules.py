"""
Tests for New Advanced Modules

Tests for:
- benchmarking.py - Model performance benchmarking
- data_pipeline.py - ETL and data pipeline
- monitoring.py - Drift detection and production monitoring
- feature_store.py - Centralized feature management
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 2))


class TestBenchmarking:
    """Test benchmarking module."""

    def test_benchmark_result_creation(self):
        """Test BenchmarkResult dataclass."""
        from src.ai_hub.benchmarking import BenchmarkResult

        result = BenchmarkResult(
            metric_name="latency",
            value=15.5,
            unit="ms",
            metadata={"mean": 15.5, "std": 2.1},
        )

        assert result.metric_name == "latency"
        assert result.value == 15.5
        assert result.unit == "ms"

    def test_system_profiler_info(self):
        """Test system profiler returns system info."""
        from src.ai_hub.benchmarking import SystemProfiler

        profiler = SystemProfiler()
        info = profiler.get_system_info()

        assert "cpu_count" in info
        assert "memory_total_gb" in info
        assert "cuda_available" in info

    def test_latency_benchmark(self, simple_model):
        """Test latency benchmarking."""
        from src.ai_hub.benchmarking import LatencyBenchmark

        benchmark = LatencyBenchmark(warmup_iterations=2)
        input_data = torch.randn(1, 10)

        result = benchmark.benchmark(
            simple_model, input_data, num_iterations=5, device="cpu"
        )

        assert result.metric_name == "inference_latency"
        assert result.unit == "ms"
        assert result.value > 0
        assert "mean_ms" in result.metadata

    def test_throughput_benchmark(self, simple_model):
        """Test throughput benchmarking."""
        from src.ai_hub.benchmarking import ThroughputBenchmark

        benchmark = ThroughputBenchmark()
        input_data = torch.randn(1, 10)

        result = benchmark.benchmark(
            simple_model, input_data, duration_seconds=0.5, batch_size=1, device="cpu"
        )

        assert result.metric_name == "throughput"
        assert result.unit == "inferences/sec"
        assert result.value > 0


class TestDataPipeline:
    """Test data pipeline module."""

    def test_data_quality_report(self):
        """Test data quality report generation."""
        from src.ai_hub.data_pipeline import DataQualityReport

        report = DataQualityReport(
            total_records=100,
            valid_records=95,
            missing_values={"col1": 3, "col2": 2},
            duplicates=5,
        )

        assert report.total_records == 100
        assert report.valid_records == 95
        assert report.quality_score > 0
        assert report.quality_score <= 1.0

    def test_data_validator_numeric(self):
        """Test data validator with numeric data."""
        from src.ai_hub.data_pipeline import DataValidator

        validator = DataValidator()

        df = pd.DataFrame(
            {"numeric": [1, 2, 3, 4, 5], "with_missing": [1, 2, np.nan, 4, 5]}
        )

        report = validator.validate_dataframe(df)

        assert report.total_records == 5
        assert "with_missing" in report.missing_values
        assert report.missing_values["with_missing"] == 1

    def test_pipeline_creation(self, temp_dir):
        """Test pipeline creation and execution."""
        from src.ai_hub.data_pipeline import DataPipeline, ExtractStage, TransformStage

        pipeline = DataPipeline("test_pipeline", "1.0.0")

        # Create sample data file
        data_file = Path(temp_dir) / "test_data.csv"
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        df.to_csv(data_file, index=False)

        # Add stages
        pipeline.add_stage(ExtractStage(str(data_file), "csv"))
        pipeline.add_stage(TransformStage(lambda df: df * 2, "double"))

        # Execute
        result = pipeline.execute(validate_each_stage=False)

        assert result is not None
        assert len(result) == 3

    def test_pipeline_checkpoint(self, temp_dir):
        """Test pipeline checkpoint save/load."""
        from src.ai_hub.data_pipeline import DataPipeline

        pipeline = DataPipeline("checkpoint_test", "1.0.0")
        checkpoint_path = Path(temp_dir) / "checkpoint.json"

        pipeline.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Load
        loaded = DataPipeline.from_checkpoint(str(checkpoint_path))
        assert loaded.name == "checkpoint_test"


class TestMonitoring:
    """Test monitoring module."""

    def test_drift_report_creation(self):
        """Test drift report dataclass."""
        from src.ai_hub.monitoring import DriftReport

        report = DriftReport(
            timestamp=datetime.now(),
            has_drift=True,
            drift_score=0.75,
            drift_type="data",
            features_drifted=["feature1", "feature2"],
        )

        assert report.has_drift is True
        assert report.drift_score == 0.75
        assert len(report.features_drifted) == 2

    def test_data_drift_detector_numeric(self):
        """Test numeric drift detection."""
        from src.ai_hub.monitoring import DataDriftDetector

        detector = DataDriftDetector(threshold=0.05)

        # Similar distributions - should not detect drift
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(0, 1, 1000)

        result = detector.detect_numeric_drift(ref, cur, "test_feature")

        assert "has_drift" in result
        assert "p_value" in result
        assert "feature" in result

    def test_data_drift_detector_dataframe(self):
        """Test drift detection on DataFrame."""
        from src.ai_hub.monitoring import DataDriftDetector

        detector = DataDriftDetector(threshold=0.05)

        ref_df = pd.DataFrame(
            {"num1": np.random.normal(0, 1, 100), "num2": np.random.normal(5, 2, 100)}
        )

        # Similar distribution
        cur_df = pd.DataFrame(
            {"num1": np.random.normal(0, 1, 100), "num2": np.random.normal(5, 2, 100)}
        )

        report = detector.detect_dataframe_drift(ref_df, cur_df)

        assert report.drift_type == "data"
        assert report.timestamp is not None

    def test_performance_metrics(self):
        """Test performance metrics tracking."""
        from src.ai_hub.monitoring import PerformanceMetrics

        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            accuracy=0.95,
            precision=0.94,
            recall=0.93,
            custom_metrics={"custom_1": 0.88},
        )

        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.94
        assert "custom_1" in metrics.custom_metrics

    def test_alert_manager(self):
        """Test alert management."""
        from src.ai_hub.monitoring import AlertManager

        alerts_received = []

        def handler(alert):
            alerts_received.append(alert)

        manager = AlertManager([handler])
        manager.trigger_alert("Test alert", "warning", {"test": True})

        assert len(alerts_received) == 1
        assert alerts_received[0]["message"] == "Test alert"
        assert len(manager.alert_history) == 1


class TestFeatureStore:
    """Test feature store module."""

    def test_feature_metadata(self):
        """Test feature metadata dataclass."""
        from src.ai_hub.feature_store import FeatureMetadata

        meta = FeatureMetadata(
            name="test_feature",
            version="v1",
            data_type="float64",
            description="Test feature",
            created_at=datetime.now(),
            owner="test_user",
        )

        assert meta.name == "test_feature"
        assert meta.version == "v1"
        assert meta.data_type == "float64"

    def test_feature_store_registration(self, temp_dir):
        """Test feature registration."""
        from src.ai_hub.feature_store import FeatureStore

        store = FeatureStore(temp_dir)

        # Register feature
        data = pd.Series([1, 2, 3, 4, 5])
        entity_ids = pd.Series(["e1", "e2", "e3", "e4", "e5"])

        meta = store.register_feature(
            "test_feature",
            data,
            entity_ids,
            description="Test feature description",
            owner="test_user",
        )

        assert meta.name == "test_feature"
        assert "test_feature" in store.metadata

    def test_feature_store_retrieval(self, temp_dir):
        """Test feature retrieval."""
        from src.ai_hub.feature_store import FeatureStore

        store = FeatureStore(temp_dir)

        # Register
        data = pd.Series([1.0, 2.0, 3.0])
        entity_ids = pd.Series(["e1", "e2", "e3"])
        store.register_feature("retrieval_test", data, entity_ids)

        # Retrieve
        feature_df = store.get_feature("retrieval_test")

        assert feature_df is not None
        assert len(feature_df) == 3

    def test_feature_view_creation(self, temp_dir):
        """Test feature view creation."""
        from src.ai_hub.feature_store import FeatureStore

        store = FeatureStore(temp_dir)

        # Register features
        for name in ["feat1", "feat2"]:
            store.register_feature(
                name, pd.Series([1, 2, 3]), pd.Series(["e1", "e2", "e3"])
            )

        # Create view
        view = store.create_feature_view(
            "my_view", ["feat1", "feat2"], description="Combined features"
        )

        assert view.name == "my_view"
        assert len(view.features) == 2
        assert "my_view" in store.feature_views

    def test_online_offline_features(self, temp_dir):
        """Test online and offline feature retrieval."""
        from src.ai_hub.feature_store import FeatureStore

        store = FeatureStore(temp_dir)

        # Register features
        store.register_feature(
            "online_feat", pd.Series([10, 20, 30]), pd.Series(["u1", "u2", "u3"])
        )

        # Get online features (specific entities)
        online = store.get_online_features(["online_feat"], ["u1", "u2"])

        assert len(online) == 2
        assert "online_feat" in online.columns

        # Get offline features (all)
        offline = store.get_offline_features(["online_feat"])

        assert len(offline) == 3


class TestIntegration:
    """Test integration between new modules."""

    def test_pipeline_to_monitoring(self, temp_dir):
        """Test data pipeline feeding into monitoring."""
        from src.ai_hub.data_pipeline import DataPipeline, ExtractStage
        from src.ai_hub.monitoring import DataDriftDetector

        # Create pipeline
        pipeline = DataPipeline("integration_test", "1.0.0")

        data_file = Path(temp_dir) / "data.csv"
        df = pd.DataFrame({"col1": np.random.normal(0, 1, 100)})
        df.to_csv(data_file, index=False)

        pipeline.add_stage(ExtractStage(str(data_file), "csv"))
        result = pipeline.execute(validate_each_stage=False)

        # Use result for drift detection
        detector = DataDriftDetector()
        ref = np.random.normal(0, 1, 100)
        cur = result["col1"].values

        drift_result = detector.detect_numeric_drift(ref, cur, "col1")

        assert "has_drift" in drift_result

    def test_benchmarking_to_monitoring(self, simple_model):
        """Test benchmarking feeding performance into monitoring."""
        from src.ai_hub.benchmarking import LatencyBenchmark
        from src.ai_hub.monitoring import PerformanceMetrics
        from datetime import datetime

        # Benchmark
        benchmark = LatencyBenchmark(warmup_iterations=1)
        input_data = torch.randn(1, 10)
        result = benchmark.benchmark(simple_model, input_data, num_iterations=3)

        # Log to monitoring
        metrics = PerformanceMetrics(timestamp=datetime.now(), latency_ms=result.value)

        assert metrics.latency_ms == result.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
