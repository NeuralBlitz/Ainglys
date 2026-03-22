"""
Model Monitoring & Drift Detection

Production monitoring tools for ML models:
- Data drift detection
- Model performance monitoring
- Concept drift detection
- Alert management
- Automated retraining triggers
- Dashboard generation

Example:
    >>> from ai_hub.monitoring import ModelMonitor
    >>> monitor = ModelMonitor(model, reference_data)
    >>> drift_report = monitor.check_drift(new_data)
    >>> if drift_report['has_drift']:
    ...     monitor.trigger_alert("Data drift detected")
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from scipy import stats
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings


@dataclass
class DriftReport:
    """Report for drift detection results."""

    timestamp: datetime
    has_drift: bool
    drift_score: float
    drift_type: str  # 'data', 'concept', 'prediction'
    features_drifted: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "has_drift": self.has_drift,
            "drift_score": self.drift_score,
            "drift_type": self.drift_type,
            "features_drifted": self.features_drifted,
            "statistics": self.statistics,
        }


@dataclass
class PerformanceMetrics:
    """Model performance metrics."""

    timestamp: datetime
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "latency_ms": self.latency_ms,
            "throughput": self.throughput,
            "custom_metrics": self.custom_metrics,
        }


class DataDriftDetector:
    """Detect data drift between reference and current data."""

    def __init__(self, threshold: float = 0.05):
        """
        Initialize drift detector.

        Args:
            threshold: P-value threshold for drift detection
        """
        self.threshold = threshold

    def detect_numeric_drift(
        self, reference: np.ndarray, current: np.ndarray, feature_name: str = "feature"
    ) -> Dict[str, Any]:
        """
        Detect drift in numeric feature.

        Uses Kolmogorov-Smirnov test for distribution comparison.
        """
        if len(reference) == 0 or len(current) == 0:
            return {"has_drift": False, "p_value": 1.0, "feature": feature_name}

        # KS test
        statistic, p_value = stats.ks_2samp(reference, current)

        return {
            "has_drift": p_value < self.threshold,
            "p_value": float(p_value),
            "statistic": float(statistic),
            "feature": feature_name,
            "reference_mean": float(np.mean(reference)),
            "current_mean": float(np.mean(current)),
            "reference_std": float(np.std(reference)),
            "current_std": float(np.std(current)),
        }

    def detect_categorical_drift(
        self, reference: np.ndarray, current: np.ndarray, feature_name: str = "feature"
    ) -> Dict[str, Any]:
        """
        Detect drift in categorical feature.

        Uses Chi-square test for distribution comparison.
        """
        # Get unique categories
        ref_unique, ref_counts = np.unique(reference, return_counts=True)
        cur_unique, cur_counts = np.unique(current, return_counts=True)

        # Align categories
        all_categories = np.union1d(ref_unique, cur_unique)
        ref_aligned = np.array(
            [
                ref_counts[ref_unique == cat][0] if cat in ref_unique else 0
                for cat in all_categories
            ]
        )
        cur_aligned = np.array(
            [
                cur_counts[cur_unique == cat][0] if cat in cur_unique else 0
                for cat in all_categories
            ]
        )

        # Chi-square test
        try:
            chi2, p_value, _, _ = stats.chi2_contingency([ref_aligned, cur_aligned])
        except ValueError:
            p_value = 1.0
            chi2 = 0.0

        return {
            "has_drift": p_value < self.threshold,
            "p_value": float(p_value),
            "chi2": float(chi2),
            "feature": feature_name,
            "reference_distribution": dict(zip(ref_unique, ref_counts)),
            "current_distribution": dict(zip(cur_unique, cur_counts)),
        }

    def detect_dataframe_drift(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> DriftReport:
        """
        Detect drift across all features in a DataFrame.
        """
        features_drifted = []
        all_stats = {}
        total_drift_score = 0.0

        common_columns = set(reference_df.columns) & set(current_df.columns)

        for col in common_columns:
            ref_data = reference_df[col].dropna()
            cur_data = current_df[col].dropna()

            # Check if numeric
            if pd.api.types.is_numeric_dtype(reference_df[col]):
                result = self.detect_numeric_drift(
                    ref_data.values, cur_data.values, col
                )
            else:
                result = self.detect_categorical_drift(
                    ref_data.values, cur_data.values, col
                )

            all_stats[col] = result

            if result["has_drift"]:
                features_drifted.append(col)

            total_drift_score += 1 - result["p_value"]

        avg_drift_score = (
            total_drift_score / len(common_columns) if common_columns else 0.0
        )

        return DriftReport(
            timestamp=datetime.now(),
            has_drift=len(features_drifted) > 0,
            drift_score=avg_drift_score,
            drift_type="data",
            features_drifted=features_drifted,
            statistics=all_stats,
        )


class ConceptDriftDetector:
    """Detect concept drift (change in relationship between X and y)."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history: List[float] = []

    def detect_performance_drift(
        self,
        recent_performance: float,
        reference_performance: float,
        threshold: float = 0.1,
    ) -> DriftReport:
        """
        Detect drift based on performance degradation.
        """
        self.performance_history.append(recent_performance)

        # Keep only recent window
        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)

        performance_drop = reference_performance - recent_performance
        has_drift = performance_drop > threshold

        return DriftReport(
            timestamp=datetime.now(),
            has_drift=has_drift,
            drift_score=performance_drop,
            drift_type="concept",
            features_drifted=[],
            statistics={
                "reference_performance": reference_performance,
                "recent_performance": recent_performance,
                "performance_drop": performance_drop,
                "performance_history": self.performance_history[-10:],
            },
        )


class AlertManager:
    """Manage monitoring alerts."""

    def __init__(self, alert_handlers: Optional[List[Callable]] = None):
        self.alert_handlers = alert_handlers or []
        self.alert_history: List[Dict] = []

    def add_handler(self, handler: Callable):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)

    def trigger_alert(self, message: str, severity: str = "warning", data: Dict = None):
        """Trigger an alert."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "severity": severity,
            "data": data or {},
        }

        self.alert_history.append(alert)

        # Call all handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                warnings.warn(f"Alert handler failed: {e}")

    def get_alerts(self, since: Optional[datetime] = None) -> List[Dict]:
        """Get alerts since a specific time."""
        if since is None:
            return self.alert_history

        return [
            alert
            for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]) > since
        ]


class ModelMonitor:
    """
    Complete model monitoring system.

    Monitors models in production for drift, performance, and anomalies.

    Example:
        >>> monitor = ModelMonitor(model, reference_data=X_ref)
        >>>
        >>> # Check for drift
        >>> report = monitor.check_drift(X_new)
        >>> if report.has_drift:
        ...     monitor.trigger_alert("Data drift detected", "critical")
        >>>
        >>> # Log performance
        >>> monitor.log_performance(accuracy=0.92, latency_ms=15)
        >>>
        >>> # Get monitoring dashboard
        >>> dashboard = monitor.generate_dashboard()
    """

    def __init__(
        self,
        model: Any,
        reference_data: Optional[pd.DataFrame] = None,
        drift_threshold: float = 0.05,
        alert_handlers: Optional[List[Callable]] = None,
    ):
        """
        Initialize model monitor.

        Args:
            model: Model being monitored
            reference_data: Reference data for drift detection
            drift_threshold: Threshold for drift detection
            alert_handlers: List of alert handler functions
        """
        self.model = model
        self.reference_data = reference_data
        self.drift_detector = DataDriftDetector(threshold=drift_threshold)
        self.concept_detector = ConceptDriftDetector()
        self.alert_manager = AlertManager(alert_handlers)

        self.performance_history: List[PerformanceMetrics] = []
        self.drift_history: List[DriftReport] = []
        self.monitoring_enabled = True

    def check_drift(self, current_data: pd.DataFrame) -> DriftReport:
        """
        Check for data drift.

        Args:
            current_data: Current data to check against reference

        Returns:
            DriftReport with detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not provided")

        report = self.drift_detector.detect_dataframe_drift(
            self.reference_data, current_data
        )

        self.drift_history.append(report)

        # Trigger alert if drift detected
        if report.has_drift and self.monitoring_enabled:
            self.alert_manager.trigger_alert(
                f"Data drift detected in {len(report.features_drifted)} features",
                severity="critical" if report.drift_score > 0.5 else "warning",
                data=report.to_dict(),
            )

        return report

    def log_performance(
        self,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1_score: Optional[float] = None,
        latency_ms: Optional[float] = None,
        throughput: Optional[float] = None,
        **custom_metrics,
    ):
        """Log model performance metrics."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            latency_ms=latency_ms,
            throughput=throughput,
            custom_metrics=custom_metrics,
        )

        self.performance_history.append(metrics)

        # Keep only last 1000 metrics
        if len(self.performance_history) > 1000:
            self.performance_history.pop(0)

    def check_performance_regression(
        self, metric: str = "accuracy", threshold: float = 0.1
    ) -> DriftReport:
        """Check for performance regression (concept drift)."""
        if len(self.performance_history) < 2:
            return DriftReport(
                timestamp=datetime.now(),
                has_drift=False,
                drift_score=0.0,
                drift_type="concept",
            )

        recent = self.performance_history[-10:]
        recent_values = [
            getattr(m, metric) for m in recent if getattr(m, metric) is not None
        ]

        if len(recent_values) < 2:
            return DriftReport(
                timestamp=datetime.now(),
                has_drift=False,
                drift_score=0.0,
                drift_type="concept",
            )

        # Calculate average recent performance
        recent_avg = np.mean(recent_values)

        # Compare to baseline (earlier performance)
        baseline_values = [
            getattr(m, metric)
            for m in self.performance_history[:50]
            if getattr(m, metric) is not None
        ]
        baseline_avg = np.mean(baseline_values) if baseline_values else recent_avg

        return self.concept_detector.detect_performance_drift(
            recent_avg, baseline_avg, threshold
        )

    def trigger_alert(self, message: str, severity: str = "warning", data: Dict = None):
        """Trigger a monitoring alert."""
        self.alert_manager.trigger_alert(message, severity, data)

    def get_monitoring_summary(self) -> Dict:
        """Get summary of monitoring status."""
        recent_drifts = [
            d
            for d in self.drift_history
            if datetime.now() - d.timestamp < timedelta(days=7)
        ]

        return {
            "monitoring_enabled": self.monitoring_enabled,
            "total_checks": len(self.drift_history),
            "drifts_detected_7d": len([d for d in recent_drifts if d.has_drift]),
            "avg_drift_score_7d": np.mean([d.drift_score for d in recent_drifts])
            if recent_drifts
            else 0.0,
            "recent_alerts": len(
                self.alert_manager.get_alerts(since=datetime.now() - timedelta(days=1))
            ),
            "performance_entries": len(self.performance_history),
            "last_check": self.drift_history[-1].timestamp.isoformat()
            if self.drift_history
            else None,
        }

    def generate_dashboard(self) -> Dict:
        """Generate monitoring dashboard data."""
        return {
            "summary": self.get_monitoring_summary(),
            "drift_trends": [d.to_dict() for d in self.drift_history[-30:]],
            "performance_trends": [
                p.to_dict() for p in self.performance_history[-100:]
            ],
            "alerts": self.alert_manager.get_alerts(
                since=datetime.now() - timedelta(days=7)
            ),
        }

    def save_state(self, filepath: str):
        """Save monitoring state to file."""
        state = {
            "drift_history": [d.to_dict() for d in self.drift_history],
            "performance_history": [p.to_dict() for p in self.performance_history],
            "alert_history": self.alert_manager.alert_history,
            "summary": self.get_monitoring_summary(),
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def should_retrain(self, drift_threshold: float = 0.5) -> bool:
        """Determine if model should be retrained based on drift."""
        if not self.drift_history:
            return False

        recent_drifts = self.drift_history[-10:]
        avg_drift = np.mean([d.drift_score for d in recent_drifts])

        return avg_drift > drift_threshold


def example_usage():
    """Example usage of model monitoring."""
    import torch.nn as nn

    # Create a simple model
    model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 2))

    # Create reference data
    np.random.seed(42)
    reference_data = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 1000),
            "feature2": np.random.normal(5, 2, 1000),
            "category": np.random.choice(["A", "B", "C"], 1000),
        }
    )

    # Initialize monitor
    monitor = ModelMonitor(model, reference_data)

    # Simulate normal data
    print("Checking normal data...")
    normal_data = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(5, 2, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
        }
    )

    report1 = monitor.check_drift(normal_data)
    print(f"  Drift detected: {report1.has_drift}")
    print(f"  Drift score: {report1.drift_score:.4f}")

    # Simulate drifted data
    print("\nChecking drifted data...")
    drifted_data = pd.DataFrame(
        {
            "feature1": np.random.normal(5, 3, 100),  # Shifted mean
            "feature2": np.random.normal(15, 5, 100),  # Shifted mean
            "category": np.random.choice(["A", "B", "D"], 100),  # New category
        }
    )

    report2 = monitor.check_drift(drifted_data)
    print(f"  Drift detected: {report2.has_drift}")
    print(f"  Drift score: {report2.drift_score:.4f}")
    print(f"  Features drifted: {report2.features_drifted}")

    # Log performance
    monitor.log_performance(accuracy=0.92, latency_ms=12.5)
    monitor.log_performance(accuracy=0.91, latency_ms=13.0)
    monitor.log_performance(accuracy=0.85, latency_ms=15.0)  # Degrading

    # Check for performance regression
    regression = monitor.check_performance_regression("accuracy")
    print(f"\nPerformance regression detected: {regression.has_drift}")

    # Get summary
    summary = monitor.get_monitoring_summary()
    print(f"\nMonitoring Summary:")
    print(f"  Total checks: {summary['total_checks']}")
    print(f"  Drifts detected (7d): {summary['drifts_detected_7d']}")
    print(f"  Should retrain: {monitor.should_retrain()}")


if __name__ == "__main__":
    example_usage()
