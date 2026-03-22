"""
Advanced Visualization Dashboard with Plotly

Provides interactive visualizations for:
- Training metrics in real-time
- Model architecture visualization
- Hyperparameter optimization landscapes
- Experiment comparisons
- Dataset distributions
"""

import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np

# Try to import plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not installed. Install with: pip install plotly")

import torch
import torch.nn as nn


@dataclass
class DashboardConfig:
    """Configuration for the dashboard."""

    title: str = "AI Hub Dashboard"
    theme: str = "plotly_white"
    realtime_update_interval: int = 1000  # ms
    max_points: int = 10000


class MetricsVisualizer:
    """Visualize training metrics with interactive plots."""

    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.history: Dict[str, List[Dict]] = {}

    def add_metric(self, run_id: str, metric_name: str, value: float, step: int):
        """Add a metric value."""
        if run_id not in self.history:
            self.history[run_id] = []

        self.history[run_id].append(
            {
                "step": step,
                "metric": metric_name,
                "value": value,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def create_training_dashboard(self, run_id: str) -> Optional[Dict]:
        """Create comprehensive training metrics dashboard."""
        if not PLOTLY_AVAILABLE:
            return None

        if run_id not in self.history:
            return None

        data = self.history[run_id]

        # Separate metrics
        metrics_by_name = {}
        for point in data:
            metric = point["metric"]
            if metric not in metrics_by_name:
                metrics_by_name[metric] = {"steps": [], "values": []}
            metrics_by_name[metric]["steps"].append(point["step"])
            metrics_by_name[metric]["values"].append(point["value"])

        # Create subplots
        num_metrics = len(metrics_by_name)
        if num_metrics == 0:
            return None

        fig = make_subplots(
            rows=num_metrics,
            cols=1,
            subplot_titles=list(metrics_by_name.keys()),
            vertical_spacing=0.1,
        )

        for idx, (metric_name, metric_data) in enumerate(metrics_by_name.items(), 1):
            fig.add_trace(
                go.Scatter(
                    x=metric_data["steps"],
                    y=metric_data["values"],
                    mode="lines+markers",
                    name=metric_name,
                    line=dict(width=2),
                    marker=dict(size=4),
                ),
                row=idx,
                col=1,
            )

            # Add rolling average
            if len(metric_data["values"]) > 10:
                window = min(10, len(metric_data["values"]) // 10)
                rolling_avg = np.convolve(
                    metric_data["values"], np.ones(window) / window, mode="valid"
                )
                fig.add_trace(
                    go.Scatter(
                        x=metric_data["steps"][window - 1 :],
                        y=rolling_avg,
                        mode="lines",
                        name=f"{metric_name} (avg)",
                        line=dict(width=2, dash="dash"),
                    ),
                    row=idx,
                    col=1,
                )

        fig.update_layout(
            title=f"Training Metrics - {run_id}",
            height=300 * num_metrics,
            showlegend=True,
            template=self.config.theme,
        )

        return fig.to_dict()

    def compare_runs(self, run_ids: List[str], metric_name: str) -> Optional[Dict]:
        """Compare multiple runs on the same metric."""
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure()

        for run_id in run_ids:
            if run_id not in self.history:
                continue

            data = [p for p in self.history[run_id] if p["metric"] == metric_name]

            if data:
                steps = [p["step"] for p in data]
                values = [p["value"] for p in data]

                fig.add_trace(
                    go.Scatter(
                        x=steps,
                        y=values,
                        mode="lines+markers",
                        name=run_id[:8],  # Shorten run_id
                    )
                )

        fig.update_layout(
            title=f"Comparison: {metric_name}",
            xaxis_title="Step",
            yaxis_title=metric_name,
            template=self.config.theme,
        )

        return fig.to_dict()


class ArchitectureVisualizer:
    """Visualize neural network architectures."""

    def __init__(self):
        self.layers = []

    def visualize_model(
        self, model: nn.Module, input_shape: tuple = (1, 128)
    ) -> Optional[Dict]:
        """Create interactive model architecture visualization."""
        if not PLOTLY_AVAILABLE:
            return None

        # Collect layer information
        layer_info = []

        def hook_fn(module, input, output):
            layer_type = type(module).__name__
            params = sum(p.numel() for p in module.parameters())

            layer_info.append(
                {
                    "type": layer_type,
                    "params": params,
                    "input_shape": list(input[0].shape) if input else None,
                    "output_shape": list(output.shape)
                    if hasattr(output, "shape")
                    else None,
                }
            )

        # Register hooks
        hooks = []
        for module in model.modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass to collect info
        try:
            dummy_input = torch.randn(*input_shape)
            model(dummy_input)
        except:
            pass
        finally:
            for hook in hooks:
                hook.remove()

        if not layer_info:
            return None

        # Create visualization
        fig = go.Figure()

        y_positions = list(range(len(layer_info)))

        # Add layers as boxes
        for idx, layer in enumerate(layer_info):
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[y_positions[idx]],
                    mode="markers+text",
                    marker=dict(
                        size=40, color=idx, colorscale="Viridis", showscale=False
                    ),
                    text=f"{layer['type']}\n{layer['params']:,} params",
                    textposition="middle center",
                    hovertemplate=(
                        f"<b>{layer['type']}</b><br>"
                        + f"Parameters: {layer['params']:,}<br>"
                        + f"Input: {layer.get('input_shape', 'N/A')}<br>"
                        + f"Output: {layer.get('output_shape', 'N/A')}<br>"
                    ),
                    showlegend=False,
                )
            )

        # Add connections
        for i in range(len(layer_info) - 1):
            fig.add_trace(
                go.Scatter(
                    x=[0, 0],
                    y=[y_positions[i], y_positions[i + 1]],
                    mode="lines",
                    line=dict(color="gray", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        total_params = sum(l["params"] for l in layer_info)

        fig.update_layout(
            title=f"Model Architecture ({total_params:,} parameters)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            height=max(400, len(layer_info) * 80),
            template="plotly_white",
        )

        return fig.to_dict()

    def create_architecture_summary(self, model: nn.Module) -> Dict[str, Any]:
        """Create text-based architecture summary."""
        summary = {"total_parameters": 0, "trainable_parameters": 0, "layers": []}

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                params = sum(p.numel() for p in module.parameters())
                trainable = sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )

                summary["layers"].append(
                    {
                        "name": name,
                        "type": type(module).__name__,
                        "parameters": params,
                        "trainable": trainable,
                    }
                )

                summary["total_parameters"] += params
                summary["trainable_parameters"] += trainable

        return summary


class HyperparameterLandscapeVisualizer:
    """Visualize hyperparameter optimization landscapes."""

    def __init__(self):
        self.history: List[Dict] = []

    def add_evaluation(self, params: Dict[str, float], score: float):
        """Add a hyperparameter evaluation."""
        evaluation = params.copy()
        evaluation["score"] = score
        self.history.append(evaluation)

    def create_landscape_plot(
        self, param_x: str, param_y: str, param_z: Optional[str] = None
    ) -> Optional[Dict]:
        """Create 2D or 3D landscape plot."""
        if not PLOTLY_AVAILABLE or len(self.history) < 2:
            return None

        data = self.history

        if param_z:  # 3D scatter
            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=[d[param_x] for d in data],
                        y=[d[param_y] for d in data],
                        z=[d[param_z] for d in data],
                        mode="markers",
                        marker=dict(
                            size=6,
                            color=[d["score"] for d in data],
                            colorscale="Viridis",
                            colorbar=dict(title="Score"),
                            showscale=True,
                        ),
                        text=[f"Score: {d['score']:.4f}" for d in data],
                        hovertemplate=(
                            f"{param_x}: %{{x}}<br>"
                            + f"{param_y}: %{{y}}<br>"
                            + f"{param_z}: %{{z}}<br>"
                            + "Score: %{marker.color:.4f}"
                        ),
                    )
                ]
            )

            fig.update_layout(
                title="Hyperparameter Landscape (3D)",
                scene=dict(
                    xaxis_title=param_x, yaxis_title=param_y, zaxis_title=param_z
                ),
                template="plotly_white",
            )
        else:  # 2D scatter with color
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=[d[param_x] for d in data],
                        y=[d[param_y] for d in data],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=[d["score"] for d in data],
                            colorscale="Viridis",
                            colorbar=dict(title="Score"),
                            showscale=True,
                        ),
                        text=[f"Score: {d['score']:.4f}" for d in data],
                        hovertemplate=(
                            f"{param_x}: %{{x}}<br>"
                            + f"{param_y}: %{{y}}<br>"
                            + "Score: %{marker.color:.4f}"
                        ),
                    )
                ]
            )

            fig.update_layout(
                title="Hyperparameter Landscape (2D)",
                xaxis_title=param_x,
                yaxis_title=param_y,
                template="plotly_white",
            )

        return fig.to_dict()

    def create_optimization_history(self) -> Optional[Dict]:
        """Plot optimization progress over time."""
        if not PLOTLY_AVAILABLE or len(self.history) < 2:
            return None

        iterations = list(range(len(self.history)))
        scores = [h["score"] for h in self.history]

        # Calculate best so far
        best_so_far = []
        current_best = scores[0]
        for score in scores:
            if score < current_best:  # Assuming minimization
                current_best = score
            best_so_far.append(current_best)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=scores,
                mode="markers",
                name="Evaluations",
                marker=dict(size=8, opacity=0.6),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=best_so_far,
                mode="lines",
                name="Best so far",
                line=dict(width=3, color="red"),
            )
        )

        fig.update_layout(
            title="Hyperparameter Optimization History",
            xaxis_title="Iteration",
            yaxis_title="Score",
            template="plotly_white",
        )

        return fig.to_dict()


class DatasetVisualizer:
    """Visualize dataset distributions and statistics."""

    def __init__(self):
        pass

    def visualize_distribution(
        self,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        labels: Optional[np.ndarray] = None,
    ) -> Optional[Dict]:
        """Create dataset distribution plots."""
        if not PLOTLY_AVAILABLE:
            return None

        num_features = data.shape[1] if len(data.shape) > 1 else 1

        if num_features == 1:
            # Single feature histogram
            fig = go.Figure()

            if labels is not None:
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    mask = labels == label
                    fig.add_trace(
                        go.Histogram(
                            x=data[mask].flatten(), name=f"Class {label}", opacity=0.7
                        )
                    )
            else:
                fig.add_trace(go.Histogram(x=data.flatten(), name="Distribution"))

            fig.update_layout(
                title="Feature Distribution",
                xaxis_title=feature_names[0] if feature_names else "Value",
                yaxis_title="Count",
                barmode="overlay",
                template="plotly_white",
            )

        elif num_features == 2:
            # 2D scatter plot
            fig = go.Figure()

            if labels is not None:
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    mask = labels == label
                    fig.add_trace(
                        go.Scatter(
                            x=data[mask, 0],
                            y=data[mask, 1],
                            mode="markers",
                            name=f"Class {label}",
                            marker=dict(size=6, opacity=0.6),
                        )
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=data[:, 0],
                        y=data[:, 1],
                        mode="markers",
                        marker=dict(size=6, opacity=0.6),
                    )
                )

            fig.update_layout(
                title="Data Distribution (2D)",
                xaxis_title=feature_names[0] if feature_names else "Feature 1",
                yaxis_title=feature_names[1] if feature_names else "Feature 2",
                template="plotly_white",
            )

        else:
            # Pair plot for multiple features (limited to first 4)
            features_to_plot = min(num_features, 4)

            fig = make_subplots(
                rows=features_to_plot,
                cols=features_to_plot,
                subplot_titles=[
                    f"{feature_names[i] if feature_names else f'Feature {i}'} vs "
                    f"{feature_names[j] if feature_names else f'Feature {j}'}"
                    for i in range(features_to_plot)
                    for j in range(features_to_plot)
                ],
            )

            for i in range(features_to_plot):
                for j in range(features_to_plot):
                    if i == j:
                        # Diagonal: histogram
                        fig.add_trace(go.Histogram(x=data[:, i]), row=i + 1, col=j + 1)
                    else:
                        # Off-diagonal: scatter
                        fig.add_trace(
                            go.Scatter(
                                x=data[:, j],
                                y=data[:, i],
                                mode="markers",
                                marker=dict(size=3, opacity=0.5),
                            ),
                            row=i + 1,
                            col=j + 1,
                        )

            fig.update_layout(
                title="Feature Correlation Matrix",
                height=300 * features_to_plot,
                showlegend=False,
                template="plotly_white",
            )

        return fig.to_dict()


class DashboardManager:
    """Manage all dashboard visualizations."""

    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.metrics_viz = MetricsVisualizer(self.config)
        self.arch_viz = ArchitectureVisualizer()
        self.hp_viz = HyperparameterLandscapeVisualizer()
        self.dataset_viz = DatasetVisualizer()

    def export_plot(self, plot_dict: Dict, filepath: str):
        """Export plot to HTML file."""
        if not PLOTLY_AVAILABLE:
            return

        import plotly.offline as pyo

        pyo.plot(plot_dict, filename=filepath, auto_open=False)

    def get_available_visualizations(self) -> List[str]:
        """Get list of available visualization types."""
        viz_types = [
            "training_metrics",
            "run_comparison",
            "model_architecture",
            "hyperparameter_landscape",
            "dataset_distribution",
        ]
        return viz_types


# Export
__all__ = [
    "DashboardManager",
    "DashboardConfig",
    "MetricsVisualizer",
    "ArchitectureVisualizer",
    "HyperparameterLandscapeVisualizer",
    "DatasetVisualizer",
    "PLOTLY_AVAILABLE",
]
