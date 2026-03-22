"""
Advanced Visualization Components for Phase 3

Extends basic visualization with:
- 3D plots and animations
- Interactive Jupyter widgets
- Architecture diagrams
- Model comparison tools
- Attention visualization
- Training animations
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# Try to import advanced libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.manifold import TSNE

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.patches as mpatches

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import graphviz

    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


@dataclass
class AnimationConfig:
    """Configuration for training animations."""

    fps: int = 10
    duration: int = 5  # seconds
    dpi: int = 150
    figsize: Tuple[int, int] = (10, 6)


class TrainingAnimator:
    """Create training process animations."""

    def __init__(self, config: Optional[AnimationConfig] = None):
        self.config = config or AnimationConfig()
        self.frames: List[Dict[str, Any]] = []

    def add_frame(
        self, epoch: int, metrics: Dict[str, float], model_state: Optional[Dict] = None
    ):
        """Add a frame to the animation."""
        self.frames.append(
            {
                "epoch": epoch,
                "metrics": metrics.copy(),
                "timestamp": datetime.now().isoformat(),
                "model_state": model_state,
            }
        )

    def create_training_animation(self, save_path: str) -> bool:
        """Create animated training visualization."""
        if not MATPLOTLIB_AVAILABLE or len(self.frames) < 2:
            return False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figsize)

        epochs = [f["epoch"] for f in self.frames]
        losses = [f["metrics"].get("loss", 0) for f in self.frames]
        accs = [f["metrics"].get("accuracy", 0) for f in self.frames]

        # Setup plots
        (line1,) = ax1.plot([], [], "b-", linewidth=2, label="Loss")
        ax1.set_xlim(0, max(epochs))
        ax1.set_ylim(0, max(losses) * 1.1)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        (line2,) = ax2.plot([], [], "g-", linewidth=2, label="Accuracy")
        ax2.set_xlim(0, max(epochs))
        ax2.set_ylim(0, 1.0)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training Accuracy")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            return line1, line2

        def update(frame):
            current_epochs = epochs[: frame + 1]
            current_losses = losses[: frame + 1]
            current_accs = accs[: frame + 1]

            line1.set_data(current_epochs, current_losses)
            line2.set_data(current_epochs, current_accs)

            # Update title with current metrics
            fig.suptitle(
                f"Epoch {epochs[frame]}: Loss={losses[frame]:.4f}, Acc={accs[frame]:.2%}",
                fontsize=14,
                fontweight="bold",
            )

            return line1, line2

        anim = FuncAnimation(
            fig, update, frames=len(epochs), init_func=init, blit=True, interval=100
        )

        # Save as GIF or MP4
        if save_path.endswith(".gif"):
            anim.save(
                save_path, writer="pillow", fps=self.config.fps, dpi=self.config.dpi
            )
        else:
            anim.save(save_path, fps=self.config.fps, dpi=self.config.dpi)

        plt.close()
        return True


class AttentionVisualizer:
    """Visualize attention weights in transformer models."""

    def __init__(self):
        self.attention_weights = []

    def capture_attention(self, module, input, output):
        """Hook to capture attention weights."""
        if hasattr(module, "attention_weights"):
            self.attention_weights.append(module.attention_weights.detach().cpu())

    def visualize_attention_heatmap(
        self, tokens: List[str], attention_matrix: np.ndarray
    ) -> Optional[Dict]:
        """Create attention heatmap visualization."""
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure(
            data=go.Heatmap(
                z=attention_matrix,
                x=tokens,
                y=tokens,
                colorscale="Viridis",
                colorbar=dict(title="Attention Weight"),
            )
        )

        fig.update_layout(
            title="Attention Weights Heatmap",
            xaxis_title="Key",
            yaxis_title="Query",
            xaxis=dict(tickangle=45),
            height=600,
            width=700,
        )

        return fig.to_dict()

    def visualize_multi_head_attention(
        self, attention_weights: List[np.ndarray], tokens: List[str]
    ) -> Optional[Dict]:
        """Visualize multi-head attention."""
        if not PLOTLY_AVAILABLE:
            return None

        num_heads = len(attention_weights)
        fig = make_subplots(
            rows=2,
            cols=(num_heads + 1) // 2,
            subplot_titles=[f"Head {i + 1}" for i in range(num_heads)],
        )

        for idx, weights in enumerate(attention_weights):
            row = idx // ((num_heads + 1) // 2) + 1
            col = idx % ((num_heads + 1) // 2) + 1

            fig.add_trace(
                go.Heatmap(
                    z=weights,
                    x=tokens,
                    y=tokens,
                    colorscale="Blues",
                    showscale=(idx == 0),
                ),
                row=row,
                col=col,
            )

        fig.update_layout(
            title="Multi-Head Attention Visualization", height=800, width=1200
        )

        return fig.to_dict()


class ArchitectureDiagrammer:
    """Generate architecture diagrams using Graphviz."""

    def __init__(self):
        pass

    def generate_model_diagram(
        self, model: nn.Module, output_path: str = "model_architecture"
    ) -> Optional[str]:
        """Generate model architecture diagram."""
        if not GRAPHVIZ_AVAILABLE:
            return None

        dot = graphviz.Digraph(comment="Model Architecture")
        dot.attr(rankdir="TB", size="10,10")
        dot.attr("node", shape="box", style="filled", fillcolor="lightblue")

        # Add nodes for each layer
        node_id = 0
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                label = f"{type(module).__name__}\n"
                if hasattr(module, "in_features") and hasattr(module, "out_features"):
                    label += f"{module.in_features} → {module.out_features}"

                dot.node(f"node_{node_id}", label)
                node_id += 1

        # Add edges
        for i in range(node_id - 1):
            dot.edge(f"node_{i}", f"node_{i + 1}")

        # Save
        dot.render(output_path, format="png", cleanup=True)
        return f"{output_path}.png"

    def generate_computation_graph(
        self,
        model: nn.Module,
        input_shape: Tuple,
        output_path: str = "computation_graph",
    ) -> Optional[str]:
        """Generate computation graph."""
        if not GRAPHVIZ_AVAILABLE:
            return None

        dot = graphviz.Digraph(comment="Computation Graph")
        dot.attr(rankdir="LR")

        # Forward pass to trace computation
        x = torch.randn(1, *input_shape)

        # Add input node
        dot.node("input", f"Input\n{input_shape}", shape="ellipse", fillcolor="green")

        # Trace through layers
        current = x
        node_names = ["input"]

        for idx, (name, module) in enumerate(model.named_children()):
            node_name = f"layer_{idx}_{type(module).__name__}"

            try:
                current = module(current)
                shape = tuple(current.shape[1:])
                label = f"{type(module).__name__}\n{shape}"

                dot.node(node_name, label, shape="box", fillcolor="lightblue")
                dot.edge(node_names[-1], node_name)
                node_names.append(node_name)
            except:
                pass

        # Add output node
        dot.node(
            "output",
            f"Output\n{tuple(current.shape[1:])}",
            shape="ellipse",
            fillcolor="red",
        )
        dot.edge(node_names[-1], "output")

        dot.render(output_path, format="png", cleanup=True)
        return f"{output_path}.png"


class ModelComparisonVisualizer:
    """Visualize comparisons between multiple models."""

    def __init__(self):
        self.models_data: Dict[str, Dict] = {}

    def add_model(
        self, name: str, metrics: Dict[str, List[float]], color: str = "blue"
    ):
        """Add model data for comparison."""
        self.models_data[name] = {"metrics": metrics, "color": color}

    def plot_training_curves_comparison(
        self, metric_name: str = "loss"
    ) -> Optional[Dict]:
        """Plot training curves for all models."""
        if not PLOTLY_AVAILABLE or not self.models_data:
            return None

        fig = go.Figure()

        for model_name, data in self.models_data.items():
            if metric_name in data["metrics"]:
                values = data["metrics"][metric_name]
                epochs = list(range(len(values)))

                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=values,
                        mode="lines+markers",
                        name=model_name,
                        line=dict(color=data["color"], width=2),
                    )
                )

        fig.update_layout(
            title=f"Training {metric_name.capitalize()} Comparison",
            xaxis_title="Epoch",
            yaxis_title=metric_name.capitalize(),
            hovermode="x unified",
            template="plotly_white",
            height=500,
            width=800,
        )

        return fig.to_dict()

    def plot_metrics_radar(
        self, final_metrics: Dict[str, Dict[str, float]]
    ) -> Optional[Dict]:
        """Create radar chart comparing final metrics."""
        if not PLOTLY_AVAILABLE:
            return None

        # Get all metric names
        all_metrics = set()
        for metrics in final_metrics.values():
            all_metrics.update(metrics.keys())
        all_metrics = sorted(list(all_metrics))

        fig = go.Figure()

        for model_name, metrics in final_metrics.items():
            values = [metrics.get(m, 0) for m in all_metrics]

            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],  # Close the polygon
                    theta=all_metrics + [all_metrics[0]],
                    fill="toself",
                    name=model_name,
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Model Performance Radar Chart",
            showlegend=True,
            height=600,
            width=700,
        )

        return fig.to_dict()

    def plot_metrics_bar(
        self, final_metrics: Dict[str, Dict[str, float]]
    ) -> Optional[Dict]:
        """Create bar chart comparing final metrics."""
        if not PLOTLY_AVAILABLE:
            return None

        models = list(final_metrics.keys())
        metric_names = list(next(iter(final_metrics.values())).keys())

        fig = go.Figure()

        for metric in metric_names:
            values = [final_metrics[m].get(metric, 0) for m in models]
            fig.add_trace(go.Bar(name=metric, x=models, y=values))

        fig.update_layout(
            title="Final Metrics Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode="group",
            template="plotly_white",
            height=500,
            width=800,
        )

        return fig.to_dict()


class FeatureImportanceVisualizer:
    """Visualize feature importance."""

    def __init__(self):
        pass

    def plot_feature_importance(
        self, feature_names: List[str], importance_scores: np.ndarray, top_n: int = 20
    ) -> Optional[Dict]:
        """Plot feature importance bar chart."""
        if not PLOTLY_AVAILABLE:
            return None

        # Sort by importance
        indices = np.argsort(importance_scores)[::-1][:top_n]
        sorted_names = [feature_names[i] for i in indices]
        sorted_scores = importance_scores[indices]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=sorted_scores[::-1],  # Reverse for horizontal bar
                    y=sorted_names[::-1],
                    orientation="h",
                    marker_color="steelblue",
                )
            ]
        )

        fig.update_layout(
            title=f"Top {top_n} Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            template="plotly_white",
            height=max(400, top_n * 25),
            width=800,
        )

        return fig.to_dict()

    def plot_permutation_importance(
        self,
        feature_names: List[str],
        baseline_score: float,
        permuted_scores: Dict[str, List[float]],
    ) -> Optional[Dict]:
        """Plot permutation importance with error bars."""
        if not PLOTLY_AVAILABLE:
            return None

        means = []
        stds = []

        for feature in feature_names:
            if feature in permuted_scores:
                scores = permuted_scores[feature]
                importance = baseline_score - np.mean(scores)
                means.append(importance)
                stds.append(np.std(scores))
            else:
                means.append(0)
                stds.append(0)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=feature_names,
                y=means,
                error_y=dict(type="data", array=stds, visible=True),
                marker_color="coral",
            )
        )

        fig.update_layout(
            title="Permutation Feature Importance",
            xaxis_title="Feature",
            yaxis_title="Importance (Score Drop)",
            template="plotly_white",
            height=500,
            width=900,
        )

        return fig.to_dict()


class ConfusionMatrixVisualizer:
    """Visualize confusion matrices and ROC curves."""

    def __init__(self):
        pass

    def plot_confusion_matrix(
        self, cm: np.ndarray, class_names: List[str]
    ) -> Optional[Dict]:
        """Plot confusion matrix heatmap."""
        if not PLOTLY_AVAILABLE:
            return None

        # Normalize
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        fig = go.Figure(
            data=go.Heatmap(
                z=cm_normalized,
                x=class_names,
                y=class_names,
                colorscale="Blues",
                colorbar=dict(title="Proportion"),
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12},
            )
        )

        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=600,
            width=650,
        )

        return fig.to_dict()

    def plot_roc_curves(
        self, roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]]
    ) -> Optional[Dict]:
        """Plot ROC curves for multiple classes/models."""
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure()

        for name, (fpr, tpr, auc) in roc_data.items():
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{name} (AUC = {auc:.3f})",
                    line=dict(width=2),
                )
            )

        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(dash="dash", color="gray"),
            )
        )

        fig.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
            height=600,
            width=700,
        )

        return fig.to_dict()


class EmbeddingVisualizer:
    """Visualize high-dimensional embeddings."""

    def __init__(self):
        pass

    def plot_tsne(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        perplexity: int = 30,
    ) -> Optional[Dict]:
        """Create t-SNE visualization of embeddings."""
        if not SKLEARN_AVAILABLE or not PLOTLY_AVAILABLE:
            return None

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embedded = tsne.fit_transform(embeddings)

        if labels is not None:
            fig = px.scatter(
                x=embedded[:, 0],
                y=embedded[:, 1],
                color=labels.astype(str),
                title="t-SNE Visualization",
                labels={"x": "t-SNE 1", "y": "t-SNE 2", "color": "Class"},
            )
        else:
            fig = px.scatter(
                x=embedded[:, 0],
                y=embedded[:, 1],
                title="t-SNE Visualization",
                labels={"x": "t-SNE 1", "y": "t-SNE 2"},
            )

        fig.update_layout(template="plotly_white", height=600, width=700)

        return fig.to_dict()


class ExperimentTimelineVisualizer:
    """Visualize experiment timeline."""

    def __init__(self):
        pass

    def plot_experiment_timeline(self, experiments: List[Dict]) -> Optional[Dict]:
        """Create Gantt chart of experiments."""
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure()

        for idx, exp in enumerate(experiments):
            start = datetime.fromisoformat(exp["start_time"])
            end = datetime.fromisoformat(
                exp.get("end_time", datetime.now().isoformat())
            )
            duration = (end - start).total_seconds() / 3600  # hours

            fig.add_trace(
                go.Bar(
                    name=exp["name"],
                    x=[duration],
                    y=[exp["name"]],
                    orientation="h",
                    base=start,
                    marker_color="lightblue",
                )
            )

        fig.update_layout(
            title="Experiment Timeline",
            xaxis_title="Duration (hours)",
            yaxis_title="Experiment",
            barmode="stack",
            template="plotly_white",
            height=max(400, len(experiments) * 40),
            width=900,
        )

        return fig.to_dict()


class HyperparameterVisualizer:
    """Visualize hyperparameter optimization."""

    def __init__(self):
        pass

    def plot_parallel_coordinates(
        self, trials: List[Dict[str, Any]], metric: str = "score"
    ) -> Optional[Dict]:
        """Create parallel coordinates plot for hyperparameters."""
        if not PLOTLY_AVAILABLE or not trials:
            return None

        # Extract dimensions
        dimensions = []
        param_names = list(trials[0]["params"].keys())

        for param in param_names:
            values = [t["params"][param] for t in trials]
            dim = dict(range=[min(values), max(values)], label=param, values=values)
            dimensions.append(dim)

        # Add metric as final dimension
        scores = [t[metric] for t in trials]
        dimensions.append(
            dict(range=[min(scores), max(scores)], label=metric, values=scores)
        )

        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=scores,
                    colorscale="Viridis",
                    showscale=True,
                    cmin=min(scores),
                    cmax=max(scores),
                ),
                dimensions=dimensions,
            )
        )

        fig.update_layout(
            title="Hyperparameter Parallel Coordinates", height=500, width=1000
        )

        return fig.to_dict()


# Export all advanced visualization components
__all__ = [
    "TrainingAnimator",
    "AnimationConfig",
    "AttentionVisualizer",
    "ArchitectureDiagrammer",
    "ModelComparisonVisualizer",
    "FeatureImportanceVisualizer",
    "ConfusionMatrixVisualizer",
    "EmbeddingVisualizer",
    "ExperimentTimelineVisualizer",
    "HyperparameterVisualizer",
]
