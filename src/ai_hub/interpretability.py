"""
Interpretability Dashboard

Provides comprehensive model interpretation and explainability features:
- Saliency maps and gradient-based attribution
- Feature importance analysis
- LIME and SHAP integration
- Interactive visualizations for model decisions
- Layer-wise relevance propagation (LRP)
- Integrated Gradients
- Counterfactual explanations

Example:
    >>> from ai_hub.interpretability import InterpretabilityDashboard
    >>> dashboard = InterpretabilityDashboard(model)
    >>> dashboard.explain_prediction(input_data, method='saliency')
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json


class SaliencyAnalyzer:
    """Compute saliency maps and gradient-based attributions."""

    def __init__(self, model: nn.Module):
        """
        Initialize saliency analyzer.

        Args:
            model: PyTorch model to analyze
        """
        self.model = model
        self.model.eval()

    def compute_saliency(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute vanilla saliency map using gradients.

        Args:
            input_tensor: Input tensor (batch_size, ...)
            target_class: Target class for attribution (None = predicted class)

        Returns:
            Saliency map as numpy array
        """
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1)

        # Compute gradients
        self.model.zero_grad()
        output[0, target_class].backward()

        # Get gradients w.r.t. input
        saliency = input_tensor.grad.data.abs()

        # Take max over channels for image data
        if len(saliency.shape) == 4:  # B, C, H, W
            saliency = saliency.max(dim=1)[0]

        return saliency.squeeze().cpu().numpy()

    def compute_smoothgrad(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        n_samples: int = 50,
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """
        Compute SmoothGrad by averaging over noisy samples.

        Args:
            input_tensor: Input tensor
            target_class: Target class
            n_samples: Number of noisy samples
            noise_level: Standard deviation of Gaussian noise

        Returns:
            Smoothed saliency map
        """
        saliencies = []

        for _ in range(n_samples):
            # Add noise
            noise = torch.randn_like(input_tensor) * noise_level
            noisy_input = input_tensor + noise

            sal = self.compute_saliency(noisy_input, target_class)
            saliencies.append(sal)

        # Average saliencies
        smooth_saliency = np.mean(saliencies, axis=0)

        return smooth_saliency

    def compute_integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        n_steps: int = 50,
    ) -> np.ndarray:
        """
        Compute Integrated Gradients attribution.

        Args:
            input_tensor: Input tensor
            target_class: Target class
            n_steps: Number of interpolation steps

        Returns:
            Integrated gradients attribution
        """
        # Baseline (zero or random)
        baseline = torch.zeros_like(input_tensor)

        # Scale inputs and compute gradients
        scaled_inputs = [
            baseline + (float(i) / n_steps) * (input_tensor - baseline)
            for i in range(n_steps + 1)
        ]

        gradients = []
        for scaled_input in scaled_inputs:
            scaled_input.requires_grad_(True)
            output = self.model(scaled_input)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            self.model.zero_grad()
            output[0, target_class].backward()
            gradients.append(scaled_input.grad.data.clone())

        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)

        # Integrated gradients
        integrated_grads = (input_tensor - baseline) * avg_gradients

        # Take absolute and sum over channels
        if len(integrated_grads.shape) == 4:
            integrated_grads = integrated_grads.abs().sum(dim=1)

        return integrated_grads.squeeze().cpu().numpy()


class FeatureAttributor:
    """Compute feature attributions using various methods."""

    def __init__(self, model: nn.Module, feature_names: Optional[List[str]] = None):
        """
        Initialize feature attributor.

        Args:
            model: PyTorch model
            feature_names: Names of input features
        """
        self.model = model
        self.feature_names = feature_names or []

    def permutation_importance(
        self, X: np.ndarray, y: np.ndarray, metric: Callable = None
    ) -> Dict[str, float]:
        """
        Compute feature importance via permutation.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels
            metric: Evaluation metric (default: accuracy)

        Returns:
            Dictionary mapping feature names to importance scores
        """
        from sklearn.metrics import accuracy_score

        metric = metric or accuracy_score

        # Baseline score
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            baseline_preds = self.model(X_tensor).argmax(dim=1).numpy()
            baseline_score = metric(y, baseline_preds)

        importance_scores = {}

        for i in range(X.shape[1]):
            # Permute feature
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])

            # Compute score with permuted feature
            with torch.no_grad():
                X_perm_tensor = torch.FloatTensor(X_permuted)
                permuted_preds = self.model(X_perm_tensor).argmax(dim=1).numpy()
                permuted_score = metric(y, permuted_preds)

            # Importance = drop in performance
            feature_name = (
                self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            )
            importance_scores[feature_name] = baseline_score - permuted_score

        return importance_scores

    def shap_values(self, X: np.ndarray, background_samples: int = 100) -> np.ndarray:
        """
        Compute SHAP values for feature attribution.

        Note: Requires shap library to be installed.

        Args:
            X: Feature matrix
            background_samples: Number of background samples for SHAP

        Returns:
            SHAP values array
        """
        try:
            import shap
        except ImportError:
            raise ImportError("shap library required. Install with: pip install shap")

        # Create SHAP explainer
        explainer = shap.DeepExplainer(self.model, X[:background_samples])
        shap_values = explainer.shap_values(X)

        return shap_values

    def lime_explanation(
        self,
        X: np.ndarray,
        instance_idx: int,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Generate LIME explanation for a single instance.

        Note: Requires lime library to be installed.

        Args:
            X: Feature matrix
            instance_idx: Index of instance to explain
            feature_names: Feature names

        Returns:
            LIME explanation dictionary
        """
        try:
            import lime
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            raise ImportError("lime library required. Install with: pip install lime")

        feature_names = feature_names or self.feature_names

        explainer = LimeTabularExplainer(
            X,
            feature_names=feature_names,
            class_names=["class_0", "class_1"],
            discretize_continuous=True,
        )

        def predict_fn(x):
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x)
                probs = torch.softmax(self.model(x_tensor), dim=1)
                return probs.numpy()

        explanation = explainer.explain_instance(
            X[instance_idx], predict_fn, num_features=10
        )

        return {
            "feature_weights": explanation.as_list(),
            "predicted_class": explanation.predicted_class,
            "score": explanation.score,
        }


class InterpretabilityDashboard:
    """
    Main interpretability dashboard combining all analysis tools.

    Provides a unified interface for model interpretation including
    saliency maps, feature importance, LIME/SHAP explanations, and
    interactive visualizations.

    Example:
        >>> dashboard = InterpretabilityDashboard(model)
        >>>
        >>> # Saliency analysis
        >>> saliency = dashboard.compute_saliency(input_image)
        >>> dashboard.visualize_saliency(input_image, saliency)
        >>>
        >>> # Feature importance
        >>> importance = dashboard.compute_feature_importance(X, y)
        >>> dashboard.plot_feature_importance(importance)
    """

    def __init__(self, model: nn.Module, feature_names: Optional[List[str]] = None):
        """
        Initialize interpretability dashboard.

        Args:
            model: PyTorch model to analyze
            feature_names: Names of input features (for tabular data)
        """
        self.model = model
        self.feature_names = feature_names

        self.saliency_analyzer = SaliencyAnalyzer(model)
        self.feature_attributor = FeatureAttributor(model, feature_names)

        self.explanation_history: List[Dict] = []

    def explain_prediction(
        self, input_data: torch.Tensor, method: str = "saliency", **kwargs
    ) -> Dict[str, Any]:
        """
        Generate explanation for a prediction.

        Args:
            input_data: Input tensor
            method: Explanation method ('saliency', 'smoothgrad',
                   'integrated_gradients')
            **kwargs: Additional method-specific arguments

        Returns:
            Explanation dictionary with attribution and metadata
        """
        explanation = {
            "method": method,
            "input_shape": input_data.shape,
            "timestamp": np.datetime64("now"),
        }

        if method == "saliency":
            target_class = kwargs.get("target_class")
            attribution = self.saliency_analyzer.compute_saliency(
                input_data, target_class
            )
            explanation["attribution"] = attribution

        elif method == "smoothgrad":
            attribution = self.saliency_analyzer.compute_smoothgrad(
                input_data,
                kwargs.get("target_class"),
                kwargs.get("n_samples", 50),
                kwargs.get("noise_level", 0.1),
            )
            explanation["attribution"] = attribution

        elif method == "integrated_gradients":
            attribution = self.saliency_analyzer.compute_integrated_gradients(
                input_data, kwargs.get("target_class"), kwargs.get("n_steps", 50)
            )
            explanation["attribution"] = attribution
        else:
            raise ValueError(f"Unknown method: {method}")

        # Store in history
        self.explanation_history.append(explanation)

        return explanation

    def compute_feature_importance(
        self, X: np.ndarray, y: np.ndarray, method: str = "permutation"
    ) -> Dict[str, float]:
        """
        Compute feature importance using specified method.

        Args:
            X: Feature matrix
            y: Target labels
            method: Method to use ('permutation', 'shap')

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if method == "permutation":
            return self.feature_attributor.permutation_importance(X, y)
        elif method == "shap":
            shap_values = self.feature_attributor.shap_values(X)
            # Aggregate SHAP values
            importance = np.abs(shap_values).mean(
                axis=(0, 2) if len(shap_values.shape) > 2 else 0
            )
            return {
                self.feature_names[i]
                if i < len(self.feature_names)
                else f"feature_{i}": float(imp)
                for i, imp in enumerate(importance)
            }
        else:
            raise ValueError(f"Unknown method: {method}")

    def visualize_saliency(
        self,
        input_image: torch.Tensor,
        saliency: np.ndarray,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive visualization of saliency map.

        Args:
            input_image: Original input image tensor
            saliency: Saliency map
            save_path: Optional path to save HTML

        Returns:
            Plotly figure object
        """
        # Convert input to numpy
        if isinstance(input_image, torch.Tensor):
            input_np = input_image.squeeze().cpu().numpy()
        else:
            input_np = input_image

        # Handle different input formats
        if len(input_np.shape) == 3 and input_np.shape[0] in [1, 3]:
            input_np = np.transpose(input_np, (1, 2, 0))

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Original Image", "Saliency Map"),
            specs=[[{"type": "image"}, {"type": "heatmap"}]],
        )

        # Original image
        fig.add_trace(go.Image(z=(input_np * 255).astype(np.uint8)), row=1, col=1)

        # Saliency map
        fig.add_trace(
            go.Heatmap(z=saliency, colorscale="Hot", showscale=True), row=1, col=2
        )

        fig.update_layout(title="Saliency Analysis", height=500, width=1000)

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        top_n: int = 20,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create interactive bar chart of feature importance.

        Args:
            importance_dict: Feature importance scores
            top_n: Number of top features to show
            save_path: Optional path to save HTML

        Returns:
            Plotly figure object
        """
        # Sort by importance
        sorted_features = sorted(
            importance_dict.items(), key=lambda x: abs(x[1]), reverse=True
        )[:top_n]

        features = [f[0] for f in sorted_features]
        scores = [f[1] for f in sorted_features]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=features,
                    y=scores,
                    marker_color=["red" if s < 0 else "blue" for s in scores],
                )
            ]
        )

        fig.update_layout(
            title=f"Top {top_n} Feature Importance",
            xaxis_title="Features",
            yaxis_title="Importance Score",
            xaxis_tickangle=-45,
            height=600,
            width=1000,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def create_interactive_dashboard(
        self,
        input_sample: torch.Tensor,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        save_path: str = "interpretability_dashboard.html",
    ) -> str:
        """
        Create comprehensive interactive dashboard with all explanations.

        Args:
            input_sample: Sample input for saliency analysis
            X_train: Training data for feature importance
            y_train: Training labels
            save_path: Path to save dashboard HTML

        Returns:
            Path to saved dashboard
        """
        # Create subplots for different explanations
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Saliency Map",
                "Feature Importance",
                "SmoothGrad",
                "Integrated Gradients",
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "heatmap"}],
            ],
        )

        # 1. Saliency
        saliency = self.saliency_analyzer.compute_saliency(input_sample)
        fig.add_trace(
            go.Heatmap(z=saliency, colorscale="Hot", name="Saliency"), row=1, col=1
        )

        # 2. Feature Importance (if data provided)
        if X_train is not None and y_train is not None:
            importance = self.compute_feature_importance(
                X_train, y_train, "permutation"
            )
            top_features = sorted(
                importance.items(), key=lambda x: abs(x[1]), reverse=True
            )[:10]
            fig.add_trace(
                go.Bar(
                    x=[f[0] for f in top_features],
                    y=[f[1] for f in top_features],
                    name="Importance",
                ),
                row=1,
                col=2,
            )

        # 3. SmoothGrad
        smoothgrad = self.saliency_analyzer.compute_smoothgrad(input_sample)
        fig.add_trace(
            go.Heatmap(z=smoothgrad, colorscale="Hot", name="SmoothGrad"), row=2, col=1
        )

        # 4. Integrated Gradients
        int_grads = self.saliency_analyzer.compute_integrated_gradients(input_sample)
        fig.add_trace(
            go.Heatmap(z=int_grads, colorscale="Hot", name="IntGrad"), row=2, col=2
        )

        fig.update_layout(
            title="Model Interpretability Dashboard",
            height=1000,
            width=1200,
            showlegend=False,
        )

        # Save dashboard
        fig.write_html(save_path)

        return save_path

    def export_explanations(self, export_path: str):
        """
        Export all explanation history to JSON.

        Args:
            export_path: Path to save JSON file
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = []
        for exp in self.explanation_history:
            exp_copy = exp.copy()
            if "attribution" in exp_copy:
                exp_copy["attribution"] = exp_copy["attribution"].tolist()
            serializable_history.append(exp_copy)

        with open(export_path, "w") as f:
            json.dump(serializable_history, f, indent=2, default=str)


def example_usage():
    """Example usage of interpretability dashboard."""
    # Create simple model
    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

    # Initialize dashboard
    dashboard = InterpretabilityDashboard(model)

    # Create dummy input
    input_data = torch.randn(1, 784)

    # Generate explanations
    saliency_exp = dashboard.explain_prediction(input_data, method="saliency")
    smoothgrad_exp = dashboard.explain_prediction(input_data, method="smoothgrad")

    # Compute feature importance
    X_train = np.random.randn(100, 784)
    y_train = np.random.randint(0, 10, 100)
    importance = dashboard.compute_feature_importance(X_train, y_train)

    # Visualize
    fig = dashboard.plot_feature_importance(importance)
    fig.show()

    # Create full dashboard
    dashboard_path = dashboard.create_interactive_dashboard(
        input_data, X_train, y_train, save_path="example_dashboard.html"
    )
    print(f"Dashboard saved to: {dashboard_path}")


if __name__ == "__main__":
    example_usage()
