"""
Jupyter Training Widgets

Provides interactive Jupyter widgets for real-time training visualization:
- Live training metrics plots
- Progress bars
- Real-time loss/accuracy curves
- GPU utilization monitoring
- Interactive hyperparameter adjustment
- Training control (pause/resume/stop)
- Model architecture visualization

Usage:
    >>> from ai_hub.training_widgets import TrainingDashboard
    >>> dashboard = TrainingDashboard()
    >>> dashboard.start_training_monitoring(trainer)
"""

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    from ipywidgets import HBox, VBox, Layout, Output

    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import threading
import time
from typing import Optional, Dict, List, Callable, Any
from collections import deque


class LivePlotter:
    """Real-time plot updater for Jupyter notebooks."""

    def __init__(self, max_points: int = 1000):
        """
        Initialize live plotter.

        Args:
            max_points: Maximum number of points to keep in history
        """
        self.max_points = max_points
        self.history: Dict[str, deque] = {}
        self.output = Output()
        self.fig = None
        self._setup_plot()

    def _setup_plot(self):
        """Setup initial plot structure."""
        self.fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Loss", "Accuracy", "Learning Rate", "Other Metrics"),
            specs=[[{}, {}], [{}, {}]],
        )

        # Loss
        self.fig.add_trace(
            go.Scatter(x=[], y=[], name="Train Loss", mode="lines"), row=1, col=1
        )
        self.fig.add_trace(
            go.Scatter(x=[], y=[], name="Val Loss", mode="lines"), row=1, col=1
        )

        # Accuracy
        self.fig.add_trace(
            go.Scatter(x=[], y=[], name="Train Acc", mode="lines"), row=1, col=2
        )
        self.fig.add_trace(
            go.Scatter(x=[], y=[], name="Val Acc", mode="lines"), row=1, col=2
        )

        # Learning Rate
        self.fig.add_trace(
            go.Scatter(x=[], y=[], name="LR", mode="lines"), row=2, col=1
        )

        # Other metrics
        self.fig.add_trace(
            go.Scatter(x=[], y=[], name="Metric", mode="lines"), row=2, col=2
        )

        self.fig.update_layout(height=600, showlegend=True, title="Training Progress")

    def update(self, metrics: Dict[str, float], step: int):
        """
        Update plot with new metrics.

        Args:
            metrics: Dictionary of metric names to values
            step: Current training step
        """
        # Store in history
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = deque(maxlen=self.max_points)
            self.history[key].append((step, value))

        # Update traces
        with self.output:
            clear_output(wait=True)

            # Update loss
            if "train_loss" in self.history:
                steps, values = zip(*self.history["train_loss"])
                self.fig.data[0].x = steps
                self.fig.data[0].y = values

            if "val_loss" in self.history:
                steps, values = zip(*self.history["val_loss"])
                self.fig.data[1].x = steps
                self.fig.data[1].y = values

            # Update accuracy
            if "train_acc" in self.history:
                steps, values = zip(*self.history["train_acc"])
                self.fig.data[2].x = steps
                self.fig.data[2].y = values

            if "val_acc" in self.history:
                steps, values = zip(*self.history["val_acc"])
                self.fig.data[3].x = steps
                self.fig.data[3].y = values

            # Update learning rate
            if "lr" in self.history:
                steps, values = zip(*self.history["lr"])
                self.fig.data[4].x = steps
                self.fig.data[4].y = values

            display(self.fig)

    def show(self):
        """Display the plot."""
        display(self.output)
        with self.output:
            display(self.fig)


class TrainingDashboard:
    """
    Interactive training dashboard with Jupyter widgets.

    Provides real-time visualization of training progress with:
    - Live metric plots
    - Progress bars
    - GPU monitoring
    - Training controls
    - Hyperparameter display

    Example:
        >>> dashboard = TrainingDashboard()
        >>> dashboard.display()
        >>>
        >>> for epoch in range(10):
        >>>     metrics = trainer.train_epoch()
        >>>     dashboard.update_metrics(metrics, epoch)
    """

    def __init__(self, update_interval: float = 1.0):
        """
        Initialize training dashboard.

        Args:
            update_interval: Update interval in seconds
        """
        if not JUPYTER_AVAILABLE:
            raise ImportError(
                "Jupyter widgets not available. Install with:\n"
                "pip install ipywidgets plotly\n"
                "jupyter nbextension enable --py widgetsnbextension"
            )

        self.update_interval = update_interval
        self.metrics_history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
            "epoch": [],
        }

        self._setup_widgets()
        self._running = False
        self._paused = False
        self._callbacks: List[Callable] = []

    def _setup_widgets(self):
        """Setup all dashboard widgets."""
        # Progress bar
        self.progress_bar = widgets.FloatProgress(
            value=0,
            min=0,
            max=100,
            description="Progress:",
            bar_style="info",
            style={"bar_color": "#00bfff"},
            layout=Layout(width="100%"),
        )

        # Metric displays
        self.train_loss_text = widgets.FloatText(
            description="Train Loss:",
            value=0.0,
            disabled=True,
            layout=Layout(width="200px"),
        )

        self.val_loss_text = widgets.FloatText(
            description="Val Loss:",
            value=0.0,
            disabled=True,
            layout=Layout(width="200px"),
        )

        self.train_acc_text = widgets.FloatText(
            description="Train Acc:",
            value=0.0,
            disabled=True,
            layout=Layout(width="200px"),
        )

        self.val_acc_text = widgets.FloatText(
            description="Val Acc:",
            value=0.0,
            disabled=True,
            layout=Layout(width="200px"),
        )

        self.epoch_text = widgets.IntText(
            description="Epoch:", value=0, disabled=True, layout=Layout(width="150px")
        )

        self.lr_text = widgets.FloatText(
            description="LR:", value=0.001, disabled=True, layout=Layout(width="150px")
        )

        # Status display
        self.status_label = widgets.Label(
            value="Status: Ready", layout=Layout(width="300px")
        )

        # Control buttons
        self.pause_button = widgets.Button(
            description="Pause", button_style="warning", icon="pause"
        )
        self.pause_button.on_click(self._on_pause)

        self.stop_button = widgets.Button(
            description="Stop", button_style="danger", icon="stop"
        )
        self.stop_button.on_click(self._on_stop)

        # GPU monitoring (if available)
        self.gpu_text = widgets.Textarea(
            value="GPU: Not available",
            description="GPU:",
            disabled=True,
            layout=Layout(width="300px", height="80px"),
        )

        # Live plot
        self.plotter = LivePlotter()

        # Assemble layout
        self.metrics_box = HBox(
            [
                self.epoch_text,
                self.train_loss_text,
                self.val_loss_text,
                self.train_acc_text,
                self.val_acc_text,
                self.lr_text,
            ]
        )

        self.controls_box = HBox(
            [self.pause_button, self.stop_button, self.status_label]
        )

        self.dashboard = VBox(
            [
                widgets.HTML("<h3>Training Dashboard</h3>"),
                self.progress_bar,
                self.metrics_box,
                self.controls_box,
                self.gpu_text,
                self.plotter.output,
            ]
        )

    def _on_pause(self, button):
        """Handle pause button click."""
        self._paused = not self._paused
        if self._paused:
            self.pause_button.description = "Resume"
            self.pause_button.icon = "play"
            self.status_label.value = "Status: Paused"
        else:
            self.pause_button.description = "Pause"
            self.pause_button.icon = "pause"
            self.status_label.value = "Status: Running"

    def _on_stop(self, button):
        """Handle stop button click."""
        self._running = False
        self.status_label.value = "Status: Stopped"
        self.pause_button.disabled = True
        self.stop_button.disabled = True

    def _update_gpu_info(self):
        """Update GPU information if available."""
        try:
            if torch.cuda.is_available():
                gpu_id = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(gpu_id)
                memory_used = torch.cuda.memory_allocated(gpu_id) / 1e9
                memory_total = (
                    torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
                )
                utilization = (memory_used / memory_total) * 100

                info = f"GPU: {gpu_name}\n"
                info += f"Memory: {memory_used:.2f} / {memory_total:.2f} GB\n"
                info += f"Utilization: {utilization:.1f}%"
                self.gpu_text.value = info
        except Exception:
            pass

    def display(self):
        """Display the dashboard."""
        display(self.dashboard)
        self.plotter.show()

    def update_metrics(
        self, metrics: Dict[str, float], epoch: int, total_epochs: int = None
    ):
        """
        Update dashboard with new metrics.

        Args:
            metrics: Dictionary of metric names to values
            epoch: Current epoch number
            total_epochs: Total number of epochs (for progress bar)
        """
        # Update text displays
        if "train_loss" in metrics:
            self.train_loss_text.value = metrics["train_loss"]
            self.metrics_history["train_loss"].append(metrics["train_loss"])

        if "val_loss" in metrics:
            self.val_loss_text.value = metrics["val_loss"]
            self.metrics_history["val_loss"].append(metrics["val_loss"])

        if "train_acc" in metrics:
            self.train_acc_text.value = metrics["train_acc"]
            self.metrics_history["train_acc"].append(metrics["train_acc"])

        if "val_acc" in metrics:
            self.val_acc_text.value = metrics["val_acc"]
            self.metrics_history["val_acc"].append(metrics["val_acc"])

        if "lr" in metrics:
            self.lr_text.value = metrics["lr"]
            self.metrics_history["lr"].append(metrics["lr"])

        self.epoch_text.value = epoch
        self.metrics_history["epoch"].append(epoch)

        # Update progress bar
        if total_epochs:
            self.progress_bar.value = (epoch / total_epochs) * 100
            self.progress_bar.max = 100

        # Update GPU info
        self._update_gpu_info()

        # Update plot
        self.plotter.update(metrics, epoch)

        # Check if paused
        while self._paused and self._running:
            time.sleep(0.1)

    def start_training_monitoring(self, trainer: Any):
        """
        Start monitoring a training process.

        Args:
            trainer: Training object with a train method
        """
        self._running = True
        self.status_label.value = "Status: Running"

        def monitor():
            epoch = 0
            while self._running:
                try:
                    # Check if trainer has current metrics
                    if hasattr(trainer, "current_metrics"):
                        metrics = trainer.current_metrics
                        self.update_metrics(metrics, epoch)
                        epoch += 1
                    time.sleep(self.update_interval)
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    break

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.daemon = True
        monitor_thread.start()

    def add_callback(self, callback: Callable):
        """
        Add callback function for metric updates.

        Args:
            callback: Function to call with metrics dict
        """
        self._callbacks.append(callback)

    def export_history(self, filepath: str):
        """
        Export training history to file.

        Args:
            filepath: Path to save history
        """
        import json

        history_dict = {
            key: list(values) for key, values in self.metrics_history.items()
        }

        with open(filepath, "w") as f:
            json.dump(history_dict, f, indent=2)


class HyperparameterWidget:
    """
    Interactive widget for adjusting hyperparameters during training.

    Allows real-time modification of:
    - Learning rate
    - Batch size
    - Dropout rate
    - Weight decay
    - Other custom parameters
    """

    def __init__(self, param_config: Dict[str, Dict] = None):
        """
        Initialize hyperparameter widget.

        Args:
            param_config: Configuration dict for parameters
                Example:
                {
                    'learning_rate': {
                        'type': 'float',
                        'min': 1e-5,
                        'max': 1e-1,
                        'default': 1e-3,
                        'step': 1e-5
                    },
                    'batch_size': {
                        'type': 'int',
                        'min': 8,
                        'max': 512,
                        'default': 32,
                        'step': 8
                    }
                }
        """
        if not JUPYTER_AVAILABLE:
            raise ImportError("Jupyter widgets not available")

        self.param_config = param_config or self._default_config()
        self.widgets = {}
        self._setup_widgets()
        self._on_change_callbacks = []

    def _default_config(self) -> Dict:
        """Get default parameter configuration."""
        return {
            "learning_rate": {
                "type": "float",
                "min": 1e-6,
                "max": 1.0,
                "default": 1e-3,
                "step": 1e-6,
                "description": "Learning Rate:",
            },
            "batch_size": {
                "type": "int",
                "min": 1,
                "max": 1024,
                "default": 32,
                "step": 1,
                "description": "Batch Size:",
            },
            "dropout": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "step": 0.05,
                "description": "Dropout:",
            },
            "weight_decay": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.0,
                "step": 1e-5,
                "description": "Weight Decay:",
            },
        }

    def _setup_widgets(self):
        """Setup parameter widgets."""
        for param_name, config in self.param_config.items():
            if config["type"] == "float":
                widget = widgets.FloatSlider(
                    value=config["default"],
                    min=config["min"],
                    max=config["max"],
                    step=config.get("step", 0.01),
                    description=config.get("description", f"{param_name}:"),
                    continuous_update=False,
                    layout=Layout(width="500px"),
                )
            elif config["type"] == "int":
                widget = widgets.IntSlider(
                    value=config["default"],
                    min=config["min"],
                    max=config["max"],
                    step=config.get("step", 1),
                    description=config.get("description", f"{param_name}:"),
                    continuous_update=False,
                    layout=Layout(width="500px"),
                )
            else:
                widget = widgets.Text(
                    value=str(config["default"]),
                    description=config.get("description", f"{param_name}:"),
                    layout=Layout(width="500px"),
                )

            widget.observe(self._on_widget_change, names="value")
            self.widgets[param_name] = widget

        # Create update button
        self.update_button = widgets.Button(
            description="Apply Changes", button_style="success", icon="check"
        )
        self.update_button.on_click(self._on_apply)

        # Status label
        self.status_label = widgets.Label(value="Ready")

        # Assemble layout
        widget_list = list(self.widgets.values())
        self.layout = VBox(
            [
                widgets.HTML("<h4>Hyperparameters</h4>"),
                *widget_list,
                HBox([self.update_button, self.status_label]),
            ]
        )

    def _on_widget_change(self, change):
        """Handle widget value changes."""
        self.status_label.value = "Changes pending..."

    def _on_apply(self, button):
        """Handle apply button click."""
        self.status_label.value = "Changes applied!"

        # Call registered callbacks
        params = self.get_values()
        for callback in self._on_change_callbacks:
            callback(params)

    def get_values(self) -> Dict[str, Any]:
        """
        Get current parameter values.

        Returns:
            Dictionary of parameter names to values
        """
        return {name: widget.value for name, widget in self.widgets.items()}

    def set_values(self, values: Dict[str, Any]):
        """
        Set parameter values.

        Args:
            values: Dictionary of parameter names to values
        """
        for name, value in values.items():
            if name in self.widgets:
                self.widgets[name].value = value

    def on_change(self, callback: Callable):
        """
        Register callback for parameter changes.

        Args:
            callback: Function called with new parameter dict
        """
        self._on_change_callbacks.append(callback)

    def display(self):
        """Display the widget."""
        display(self.layout)


def demo_training_widget():
    """Demonstrate training widget functionality."""
    if not JUPYTER_AVAILABLE:
        print("Jupyter widgets not available. Install with:")
        print("  pip install ipywidgets plotly")
        return

    # Create dashboard
    dashboard = TrainingDashboard()
    dashboard.display()

    # Create hyperparameter widget
    hp_widget = HyperparameterWidget()
    hp_widget.display()

    # Simulate training
    print("Simulating training...")
    for epoch in range(100):
        metrics = {
            "train_loss": 1.0 / (epoch + 1) + np.random.randn() * 0.1,
            "val_loss": 1.1 / (epoch + 1) + np.random.randn() * 0.1,
            "train_acc": min(0.99, epoch / 100 + np.random.randn() * 0.05),
            "val_acc": min(0.95, epoch / 100 + np.random.randn() * 0.05),
            "lr": 0.001 * (0.95**epoch),
        }

        dashboard.update_metrics(metrics, epoch, total_epochs=100)
        time.sleep(0.5)


if __name__ == "__main__":
    demo_training_widget()
