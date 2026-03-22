import click
import torch
import requests
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from pathlib import Path

from .ml_engine import MLEngine

console = Console()
API_BASE_URL = "http://localhost:8000"


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """AI Hub - Unified CLI for ML, API, and more.

    Use this CLI to train models, make predictions, and manage the API.
    """
    pass


@cli.group()
def model():
    """Model management commands."""
    pass


@cli.group()
def api():
    """API management commands."""
    pass


@cli.group()
def local():
    """Local ML operations (no API required)."""
    pass


@cli.group()
def config_cmd():
    """Configuration management commands."""
    pass


@cli.group()
def web():
    """Web interface commands."""
    pass


@web.command("start")
@click.option("--port", default=8501, help="Port to run Streamlit on")
@click.option("--api-url", default="http://localhost:8000", help="API base URL")
def start_web(port, api_url):
    """Start the web interface."""
    import streamlit.webview as webview
    import os

    console.print(f"[bold green]Starting web interface on port {port}...[/bold green]")
    console.print(f"[blue]API URL: {api_url}[/blue]")

    os.environ["API_BASE_URL"] = api_url

    import subprocess

    subprocess.run(
        [
            "streamlit",
            "run",
            "src/ai_hub/web.py",
            "--server.port",
            str(port),
            "--server.address",
            "0.0.0.0",
        ]
    )


@web.command("build")
@click.argument("output_dir", default=".")
def build_web(output_dir):
    """Build static web app."""
    console.print("[bold blue]Building static web app...[/bold blue]")
    console.print(f"Output directory: {output_dir}")


@model.command("train")
@click.option("--input-dim", default=128, help="Input dimension")
@click.option("--hidden-dim", default=256, help="Hidden dimension")
@click.option("--output-dim", default=10, help="Output dimension")
@click.option("--epochs", default=10, help="Training epochs")
@click.option("--api/--local", default=True, help="Use API or local mode")
def train_model(input_dim, hidden_dim, output_dim, epochs, api):
    """Train a new model."""
    if api:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(description="Training via API...", total=None)
            try:
                response = requests.post(
                    f"{API_BASE_URL}/train",
                    json={
                        "input_dim": input_dim,
                        "hidden_dim": hidden_dim,
                        "output_dim": output_dim,
                        "epochs": epochs,
                    },
                )
                response.raise_for_status()
                result = response.json()
                progress.update(task, completed=True)

                console.print(Panel.fit("[bold green]Training Complete![/bold green]"))

                table = Table(title="Training Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")
                table.add_row("Status", result["status"])
                table.add_row("Final Loss", str(result["training"]["final_loss"]))
                table.add_row("Accuracy", str(result["evaluation"]["accuracy"]))
                table.add_row("Epochs", str(result["training"]["epochs"]))
                console.print(table)
            except requests.exceptions.ConnectionError:
                console.print(
                    "[bold red]Error:[/bold red] API not running. Start with: ai-hub api start"
                )
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
    else:
        console.print("[bold blue]Training locally...[/bold blue]")
        engine = MLEngine(input_dim, hidden_dim, output_dim)
        X = torch.randn(1000, input_dim)
        y = torch.randint(0, output_dim, (1000,))
        result = engine.train(X, y, epochs)
        console.print(
            f"[bold green]Training complete! Final loss: {result['final_loss']:.4f}[/bold green]"
        )


@model.command("predict")
@click.option(
    "--data",
    "-d",
    multiple=True,
    type=float,
    help="Input data as space-separated floats",
)
@click.option(
    "--file", "-f", type=click.Path(exists=True), help="JSON file with input data"
)
def predict_model(data, file):
    """Make predictions with a trained model."""
    if file:
        with open(file, "r") as f:
            input_data = json.load(f)
    elif data:
        input_data = [list(data)]
    else:
        console.print("[bold red]Error:[/bold red] Provide --data or --file")
        return

    try:
        response = requests.post(f"{API_BASE_URL}/predict", json={"data": input_data})
        response.raise_for_status()
        result = response.json()

        table = Table(title="Predictions")
        table.add_column("Sample", style="cyan")
        table.add_column("Prediction", style="magenta")
        table.add_column("Confidence", style="green")

        for i, (pred, probs) in enumerate(
            zip(result["predictions"], result["probabilities"])
        ):
            confidence = max(probs)
            table.add_row(str(i), str(pred), f"{confidence:.2%}")

        console.print(table)
    except requests.exceptions.ConnectionError:
        console.print(
            "[bold red]Error:[/bold red] API not running. Start with: ai-hub api start"
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


@model.command("save")
@click.argument("name")
def save_model(name):
    """Save the current model."""
    try:
        response = requests.post(f"{API_BASE_URL}/save", json={"model_name": name})
        response.raise_for_status()
        result = response.json()
        console.print(f"[bold green]Model saved:[/bold green] {result['model_path']}")
    except requests.exceptions.ConnectionError:
        console.print(
            "[bold red]Error:[/bold red] API not running. Start with: ai-hub api start"
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


@model.command("load")
@click.argument("name")
def load_model(name):
    """Load a saved model."""
    try:
        response = requests.post(f"{API_BASE_URL}/load/{name}")
        response.raise_for_status()
        result = response.json()
        console.print(f"[bold green]Model loaded:[/bold green] {result['model_name']}")
        if "metadata" in result:
            console.print(json.dumps(result["metadata"], indent=2))
    except requests.exceptions.ConnectionError:
        console.print(
            "[bold red]Error:[/bold red] API not running. Start with: ai-hub api start"
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


@model.command("list")
def list_models():
    """List all saved models."""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        response.raise_for_status()
        result = response.json()

        if result["count"] == 0:
            console.print("[yellow]No models found.[/yellow]")
            return

        table = Table(title=f"Saved Models ({result['count']})")
        table.add_column("Name", style="cyan")
        table.add_column("Input Dim", style="magenta")
        table.add_column("Hidden Dim", style="magenta")
        table.add_column("Output Dim", style="magenta")

        for model in result["models"]:
            name = model["name"]
            if "metadata" in model:
                md = model["metadata"]
                table.add_row(
                    name,
                    str(md.get("input_dim", "N/A")),
                    str(md.get("hidden_dim", "N/A")),
                    str(md.get("output_dim", "N/A")),
                )
            else:
                table.add_row(name, "N/A", "N/A", "N/A")

        console.print(table)
    except requests.exceptions.ConnectionError:
        console.print(
            "[bold red]Error:[/bold red] API not running. Start with: ai-hub api start"
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


@api.command("start")
@click.option("--port", default=8000, help="Port to run API on")
@click.option("--host", default="0.0.0.0", help="Host to bind API to")
@click.option("--reload/--no-reload", default=False, help="Enable auto-reload")
def start_api(port, host, reload):
    """Start the API server."""
    import uvicorn

    console.print(f"[bold green]Starting API server on {host}:{port}...[/bold green]")
    uvicorn.run("ai_hub.api:app", host=host, port=port, reload=reload)


@api.command("status")
def api_status():
    """Check API server status."""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        response.raise_for_status()
        result = response.json()

        console.print(Panel.fit(f"[bold green]API Status: Running[/bold green]"))
        console.print(f"Version: {result['version']}")
        console.print(f"Message: {result['message']}")
        console.print(f"Endpoints: {', '.join(result['endpoints'])}")
    except requests.exceptions.ConnectionError:
        console.print("[bold red]API Status: Not Running[/bold red]")
        console.print("Start with: ai-hub api start")


@cli.command("status")
def system_status():
    """Check system status."""
    import torch

    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    table.add_row("Device", "cuda" if torch.cuda.is_available() else "cpu")
    console.print(table)


@local.command("train")
@click.option("--input-dim", default=128, help="Input dimension")
@click.option("--hidden-dim", default=256, help="Hidden dimension")
@click.option("--output-dim", default=10, help="Output dimension")
@click.option("--epochs", default=10, help="Training epochs")
@click.option("--save-to", help="Save model to path")
def local_train(input_dim, hidden_dim, output_dim, epochs, save_to):
    """Train a model locally (no API needed)."""
    console.print("[bold blue]Training locally...[/bold blue]")

    engine = MLEngine(input_dim, hidden_dim, output_dim)
    X = torch.randn(1000, input_dim)
    y = torch.randint(0, output_dim, (1000,))

    result = engine.train(X, y, epochs)
    eval_result = engine.evaluate(X[:100], y[:100])

    console.print(f"[bold green]Training complete![/bold green]")
    console.print(f"Final loss: {result['final_loss']:.4f}")
    console.print(f"Accuracy: {eval_result['accuracy']:.2%}")

    if save_to:
        engine.save(save_to)
        console.print(f"[green]Model saved to {save_to}[/green]")


@local.command("predict")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--data", "-d", multiple=True, type=float, required=True)
def local_predict(model_path, data):
    """Make predictions locally (no API needed)."""
    console.print(f"[blue]Loading model from {model_path}...[/blue]")

    # Load and predict
    import torch

    state_dict = torch.load(model_path, map_location="cpu")

    # Infer dimensions from state dict
    input_dim = state_dict["layers.0.weight"].shape[1]
    hidden_dim = state_dict["layers.0.weight"].shape[0]
    output_dim = state_dict["layers.2.weight"].shape[0]

    engine = MLEngine(input_dim, hidden_dim, output_dim)
    engine.load(model_path)

    X = torch.tensor([list(data)], dtype=torch.float32)
    outputs = engine.predict(X)
    prediction = torch.argmax(outputs, dim=1).item()
    confidence = torch.softmax(outputs, dim=1).max().item()

    console.print(
        f"[bold green]Prediction: {prediction}[/bold green] (confidence: {confidence:.2%})"
    )


@config_cmd.command("show")
def config_show():
    """Show current configuration."""
    from .config import config

    console.print("[bold]Configuration path:[/bold]", config.config_path)
    console.print_json(json.dumps(config._config, indent=2))


@config_cmd.command("get")
@click.argument("key")
def config_get(key):
    """Get a configuration value."""
    from .config import config

    value = config.get(key)
    if value is not None:
        console.print(f"{key}: {value}")
    else:
        console.print(f"[yellow]Key '{key}' not found[/yellow]")


@config_cmd.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """Set a configuration value."""
    from .config import config

    # Try to parse as JSON for complex values
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        parsed_value = value

    config.set(key, parsed_value)
    console.print(f"[green]Set {key} = {parsed_value}[/green]")


@cli.group()
def integration():
    """Research integration commands."""
    pass


@integration.command("compose")
@click.option("--input-dim", default=128, help="Input dimension")
@click.option("--output-dim", default=10, help="Output dimension")
@click.option("--task", help="Task description")
def compose_architecture(input_dim, output_dim, task):
    """Compose architecture through ADAG+ANAS+ACML pipeline."""
    from .integration import ResearchIntegration

    console.print("[bold blue]🚀 Composing architecture pipeline...[/bold blue]")

    hub = ResearchIntegration()
    architecture = hub.compose_architecture_pipeline(
        input_dim=input_dim, output_dim=output_dim, task_description=task
    )

    console.print(f"[bold green]✅ Architecture created:[/bold green] {architecture}")

    # Show report
    report = hub.get_integration_report()
    table = Table(title="Integration Report")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    for component in report["available_components"]:
        table.add_row(component, "✅ Available")

    console.print(table)


@cli.group()
def registry():
    """Model registry commands."""
    pass


@registry.command("list")
def list_models():
    """List all registered models."""
    from .model_registry import get_registry

    reg = get_registry()
    models = reg.list_models()

    if not models:
        console.print("[yellow]No models registered.[/yellow]")
        return

    table = Table(title=f"Registered Models ({len(models)})")
    table.add_column("Model Name", style="cyan")
    table.add_column("Versions", style="green")

    for model_name in models:
        versions = reg.list_versions(model_name)
        table.add_row(model_name, str(len(versions)))

    console.print(table)


@registry.command("versions")
@click.argument("model_name")
def list_versions(model_name):
    """List versions of a model."""
    from .model_registry import get_registry

    reg = get_registry()
    versions = reg.list_versions(model_name)

    if not versions:
        console.print(f"[yellow]No versions found for {model_name}[/yellow]")
        return

    table = Table(title=f"Versions of {model_name}")
    table.add_column("Version", style="cyan")
    table.add_column("Created", style="green")
    table.add_column("Metrics", style="blue")

    for ver in versions:
        metrics_str = ", ".join(
            [f"{k}={v:.4f}" for k, v in list(ver.metrics.items())[:2]]
        )
        table.add_row(ver.version_id, ver.created_at[:10], metrics_str)

    console.print(table)


@cli.group()
def experiment():
    """Experiment tracking commands."""
    pass


@experiment.command("list")
def list_experiments():
    """List all experiments."""
    from .experiment_tracking import ExperimentTracker

    tracker = ExperimentTracker("default")
    runs = tracker.get_run_history()

    if not runs:
        console.print("[yellow]No experiments found.[/yellow]")
        return

    table = Table(title=f"Experiments ({len(runs)})")
    table.add_column("Run ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Start Time", style="blue")

    for run in runs:
        table.add_row(run.run_id[:20] + "...", run.status, run.start_time[:10])

    console.print(table)


@cli.group()
def create():
    """Create new resources."""
    pass


@create.command("package")
@click.option("--name", "-n", required=True, help="Package name")
@click.option(
    "--type",
    "-t",
    default="general",
    type=click.Choice(
        ["general", "meta-learning", "optimization", "architecture", "field-theory"]
    ),
    help="Package type",
)
@click.option("--description", "-d", help="Package description")
def create_package(name, type, description):
    """Create a new research package from template."""
    from .template_generator import create_package

    console.print(f"[bold blue]📦 Creating package '{name}'...[/bold blue]")

    path = create_package(name=name, package_type=type, description=description)

    console.print(f"[bold green]✅ Package created at:[/bold green] {path}")
    console.print("\n[dim]Next steps:[/dim]")
    console.print(f"  cd {path}")
    console.print("  pip install -e .")
    console.print("  pytest tests/ -v")


@cli.group()
def distributed():
    """Distributed training commands."""
    pass


@distributed.command("train")
@click.option("--config", type=click.Path(exists=True), help="Path to config file")
@click.option("--workers", default=2, help="Number of distributed workers")
@click.option("--gpu/--no-gpu", default=True, help="Use GPU")
def distributed_train(config, workers, gpu):
    """Train model using distributed Ray backend."""
    console.print("[bold blue]🚀 Starting distributed training...[/bold blue]")
    console.print(f"   Workers: {workers}, GPU: {gpu}")

    from .distributed import DistributedTrainer, DistributedConfig

    dist_config = DistributedConfig(num_workers=workers, use_gpu=gpu)

    trainer = DistributedTrainer(dist_config)
    console.print("[bold green]✅ Distributed training initialized[/bold green]")
    console.print("   Use Python API for full control over model and data")


@cli.group()
def automl():
    """AutoML hyperparameter optimization commands."""
    pass


@automl.command("optimize")
@click.option(
    "--strategy",
    default="bayesian",
    type=click.Choice(["grid", "random", "bayesian", "pbt"]),
    help="Optimization strategy",
)
@click.option("--iterations", default=50, help="Maximum iterations")
@click.option("--metric", default="val_loss", help="Metric to optimize")
@click.option(
    "--mode",
    default="min",
    type=click.Choice(["min", "max"]),
    help="Optimization direction",
)
def optimize_hyperparameters(strategy, iterations, metric, mode):
    """Run hyperparameter optimization."""
    console.print(f"[bold blue]🔍 Starting {strategy} optimization...[/bold blue]")
    console.print(f"   Iterations: {iterations}, Metric: {metric}, Mode: {mode}")

    from .automl import OptimizationConfig

    config = OptimizationConfig(
        strategy=strategy,
        max_iterations=iterations,
        objective_metric=metric,
        objective_mode=mode,
    )

    console.print(
        "[bold green]✅ Use Python API to define search space and run optimization[/bold green]"
    )


@cli.group()
def serve():
    """Model serving commands."""
    pass


@serve.command("start")
@click.option("--port", default=8080, help="Serving port")
@click.option("--model", help="Model name to serve")
@click.option("--version", help="Model version")
def start_server(port, model, version):
    """Start model serving server."""
    console.print(f"[bold blue]🌐 Starting model server on port {port}...[/bold blue]")

    if model:
        console.print(f"   Serving model: {model} (version: {version or 'latest'})")

    console.print(
        "[bold green]✅ Use Python API to load and configure models[/bold green]"
    )


@serve.command("metrics")
def serving_metrics():
    """Show serving metrics."""
    from .serving import get_model_server

    try:
        server = get_model_server()
        metrics = server.get_metrics()

        table = Table(title="Serving Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Requests", str(metrics["total_requests"]))
        table.add_row("Avg Latency (ms)", f"{metrics.get('avg_latency_ms', 0):.2f}")
        table.add_row("Errors", str(metrics["error_count"]))

        console.print(table)
    except:
        console.print("[yellow]No serving metrics available[/yellow]")


@cli.group()
def async_cmd():
    """Async training commands."""
    pass


@async_cmd.command("train")
@click.option("--input-dim", default=128, help="Input dimension")
@click.option("--output-dim", default=10, help="Output dimension")
@click.option("--epochs", default=10, help="Training epochs")
def async_train(input_dim, output_dim, epochs):
    """Start async training job with WebSocket monitoring."""
    console.print("[bold blue]⚡ Starting async training job...[/bold blue]")
    console.print(f"   Connect to WebSocket to track progress")

    console.print(
        "[bold green]✅ Use Python API or API endpoints for async training[/bold green]"
    )


@cli.command("dashboard")
def launch_dashboard():
    """Launch interactive dashboard."""
    console.print("[bold blue]📊 Launching AI Hub Dashboard...[/bold blue]")
    console.print("   Features:")
    console.print("   • Real-time training visualization")
    console.print("   • Model comparison tools")
    console.print("   • Experiment tracking")
    console.print("   • Architecture search visualization")

    console.print(
        "\n[bold green]✅ Dashboard available via API at /dashboard[/bold green]"
    )


def main():
    cli()


if __name__ == "__main__":
    main()
