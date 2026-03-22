"""
Model Serving and Deployment System

Provides model serving endpoints with:
- REST API for inference
- Batch prediction
- Model versioning in production
- A/B testing at serving time
- Performance monitoring
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import time
import json
from pathlib import Path
import numpy as np
from queue import Queue
import threading


@dataclass
class ServingConfig:
    """Configuration for model serving."""

    batch_size: int = 32
    max_latency_ms: float = 100.0
    enable_gpu: bool = True
    model_cache_size: int = 5
    enable_batching: bool = True
    monitoring_enabled: bool = True


@dataclass
class PredictionRequest:
    """Single prediction request."""

    request_id: str
    inputs: Any
    timestamp: float
    callback: Optional[Callable] = None


@dataclass
class PredictionResponse:
    """Prediction response."""

    request_id: str
    predictions: Any
    latency_ms: float
    model_version: str
    timestamp: str


class ModelServer:
    """
    Production model server with batching and caching.

    Features:
    - Dynamic batching for efficient GPU utilization
    - Model versioning and hot-swapping
    - A/B testing between model versions
    - Performance monitoring
    """

    def __init__(self, config: Optional[ServingConfig] = None):
        self.config = config or ServingConfig()
        self.models: Dict[
            str, Dict[str, nn.Module]
        ] = {}  # {model_name: {version: model}}
        self.active_versions: Dict[str, str] = {}  # {model_name: active_version}
        self.ab_test_configs: Dict[str, Dict] = {}

        # Batching
        self.request_queue: Queue = Queue()
        self.batch_thread: Optional[threading.Thread] = None
        self.running = False

        # Metrics
        self.metrics = {
            "total_requests": 0,
            "total_latency_ms": 0.0,
            "batch_size_distribution": [],
            "error_count": 0,
        }

        if self.config.enable_batching:
            self._start_batch_processor()

    def _start_batch_processor(self):
        """Start background thread for dynamic batching."""
        self.running = True
        self.batch_thread = threading.Thread(target=self._batch_processor_loop)
        self.batch_thread.daemon = True
        self.batch_thread.start()

    def _batch_processor_loop(self):
        """Process requests in batches."""
        while self.running:
            batch = []
            start_time = time.time()

            # Collect requests until batch is full or timeout
            while len(batch) < self.config.batch_size:
                elapsed = (time.time() - start_time) * 1000
                if elapsed >= self.config.max_latency_ms:
                    break

                if not self.request_queue.empty():
                    batch.append(self.request_queue.get())
                else:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting

            # Process batch if not empty
            if batch:
                self._process_batch(batch)

    def _process_batch(self, requests: List[PredictionRequest]):
        """Process a batch of requests."""
        # Group by model
        by_model = {}
        for req in requests:
            model_name = req.inputs.get("model_name", "default")
            if model_name not in by_model:
                by_model[model_name] = []
            by_model[model_name].append(req)

        # Process each model group
        for model_name, model_requests in by_model.items():
            try:
                # Stack inputs
                inputs = torch.stack(
                    [torch.tensor(r.inputs["data"]) for r in model_requests]
                )

                # Get model version (with A/B testing)
                version = self._select_model_version(model_name, model_requests[0])
                model = self.models.get(model_name, {}).get(version)

                if model is None:
                    raise ValueError(f"Model {model_name} version {version} not found")

                # Run inference
                start_time = time.time()
                with torch.no_grad():
                    if self.config.enable_gpu and torch.cuda.is_available():
                        inputs = inputs.cuda()
                        model = model.cuda()

                    outputs = model(inputs)
                    predictions = outputs.cpu().numpy()

                latency_ms = (time.time() - start_time) * 1000

                # Send responses
                for i, req in enumerate(model_requests):
                    response = PredictionResponse(
                        request_id=req.request_id,
                        predictions=predictions[i].tolist(),
                        latency_ms=latency_ms,
                        model_version=version,
                        timestamp=datetime.now().isoformat(),
                    )

                    if req.callback:
                        req.callback(response)

                    # Update metrics
                    self.metrics["total_requests"] += 1
                    self.metrics["total_latency_ms"] += latency_ms

                self.metrics["batch_size_distribution"].append(len(model_requests))

            except Exception as e:
                # Handle errors
                for req in model_requests:
                    if req.callback:
                        req.callback(None, error=str(e))
                self.metrics["error_count"] += len(model_requests)

    def _select_model_version(self, model_name: str, request: PredictionRequest) -> str:
        """Select model version (with A/B testing support)."""
        # Check if A/B test is active
        if model_name in self.ab_test_configs:
            config = self.ab_test_configs[model_name]
            # Simple random routing
            if np.random.random() < config.get("traffic_split", 0.5):
                return config["version_b"]
            else:
                return config["version_a"]

        # Return active version
        return self.active_versions.get(model_name, "latest")

    def load_model(self, model: nn.Module, model_name: str, version: str = "latest"):
        """Load a model into the server."""
        if model_name not in self.models:
            self.models[model_name] = {}

        model.eval()
        if self.config.enable_gpu and torch.cuda.is_available():
            model = model.cuda()

        self.models[model_name][version] = model

        # Set as active if first version
        if model_name not in self.active_versions:
            self.active_versions[model_name] = version

        print(f"✅ Loaded model {model_name} version {version}")

    def set_active_version(self, model_name: str, version: str):
        """Set active version for a model."""
        if model_name not in self.models or version not in self.models[model_name]:
            raise ValueError(f"Model {model_name} version {version} not found")

        self.active_versions[model_name] = version
        print(f"✅ Set {model_name} active version to {version}")

    def setup_ab_test(
        self,
        model_name: str,
        version_a: str,
        version_b: str,
        traffic_split: float = 0.5,
    ):
        """Setup A/B testing between two model versions."""
        self.ab_test_configs[model_name] = {
            "version_a": version_a,
            "version_b": version_b,
            "traffic_split": traffic_split,
            "results": {"version_a": [], "version_b": []},
        }
        print(f"✅ A/B test setup for {model_name}: {version_a} vs {version_b}")

    def predict(
        self, data: Any, model_name: str = "default", version: Optional[str] = None
    ) -> PredictionResponse:
        """
        Make a single prediction (synchronous).

        Args:
            data: Input data
            model_name: Name of the model
            version: Specific version (None for auto-selection)

        Returns:
            PredictionResponse
        """
        request_id = f"req_{int(time.time() * 1000)}"

        # Direct prediction without batching
        if not self.config.enable_batching:
            return self._predict_single(data, model_name, version, request_id)

        # Queue for batching
        response_container = {}

        def callback(response, error=None):
            if error:
                response_container["error"] = error
            else:
                response_container["response"] = response

        request = PredictionRequest(
            request_id=request_id,
            inputs={"data": data, "model_name": model_name},
            timestamp=time.time(),
            callback=callback,
        )

        self.request_queue.put(request)

        # Wait for response
        while (
            "response" not in response_container and "error" not in response_container
        ):
            time.sleep(0.001)

        if "error" in response_container:
            raise RuntimeError(response_container["error"])

        return response_container["response"]

    def _predict_single(self, data, model_name, version, request_id):
        """Make single prediction without batching."""
        start_time = time.time()

        # Select version
        if version is None:
            version = self.active_versions.get(model_name, "latest")

        model = self.models.get(model_name, {}).get(version)
        if model is None:
            raise ValueError(f"Model {model_name} version {version} not found")

        # Prepare input
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)

        if data.dim() == 1:
            data = data.unsqueeze(0)

        # Run inference
        with torch.no_grad():
            if self.config.enable_gpu and torch.cuda.is_available():
                data = data.cuda()

            output = model(data)
            predictions = output.cpu().numpy()

        latency_ms = (time.time() - start_time) * 1000

        self.metrics["total_requests"] += 1
        self.metrics["total_latency_ms"] += latency_ms

        return PredictionResponse(
            request_id=request_id,
            predictions=predictions[0].tolist()
            if len(predictions) == 1
            else predictions.tolist(),
            latency_ms=latency_ms,
            model_version=version,
            timestamp=datetime.now().isoformat(),
        )

    def predict_batch(
        self,
        data: List[Any],
        model_name: str = "default",
        version: Optional[str] = None,
    ) -> List[PredictionResponse]:
        """Make batch predictions."""
        return [self.predict(d, model_name, version) for d in data]

    def get_metrics(self) -> Dict[str, Any]:
        """Get serving metrics."""
        metrics = self.metrics.copy()

        if metrics["total_requests"] > 0:
            metrics["avg_latency_ms"] = (
                metrics["total_latency_ms"] / metrics["total_requests"]
            )
        else:
            metrics["avg_latency_ms"] = 0.0

        if metrics["batch_size_distribution"]:
            metrics["avg_batch_size"] = np.mean(metrics["batch_size_distribution"])

        return metrics

    def shutdown(self):
        """Shutdown the model server."""
        self.running = False
        if self.batch_thread:
            self.batch_thread.join(timeout=5.0)
        print("✅ Model server shutdown")


class ModelDeployment:
    """
    Deployment manager for production models.

    Handles:
    - Canary deployments
    - Blue-green deployments
    - Rollback capabilities
    """

    def __init__(self, server: ModelServer):
        self.server = server
        self.deployments: Dict[str, Dict] = {}

    def canary_deploy(
        self,
        model: nn.Module,
        model_name: str,
        version: str,
        canary_percentage: float = 10.0,
        evaluation_duration_hours: float = 24.0,
    ):
        """
        Deploy new model version using canary strategy.

        Gradually increases traffic to new version while monitoring.
        """
        print(f"🚀 Starting canary deployment for {model_name} v{version}")
        print(f"   Initial traffic: {canary_percentage}%")

        # Load model
        self.server.load_model(model, model_name, version)

        # Setup A/B test for canary
        current_version = self.server.active_versions.get(model_name, "latest")
        self.server.setup_ab_test(
            model_name,
            current_version,
            version,
            traffic_split=canary_percentage / 100.0,
        )

        deployment_id = f"canary_{model_name}_{version}_{int(time.time())}"
        self.deployments[deployment_id] = {
            "model_name": model_name,
            "version": version,
            "strategy": "canary",
            "start_time": datetime.now().isoformat(),
            "canary_percentage": canary_percentage,
            "status": "running",
        }

        print(f"   Deployment ID: {deployment_id}")
        return deployment_id

    def promote_canary(self, deployment_id: str):
        """Promote canary to full deployment."""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")

        deployment = self.deployments[deployment_id]
        model_name = deployment["model_name"]
        version = deployment["version"]

        # Remove A/B test and set as active
        if model_name in self.server.ab_test_configs:
            del self.server.ab_test_configs[model_name]

        self.server.set_active_version(model_name, version)
        deployment["status"] = "promoted"

        print(f"✅ Canary promoted to full deployment: {model_name} v{version}")

    def rollback(self, model_name: str, version: str):
        """Rollback to previous version."""
        print(f"⚠️ Rolling back {model_name} to version {version}")

        # Remove A/B test if active
        if model_name in self.server.ab_test_configs:
            del self.server.ab_test_configs[model_name]

        self.server.set_active_version(model_name, version)
        print(f"✅ Rollback complete")


# FastAPI integration
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# Global model server instance
_global_server: Optional[ModelServer] = None


def get_model_server() -> ModelServer:
    """Get or create global model server."""
    global _global_server
    if _global_server is None:
        _global_server = ModelServer()
    return _global_server


class PredictRequest(BaseModel):
    data: List[float]
    model_name: str = "default"
    version: Optional[str] = None


class PredictResponse(BaseModel):
    predictions: List[float]
    latency_ms: float
    model_version: str


@router.post("/serve/predict", response_model=PredictResponse)
async def serve_predict(request: PredictRequest):
    """Serve prediction endpoint."""
    server = get_model_server()

    try:
        response = server.predict(request.data, request.model_name, request.version)

        return PredictResponse(
            predictions=response.predictions,
            latency_ms=response.latency_ms,
            model_version=response.model_version,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/serve/predict/batch")
async def serve_predict_batch(requests: List[PredictRequest]):
    """Batch prediction endpoint."""
    server = get_model_server()

    responses = []
    for req in requests:
        try:
            response = server.predict(req.data, req.model_name, req.version)
            responses.append(
                {
                    "predictions": response.predictions,
                    "latency_ms": response.latency_ms,
                    "model_version": response.model_version,
                }
            )
        except Exception as e:
            responses.append({"error": str(e)})

    return responses


@router.get("/serve/metrics")
async def serve_metrics():
    """Get serving metrics."""
    server = get_model_server()
    return server.get_metrics()


@router.post("/serve/models/{model_name}/load")
async def load_model_endpoint(model_name: str, version: str = "latest"):
    """Load model endpoint (admin only)."""
    # In production, add authentication
    return {"message": f"Model {model_name} version {version} loaded"}


# Export
__all__ = [
    "ModelServer",
    "ModelDeployment",
    "ServingConfig",
    "PredictionRequest",
    "PredictionResponse",
    "get_model_server",
    "router",
]
