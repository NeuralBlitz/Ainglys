import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator, ValidationError
from typing import List, Optional, Dict, Any
import torch
import json
import math
import threading
import logging
from .ml_engine import MLEngine
from .supervised import SupervisedTrainRequest, stratified_holdout_indices
from .api_auth import require_api_key, validate_model_name
from pathlib import Path

app = FastAPI(title="AI Hub API", version="0.1.0")
logger = logging.getLogger(__name__)
_training_lock = threading.Lock()

engine: Optional[MLEngine] = None
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


class TrainRequest(BaseModel):
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 10
    epochs: int = 10
    learning_rate: float = 0.001

    @field_validator("input_dim", "hidden_dim", "output_dim", "epochs")
    @classmethod
    def bounded_training(cls, value, info):
        limits = {"input_dim": 512, "hidden_dim": 1024, "output_dim": 128, "epochs": 20}
        if not 1 <= value <= limits[info.field_name]:
            raise ValueError(f"{info.field_name} must be between 1 and {limits[info.field_name]}")
        return value

    @field_validator("learning_rate")
    @classmethod
    def bounded_learning_rate(cls, value):
        if not 1e-7 <= value <= 1:
            raise ValueError("learning_rate must be between 1e-7 and 1")
        return value


class PredictRequest(BaseModel):
    data: List[List[float]]

    @field_validator("data")
    @classmethod
    def data_not_empty(cls, value):
        if not value or len(value) > 256 or any(not row or len(row) > 512 for row in value):
            raise ValueError("data must contain 1-256 rows of 1-512 features")
        if len({len(row) for row in value}) != 1:
            raise ValueError("every row must contain the same number of features")
        if any(not math.isfinite(entry) for row in value for entry in row):
            raise ValueError("prediction features must be finite")
        return value


class SaveRequest(BaseModel):
    model_name: str = "model"


_MAX_MODEL_BYTES = 256 * 1024 * 1024
_MAX_METADATA_BYTES = 64 * 1024


def _model_artifacts(model_name: str):
    """Confine model artifact operations to regular, named files."""
    name = validate_model_name(model_name)
    model_path = MODELS_DIR / f"{name}.pt"
    metadata_path = MODELS_DIR / f"{name}.json"
    if model_path.is_symlink() or metadata_path.is_symlink():
        raise HTTPException(status_code=422, detail="Model artifact symlinks are not permitted")
    return model_path, metadata_path


def _load_model_metadata(name: str, metadata_path: Path) -> dict:
    if not metadata_path.is_file():
        raise HTTPException(status_code=404, detail="Model metadata not found")
    if metadata_path.stat().st_size > _MAX_METADATA_BYTES:
        raise HTTPException(status_code=422, detail="Model metadata exceeds allowed size")
    try:
        with metadata_path.open(encoding="utf-8") as source:
            metadata = json.load(source)
        if not isinstance(metadata, dict) or any(
            key not in metadata for key in ("input_dim", "hidden_dim", "output_dim")
        ):
            raise ValueError("Required model dimensions missing")
        if metadata.get("model_name", name) != name:
            raise ValueError("Model metadata does not match identifier")
        request = TrainRequest.model_validate(metadata)
        return {
            "model_name": name,
            "input_dim": request.input_dim,
            "hidden_dim": request.hidden_dim,
            "output_dim": request.output_dim,
        }
    except (ValueError, TypeError, json.JSONDecodeError, ValidationError):
        raise HTTPException(status_code=422, detail="Invalid model metadata")


@app.get("/")
def root():
    return {
        "message": "AI Hub API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": ["/train", "/train/supervised", "/predict", "/status", "/save", "/load", "/models"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status", dependencies=[Depends(require_api_key)])
def status():
    return {
        "trained": engine is not None and engine.is_trained,
        "model_loaded": engine is not None,
        "models_available": list(MODELS_DIR.glob("*.pt")),
    }


@app.post("/train", dependencies=[Depends(require_api_key)])
def train(request: TrainRequest):
    global engine
    if not _training_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Training already in progress")
    try:
        # Do not replace the current working model until training succeeds.
        candidate = MLEngine(request.input_dim, request.hidden_dim, request.output_dim)
        X = torch.randn(1000, request.input_dim)
        y = torch.randint(0, request.output_dim, (1000,))

        result = candidate.train(X, y, request.epochs, learning_rate=request.learning_rate)
        eval_result = candidate.evaluate(X[:100], y[:100])
        engine = candidate

        return {
            "status": "trained",
            "data_source": "synthetic_demo_random_labels",
            "model_info": {
                "input_dim": request.input_dim,
                "hidden_dim": request.hidden_dim,
                "output_dim": request.output_dim,
            },
            "training": result,
            "evaluation": eval_result,
        }
    except Exception:
        logger.exception("Model training failed")
        raise HTTPException(status_code=500, detail="Training failed")
    finally:
        _training_lock.release()


@app.post("/train/supervised", dependencies=[Depends(require_api_key)])
def train_supervised(request: SupervisedTrainRequest):
    """Train a small classifier on supplied numeric data and evaluate a holdout.

    Validation records are never passed to the training optimizer; the
    synthetic-demo endpoint remains available separately at POST /train.
    """
    global engine
    if not _training_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Training already in progress")
    try:
        train_indices, val_indices = stratified_holdout_indices(
            request.labels, request.validation_fraction, request.seed
        )
        inputs = torch.tensor(request.features, dtype=torch.float32)
        targets = torch.tensor(request.labels, dtype=torch.long)
        train_x = inputs[train_indices]
        train_y = targets[train_indices]
        validation_x = inputs[val_indices]
        validation_y = targets[val_indices]
        classes = len(set(request.labels))
        feature_dim = inputs.shape[1]

        # Seed model initialization and training, without leaving changes in
        # the process-global torch CPU/GPU random state after the request.
        rng_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(request.seed)
            if rng_devices:
                torch.cuda.manual_seed_all(request.seed)
            candidate = MLEngine(feature_dim, request.hidden_dim, classes)
            training = candidate.train(
                train_x, train_y,
                epochs=request.epochs,
                learning_rate=request.learning_rate,
            )

        validation = candidate.evaluate(validation_x, validation_y)
        with torch.no_grad():
            logits = candidate.predict(validation_x)
            validation_loss = torch.nn.functional.cross_entropy(
                logits, validation_y.to(logits.device)
            ).item()
        # Reject numerically unstable outcomes before publishing any model.
        quality_metrics = (
            training.get("final_loss"), validation.get("accuracy"), validation_loss
        )
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in quality_metrics
        ):
            raise ValueError("Training or validation produced nonfinite metrics")

        # Commit only after both training and held-out evaluation succeed.
        engine = candidate
        return {
            "status": "trained",
            "data_source": "user_provided_supervised",
            "model_info": {
                "input_dim": feature_dim,
                "hidden_dim": request.hidden_dim,
                "output_dim": classes,
            },
            "dataset": {
                "samples": len(request.labels),
                "training_samples": len(train_indices),
                "validation_samples": len(val_indices),
                "classes": classes,
                "validation_fraction_requested": request.validation_fraction,
                "random_seed": request.seed,
                "split_strategy": "stratified_holdout",
            },
            "training": training,
            "validation": {
                "accuracy": validation["accuracy"],
                "cross_entropy": validation_loss,
            },
        }
    except Exception:
        logger.exception("Supervised model training failed")
        raise HTTPException(status_code=500, detail="Supervised training failed")
    finally:
        _training_lock.release()


@app.post("/predict", dependencies=[Depends(require_api_key)])
def predict(request: PredictRequest):
    global engine
    if engine is None or not engine.is_trained:
        raise HTTPException(
            status_code=400, detail="Model not trained. Call /train first."
        )

    try:
        X = torch.tensor(request.data, dtype=torch.float32)
        if X.shape[1] != engine.model.layers[0].in_features:
            raise HTTPException(
                status_code=400,
                detail=f"Expected input dimension {engine.model.layers[0].in_features}, got {X.shape[1]}",
            )

        outputs = engine.predict(X)
        predictions = torch.argmax(outputs, dim=1).tolist()
        probabilities = torch.softmax(outputs, dim=1).tolist()

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "count": len(predictions),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save", dependencies=[Depends(require_api_key)])
def save_model(request: SaveRequest):
    global engine
    if engine is None or not engine.is_trained:
        raise HTTPException(status_code=400, detail="No trained model to save")

    try:
        model_path, metadata_path = _model_artifacts(request.model_name)
        model_name = request.model_name
        engine.save(str(model_path))

        metadata = {
            "model_name": request.model_name,
            "input_dim": engine.model.layers[0].in_features,
            "hidden_dim": engine.model.layers[0].out_features,
            "output_dim": engine.model.layers[2].out_features,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {"status": "saved", "model_path": str(model_path)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load/{model_name}", dependencies=[Depends(require_api_key)])
def load_model(model_name: str):
    global engine
    if not _training_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Model update already in progress")
    try:
        model_path, metadata_path = _model_artifacts(model_name)
        if not model_path.is_file():
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        if model_path.stat().st_size > _MAX_MODEL_BYTES:
            raise HTTPException(status_code=422, detail="Model artifact exceeds allowed size")

        metadata = _load_model_metadata(model_name, metadata_path)
        candidate = MLEngine(
            metadata["input_dim"], metadata["hidden_dim"], metadata["output_dim"]
        )
        candidate.load(str(model_path))
        # Preserve the old, functioning model if deserialization fails.
        engine = candidate
        return {"status": "loaded", "model_name": model_name, "metadata": metadata}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Model loading failed")
        raise HTTPException(status_code=422, detail="Invalid or incompatible model artifact")
    finally:
        _training_lock.release()


@app.get("/models", dependencies=[Depends(require_api_key)])
def list_models():
    models = []
    for model_file in MODELS_DIR.glob("*.pt"):
        model_name = model_file.stem
        metadata_path = MODELS_DIR / f"{model_name}.json"

        model_info = {"name": model_name, "path": str(model_file)}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                model_info["metadata"] = json.load(f)
        models.append(model_info)

    return {"models": models, "count": len(models)}


@app.delete("/models/{model_name}", dependencies=[Depends(require_api_key)])
def delete_model(model_name: str):
    try:
        validate_model_name(model_name)
        model_path = MODELS_DIR / f"{model_name}.pt"
        metadata_path = MODELS_DIR / f"{model_name}.json"

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        model_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()

        return {"status": "deleted", "model_name": model_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def create_app() -> FastAPI:
    return app
