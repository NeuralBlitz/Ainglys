import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any
import torch
import json
import math
import threading
import logging
from .ml_engine import MLEngine
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


@app.get("/")
def root():
    return {
        "message": "AI Hub API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": ["/train", "/predict", "/status", "/save", "/load", "/models"],
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
        model_name = validate_model_name(request.model_name)
        model_path = MODELS_DIR / f"{model_name}.pt"
        engine.save(str(model_path))

        metadata = {
            "model_name": request.model_name,
            "input_dim": engine.model.layers[0].in_features,
            "hidden_dim": engine.model.layers[0].out_features,
            "output_dim": engine.model.layers[2].out_features,
        }

        metadata_path = MODELS_DIR / f"{model_name}.json"
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
    try:
        validate_model_name(model_name)
        model_path = MODELS_DIR / f"{model_name}.pt"
        metadata_path = MODELS_DIR / f"{model_name}.json"

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        engine = MLEngine(
            metadata["input_dim"], metadata["hidden_dim"], metadata["output_dim"]
        )
        engine.load(str(model_path))

        return {"status": "loaded", "model_name": model_name, "metadata": metadata}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
