"""Fail-closed API-key authentication and model identifier validation."""
import os
import re
import secrets

from fastapi import HTTPException, Request, Security

from .request_quota import request_quota
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def require_api_key(request: Request, provided: str | None = Security(_api_key_header)) -> None:
    """Require a strong server-configured key for every sensitive API operation."""
    expected = os.environ.get("AI_HUB_API_KEY", "")
    if len(expected) < 32:
        raise HTTPException(
            status_code=503, detail="API authentication is not configured"
        )
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "APIKey"},
        )


    # Shared-key, single-process abuse control. These are bounded defaults;
    # invalid configuration is a deployment error and fails closed.
    try:
        request_limit = int(os.environ.get("AI_HUB_REQUESTS_PER_MINUTE", "600"))
        training_limit = int(os.environ.get("AI_HUB_TRAININGS_PER_MINUTE", "60"))
        if not (1 <= training_limit <= request_limit <= 10000):
            raise ValueError("quota settings out of range")
    except ValueError:
        raise HTTPException(status_code=503, detail="API quota configuration is invalid")

    wait = request_quota.check(
        expected,
        training=request.url.path in {"/train", "/train/supervised"},
        request_limit=request_limit,
        training_limit=training_limit,
    )
    if wait is not None:
        raise HTTPException(
            status_code=429,
            detail="API request quota exceeded",
            headers={"Retry-After": str(wait)},
        )


def validate_model_name(name: str) -> str:
    """Reject path separators, traversal components and ambiguous model names."""
    if not _MODEL_NAME.fullmatch(name):
        raise HTTPException(
            status_code=422,
            detail="Model name must be 1-64 ASCII letters, numbers, '-' or '_', "
                   "starting with a letter or number",
        )
    return name
