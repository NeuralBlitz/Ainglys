"""Fail-closed API-key authentication and model identifier validation."""
import os
import re
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def require_api_key(provided: str | None = Security(_api_key_header)) -> None:
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


def validate_model_name(name: str) -> str:
    """Reject path separators, traversal components and ambiguous model names."""
    if not _MODEL_NAME.fullmatch(name):
        raise HTTPException(
            status_code=422,
            detail="Model name must be 1-64 ASCII letters, numbers, '-' or '_', "
                   "starting with a letter or number",
        )
    return name
