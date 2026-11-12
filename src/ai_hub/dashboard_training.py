"""Input validation for the Streamlit supervised-learning upload workflow."""
from __future__ import annotations

import json
from urllib.parse import urlsplit

from pydantic import ValidationError

from .supervised import SupervisedTrainRequest

MAX_UPLOAD_BYTES = 1_000_000


def parse_supervised_upload(contents: bytes) -> SupervisedTrainRequest:
    """Safely parse a small JSON file using exactly the API training schema.

    The upload remains in memory and is never written to server filesystem.
    """
    if not isinstance(contents, bytes):
        raise ValueError("Dataset upload must be a JSON byte sequence")
    if not contents or len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError("Dataset JSON must be between 1 byte and 1 MB")
    try:
        parsed = json.loads(contents.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("Dataset JSON must be an object")
        return SupervisedTrainRequest.model_validate(parsed)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Invalid supervised dataset: {exc}") from exc


def safe_training_api_base(url: str) -> str:
    """Do not transmit raw data or shared API secrets over remote plain HTTP."""
    if not isinstance(url, str):
        raise ValueError("API URL must be a string")
    value = url.rstrip("/")
    parsed = urlsplit(value)
    if (parsed.scheme not in {"https", "http"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("Use a base HTTPS URL without credentials, paths or queries")
    if parsed.scheme == "http" and parsed.hostname not in {
        "localhost", "127.0.0.1", "::1", "api",
    }:
        raise ValueError("Unencrypted API access is permitted only on localhost or the internal api service")
    return value
