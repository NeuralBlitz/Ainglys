"""Input validation for the Streamlit supervised-learning upload workflow."""
from __future__ import annotations

import json

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
