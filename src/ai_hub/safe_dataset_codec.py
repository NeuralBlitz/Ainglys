"""Non-executable dataset snapshot codec for the Ainglys dataset repository.

Never load Python pickle or NumPy object arrays. Formats are intentionally
limited to numerical ndarrays, pandas DataFrames, and JSON-compatible values.
"""
from __future__ import annotations

from io import BytesIO, StringIO
import json
import math
from typing import Any

import numpy as np
import pandas as pd


MAX_DATASET_BYTES = 32 * 1024 * 1024
_FORMAT_SUFFIXES = {"npy": ".npy", "dataframe-json": ".frame.json", "json": ".json"}


def file_suffix(format_name: str) -> str:
    try:
        return _FORMAT_SUFFIXES[format_name]
    except KeyError as exc:
        raise ValueError("Unsupported or unsafe dataset storage format") from exc


def encode_dataset(data: Any) -> tuple[bytes, str]:
    """Serialize supported data without executing or pickling Python objects."""
    if isinstance(data, np.ndarray):
        if data.dtype.hasobject:
            raise TypeError("Object-valued NumPy arrays cannot be safely serialized")
        buffer = BytesIO()
        np.save(buffer, data, allow_pickle=False)
        payload, kind = buffer.getvalue(), "npy"
    elif isinstance(data, pd.DataFrame):
        if any(not isinstance(col, str) for col in data.columns):
            raise TypeError("DataFrame column labels must be strings")
        # This intentionally treats a DataFrame as a tabular dataset;
        # original dtype metadata is not guaranteed to round-trip exactly.
        payload = data.to_json(orient="split", date_format="iso").encode("utf-8")
        kind = "dataframe-json"
    elif isinstance(data, (dict, list, str, int, float, bool, type(None))):
        payload = json.dumps(
            data, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        kind = "json"
    else:
        raise TypeError("Unsupported dataset type; use ndarray, DataFrame or JSON data")

    if not payload or len(payload) > MAX_DATASET_BYTES:
        raise ValueError("Serialized dataset exceeds the 32 MiB limit")
    return payload, kind


def decode_dataset(payload: bytes, format_name: str) -> Any:
    """Deserialize only explicitly allowlisted formats, within strict size caps."""
    file_suffix(format_name)
    if not payload or len(payload) > MAX_DATASET_BYTES:
        raise ValueError("Dataset snapshot is missing or exceeds the allowed size")
    if format_name == "npy":
        # NumPy's .npy loader avoids arbitrary unpickling when this is False.
        data = np.load(BytesIO(payload), allow_pickle=False, max_header_size=10_000)
        if not isinstance(data, np.ndarray) or data.dtype.hasobject:
            raise ValueError("Invalid NumPy snapshot")
        return data
    if format_name == "dataframe-json":
        obj = json.loads(payload.decode("utf-8"))
        if not isinstance(obj, dict) or set(obj) != {"columns", "index", "data"}:
            raise ValueError("Malformed DataFrame snapshot")
        return pd.read_json(StringIO(payload.decode("utf-8")), orient="split")
    return json.loads(payload.decode("utf-8"))
