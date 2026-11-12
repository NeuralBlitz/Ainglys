"""Validation regression tests for Streamlit dataset ingestion."""
import json

import pytest

from ai_hub.dashboard_training import MAX_UPLOAD_BYTES, parse_supervised_upload


def dataset():
    return {
        "features": [[i / 10.0, (i % 3) * 1.0] for i in range(40)],
        "labels": [i % 2 for i in range(40)],
        "epochs": 2,
    }


def test_accepts_valid_dataset_and_defaults():
    result = parse_supervised_upload(json.dumps(dataset()).encode())
    assert len(result.features) == 40
    assert result.labels == dataset()["labels"]
    assert result.epochs == 2
    assert result.validation_fraction == 0.2


@pytest.mark.parametrize("contents", [
    b"", b"not json", b"[]", b"null", b'{"features": []}',
    b"\xff", b"x" * (MAX_UPLOAD_BYTES + 1),
])
def test_rejects_invalid_upload(contents):
    with pytest.raises(ValueError):
        parse_supervised_upload(contents)


def test_rejects_invalid_class_structure():
    payload = dataset()
    payload["labels"] = [0] * 40
    with pytest.raises(ValueError):
        parse_supervised_upload(json.dumps(payload).encode())


def test_rejects_mismatched_rows():
    payload = dataset()
    payload["features"][0] = [0.0]
    with pytest.raises(ValueError):
        parse_supervised_upload(json.dumps(payload).encode())


def test_does_not_write_upload_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parse_supervised_upload(json.dumps(dataset()).encode())
    assert not list(tmp_path.iterdir())
