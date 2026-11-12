"""Validation regression tests for Streamlit dataset ingestion."""
import json

import pytest

from ai_hub.dashboard_training import MAX_UPLOAD_BYTES, parse_supervised_upload, safe_training_api_base


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


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000", "http://localhost:8000",
    "http://api:8000", "https://training.example.com",
])
def test_dashboard_allows_local_or_https_api(url):
    assert safe_training_api_base(url) == url


@pytest.mark.parametrize("url", [
    "http://training.example.com",
    "http://api:8000@evil.example.com",
    "file:///tmp/data", "http://localhost:8000/path",
    "http://localhost:8000?token=secret", "http://localhost:8000#redirect",
])
def test_dashboard_refuses_insecure_api_url(url):
    with pytest.raises(ValueError):
        safe_training_api_base(url)
