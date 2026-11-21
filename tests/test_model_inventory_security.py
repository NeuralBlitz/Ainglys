"""Regression coverage for model inventory and deletion filesystem boundaries."""
import json

import pytest
from fastapi.testclient import TestClient

import ai_hub.api as api_module
from ai_hub.api import app
from ai_hub.request_quota import request_quota, principal_request_quotas


KEY = "model-inventory-test-credential-with-32-characters"


@pytest.fixture
def model_api(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setattr(api_module, "MODELS_DIR", model_dir)
    monkeypatch.delenv("AI_HUB_API_KEYS_JSON", raising=False)
    monkeypatch.setenv("AI_HUB_API_KEY", KEY)
    monkeypatch.setenv("AI_HUB_REQUESTS_PER_MINUTE", "600")
    monkeypatch.setenv("AI_HUB_TRAININGS_PER_MINUTE", "60")
    request_quota.reset()
    principal_request_quotas.reset()
    yield model_dir, TestClient(app, headers={"X-API-Key": KEY})
    request_quota.reset()
    principal_request_quotas.reset()


def test_inventory_preserves_valid_metadata(model_api):
    model_dir, client = model_api
    (model_dir / "good.pt").write_bytes(b"sample weights")
    sidecar = {
        "model_name": "good", "input_dim": 3,
        "hidden_dim": 4, "output_dim": 2,
    }
    (model_dir / "good.json").write_text(json.dumps(sidecar), encoding="utf-8")

    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == {
        "models": [{"name": "good", "path": str(model_dir / "good.pt"),
                    "metadata": sidecar}],
        "count": 1,
    }


@pytest.mark.parametrize("bad_content", [
    "{this is not json}",
    "x" * (64 * 1024 + 1),
    json.dumps({"model_name": "wrong", "input_dim": 3, "hidden_dim": 4, "output_dim": 2}),
])
def test_invalid_or_oversized_sidecars_do_not_crash_listing(model_api, bad_content):
    model_dir, client = model_api
    (model_dir / "safe.pt").write_bytes(b"sample weights")
    (model_dir / "safe.json").write_text(bad_content, encoding="utf-8")

    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == {
        "models": [{"name": "safe", "path": str(model_dir / "safe.pt")}],
        "count": 1,
    }


def test_inventory_ignores_symlinks_and_invalid_model_names(model_api, tmp_path):
    model_dir, client = model_api
    outside_model = tmp_path / "private.pt"
    outside_model.write_bytes(b"private artifact")
    outside_metadata = tmp_path / "private.json"
    outside_metadata.write_text('{"secret":"outside"}', encoding="utf-8")

    (model_dir / "linked.pt").symlink_to(outside_model)
    (model_dir / "linked.json").symlink_to(outside_metadata)
    (model_dir / "metadata_link.pt").write_bytes(b"sample weights")
    (model_dir / "metadata_link.json").symlink_to(outside_metadata)
    (model_dir / "invalid.name.pt").write_bytes(b"sample weights")

    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == {"models": [], "count": 0}
    assert outside_model.read_bytes() == b"private artifact"


def test_delete_refuses_symlinked_model_and_metadata(model_api, tmp_path):
    model_dir, client = model_api
    outside_model = tmp_path / "private.pt"
    outside_model.write_bytes(b"private artifact")
    (model_dir / "linked.pt").symlink_to(outside_model)

    response = client.delete("/models/linked")
    assert response.status_code == 422
    assert (model_dir / "linked.pt").is_symlink()
    assert outside_model.read_bytes() == b"private artifact"

    safe = model_dir / "safe.pt"
    safe.write_bytes(b"sample weights")
    outside_metadata = tmp_path / "private.json"
    outside_metadata.write_text('{"secret":"outside"}', encoding="utf-8")
    (model_dir / "safe.json").symlink_to(outside_metadata)

    response = client.delete("/models/safe")
    assert response.status_code == 422
    assert safe.read_bytes() == b"sample weights"
    assert outside_metadata.is_file()


def test_delete_regular_model_and_metadata(model_api):
    model_dir, client = model_api
    model = model_dir / "regular.pt"
    sidecar = model_dir / "regular.json"
    model.write_bytes(b"sample weights")
    sidecar.write_text("{}", encoding="utf-8")

    response = client.delete("/models/regular")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"
    assert not model.exists()
    assert not sidecar.exists()
