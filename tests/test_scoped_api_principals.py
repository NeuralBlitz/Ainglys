"""Scoped API-key integration, revocation and quota isolation tests."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os

import pytest
import ai_hub.api as api_module
from fastapi.testclient import TestClient

from ai_hub.api import app
from ai_hub.api_principals import (
    APIPrincipal,
    IdentityConfigError,
    authenticate,
    load_keyring,
    required_scope,
    token_sha256,
)
from ai_hub.request_quota import PrincipalRequestQuotas, principal_request_quotas, request_quota


READER_KEY = "test-reader-scoped-principal-key-32-plus-characters"
READER_TWO_KEY = "test-reader-second-principal-key-32-plus-characters"
TRAIN_KEY = "test-trainer-scoped-principal-key-32-plus-characters"
PREDICT_KEY = "test-predictor-scoped-principal-key-32-plus-characters"
WRITER_KEY = "test-writer-scoped-principal-key-32-plus-characters"
LEGACY_KEY = "legacy-master-api-key-with-32-plus-characters"


def record(client_id, key, scopes):
    return {"id": client_id, "sha256": token_sha256(key), "scopes": scopes}


def base_registry():
    return [
        record("reader", READER_KEY, ["read"]),
        record("reader-2", READER_TWO_KEY, ["read"]),
        record("trainer", TRAIN_KEY, ["train"]),
        record("predictor", PREDICT_KEY, ["predict"]),
        record("writer", WRITER_KEY, ["models:write"]),
    ]


@pytest.fixture
def scoped_env(monkeypatch):
    monkeypatch.setenv("AI_HUB_API_KEY", LEGACY_KEY)
    monkeypatch.setenv("AI_HUB_API_KEYS_JSON", json.dumps(base_registry()))
    monkeypatch.setenv("AI_HUB_REQUESTS_PER_MINUTE", "600")
    monkeypatch.setenv("AI_HUB_TRAININGS_PER_MINUTE", "60")
    request_quota.reset()
    principal_request_quotas.reset()
    monkeypatch.setattr(api_module, "engine", None)
    yield monkeypatch
    request_quota.reset()
    principal_request_quotas.reset()


def client(key):
    return TestClient(app, headers={"X-API-Key": key})


def test_scope_allows_only_its_selected_routes(scoped_env):
    reader = client(READER_KEY)
    assert reader.get("/models").status_code == 200
    assert reader.get("/status").status_code == 200
    assert reader.post("/predict", json={"data": [[0, 1]]}).status_code == 403
    assert reader.post("/train", json={"epochs": 1}).status_code == 403
    assert reader.post("/save", json={"model_name": "test"}).status_code == 403
    assert reader.post("/load/test").status_code == 403
    assert reader.delete("/models/test").status_code == 403

    trainer = client(TRAIN_KEY)
    assert trainer.post("/train", json={"epochs": 0}).status_code == 422
    assert trainer.post("/train/supervised", json={}).status_code == 422
    assert trainer.get("/models").status_code == 403
    assert trainer.post("/save", json={"model_name": "test"}).status_code == 403

    predictor = client(PREDICT_KEY)
    # Validation may report 400 (no model) or 422 (bad body), but not 403.
    assert predictor.post("/predict", json={"data": [[0, 1]]}).status_code in (200, 400)
    assert predictor.get("/status").status_code == 403

    writer = client(WRITER_KEY)
    assert writer.post("/save", json={"model_name": "test"}).status_code == 400
    assert writer.post("/load/missing-model").status_code == 404
    assert writer.delete("/models/missing-model").status_code == 404
    assert writer.get("/models").status_code == 403


def test_unrecognized_key_and_legacy_master_key_are_rejected(scoped_env):
    assert client(LEGACY_KEY).get("/models").status_code == 401
    assert client("unknown-but-long-enough-token-0123456789").get("/models").status_code == 401
    assert TestClient(app).get("/models").status_code == 401
    assert client(READER_KEY).get("/models").status_code == 200


def test_immediate_revocation_after_configuration_update(scoped_env):
    assert client(READER_KEY).get("/models").status_code == 200
    remaining = [record for record in base_registry() if record["id"] != "reader"]
    scoped_env.setenv("AI_HUB_API_KEYS_JSON", json.dumps(remaining))
    assert client(READER_KEY).get("/models").status_code == 401
    assert client(READER_TWO_KEY).get("/models").status_code == 200


def test_scopes_update_immediately(scoped_env):
    assert client(READER_KEY).post("/save", json={"model_name": "no"}).status_code == 403
    updated = base_registry()
    updated[0]["scopes"] = ["models:write"]
    scoped_env.setenv("AI_HUB_API_KEYS_JSON", json.dumps(updated))
    assert client(READER_KEY).post("/save", json={"model_name": "no"}).status_code == 400
    assert client(READER_KEY).get("/models").status_code == 403


def test_scoped_clients_have_separate_quota_windows(scoped_env):
    scoped_env.setenv("AI_HUB_REQUESTS_PER_MINUTE", "2")
    scoped_env.setenv("AI_HUB_TRAININGS_PER_MINUTE", "1")
    reader = client(READER_KEY)
    assert reader.get("/models").status_code == 200
    assert reader.get("/models").status_code == 200
    third = reader.get("/models")
    assert third.status_code == 429
    assert third.headers.get("retry-after") is not None

    # Another client must not reset or share the first client's budget.
    assert client(READER_TWO_KEY).get("/models").status_code == 200
    assert reader.get("/models").status_code == 429
    trainer = client(TRAIN_KEY)
    assert trainer.post("/train", json={"epochs": 0}).status_code == 422
    assert trainer.post("/train/supervised", json={}).status_code == 429


@pytest.mark.parametrize("bad", [
    "",
    "[]",
    "{}",
    "not-json",
    json.dumps([{"id": "x", "sha256": "bad", "scopes": ["read"]}]),
    json.dumps([record("x", READER_KEY, ["unknown"])]),
    json.dumps([record("x", READER_KEY, [])]),
    json.dumps([record("x", READER_KEY, ["read", "read"])]),
    json.dumps([record("x", READER_KEY, ["read"]), record("x", READER_TWO_KEY, ["read"])]),
    json.dumps([record("x", READER_KEY, ["read"]), record("y", READER_KEY, ["read"])]),
    json.dumps([record("x", READER_KEY, ["read"])] * 33),
    "x" * 40_000,
])
def test_invalid_keyring_never_falls_back_to_master(scoped_env, bad):
    scoped_env.setenv("AI_HUB_API_KEYS_JSON", bad)
    assert client(LEGACY_KEY).get("/models").status_code == 503
    assert client(READER_KEY).get("/models").status_code == 503


def test_single_key_compatibility_when_registry_unset(scoped_env):
    scoped_env.delenv("AI_HUB_API_KEYS_JSON")
    assert client(LEGACY_KEY).get("/models").status_code == 200
    assert client(LEGACY_KEY).post("/train", json={"epochs": 0}).status_code == 422
    assert client(READER_KEY).get("/models").status_code == 401
    principal = authenticate(LEGACY_KEY)
    assert principal.legacy and principal.scopes == frozenset(
        {"read", "predict", "train", "models:write"}
    )


def test_keyring_records_only_store_hashes(scoped_env):
    items = load_keyring()
    assert len(items) == len(base_registry())
    assert all(isinstance(p, APIPrincipal) and len(p.fingerprint) == 64 for p in items)
    assert READER_KEY not in os.environ["AI_HUB_API_KEYS_JSON"]


def test_unknown_route_policy_is_fail_closed():
    assert required_scope("POST", "/unlisted-future-route") is None
    assert required_scope("GET", "/models") == "read"
    assert required_scope("POST", "/predict") == "predict"
    assert required_scope("POST", "/train/supervised") == "train"
    assert required_scope("POST", "/load/some-model") == "models:write"
    assert required_scope("DELETE", "/models/some-model") == "models:write"


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_per_key_quota_pool_is_thread_safe_and_isolated():
    clock = FakeClock()
    pool = PrincipalRequestQuotas(clock)

    def attempt(_):
        return pool.check(READER_KEY, training=False, request_limit=5, training_limit=5)

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(attempt, range(20)))
    assert sum(x is None for x in results) == 5
    assert all(x == 60 for x in results if x is not None)
    assert pool.check(READER_TWO_KEY, training=False, request_limit=5, training_limit=5) is None
    assert pool.check(READER_KEY, training=False, request_limit=5, training_limit=5) == 60
    clock.now += 60
    assert pool.check(READER_KEY, training=False, request_limit=5, training_limit=5) is None


def test_token_sha256_requires_minimum_key_length():
    with pytest.raises(ValueError):
        token_sha256("short")
    assert token_sha256(READER_KEY) == hashlib.sha256(READER_KEY.encode()).hexdigest()
