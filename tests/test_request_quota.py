"""Rate-limit unit and API integration tests; no external clock sleeps."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from ai_hub.api import app
from ai_hub.request_quota import RequestQuota, request_quota


KEY = "quota-test-api-key-with-at-least-32-characters"


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_window_boundary_and_retry_after():
    clock = FakeClock()
    quota = RequestQuota(clock)
    assert quota.check(KEY, training=False, request_limit=2, training_limit=1) is None
    assert quota.check(KEY, training=True, request_limit=2, training_limit=1) is None
    assert quota.check(KEY, training=False, request_limit=2, training_limit=1) == 60
    clock.now += 59.1
    assert quota.check(KEY, training=False, request_limit=2, training_limit=1) == 1
    clock.now += 0.9
    assert quota.check(KEY, training=True, request_limit=2, training_limit=1) is None


def test_training_quota_is_separate_from_request_quota():
    quota = RequestQuota(FakeClock())
    assert quota.check(KEY, training=True, request_limit=6, training_limit=1) is None
    assert quota.check(KEY, training=True, request_limit=6, training_limit=1) == 60
    assert quota.check(KEY, training=False, request_limit=6, training_limit=1) is None


def test_key_rotation_resets_the_shared_quota_without_storing_raw_key():
    quota = RequestQuota(FakeClock())
    assert quota.check(KEY, training=False, request_limit=1, training_limit=1) is None
    assert quota.check(KEY, training=False, request_limit=1, training_limit=1) == 60
    rotated = KEY + "-rotated"
    assert quota.check(rotated, training=False, request_limit=1, training_limit=1) is None
    assert KEY.encode() not in quota._fingerprint


def test_atomic_limit_under_concurrent_clients():
    quota = RequestQuota(FakeClock())

    def one_call(_):
        return quota.check(KEY, training=False, request_limit=5, training_limit=5)

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(one_call, range(30)))
    assert sum(result is None for result in results) == 5
    assert all(result == 60 for result in results if result is not None)


@pytest.mark.parametrize("request_limit,training_limit", [
    (0, 1), (10001, 1), (3, 4), (5, 0),
])
def test_invalid_limits_raise_value_error(request_limit, training_limit):
    with pytest.raises(ValueError):
        RequestQuota(FakeClock()).check(
            KEY, training=True,
            request_limit=request_limit, training_limit=training_limit,
        )


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("AI_HUB_API_KEY", KEY)
    monkeypatch.setenv("AI_HUB_REQUESTS_PER_MINUTE", "2")
    monkeypatch.setenv("AI_HUB_TRAININGS_PER_MINUTE", "1")
    request_quota.reset()
    yield TestClient(app, headers={"X-API-Key": KEY})
    request_quota.reset()


def test_api_enforces_authenticated_request_quota(api_client):
    assert api_client.get("/models").status_code == 200
    assert api_client.get("/status").status_code == 200
    blocked = api_client.get("/models")
    assert blocked.status_code == 429
    assert 1 <= int(blocked.headers["retry-after"]) <= 60
    assert api_client.get("/health").status_code == 200


def test_bad_key_does_not_consume_valid_budget(api_client):
    invalid = TestClient(app, headers={"X-API-Key": "bad-key"})
    for _ in range(4):
        assert invalid.get("/models").status_code == 401
    assert api_client.get("/models").status_code == 200
    assert TestClient(app).get("/models").status_code == 401
    assert api_client.get("/models").status_code == 200


def test_training_limit_applies_to_both_training_routes(api_client):
    # FastAPI authentication executes before request-body validation; invalid
    # train attempts still consume a budget slot, preventing expensive retries.
    assert api_client.post("/train", json={"epochs": 0}).status_code == 422
    limited = api_client.post("/train/supervised", json={})
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1
    assert api_client.get("/models").status_code == 200


def test_invalid_quota_configuration_fails_closed(api_client, monkeypatch):
    monkeypatch.setenv("AI_HUB_REQUESTS_PER_MINUTE", "not-a-number")
    assert api_client.get("/models").status_code == 503
    monkeypatch.setenv("AI_HUB_REQUESTS_PER_MINUTE", "2")
    monkeypatch.setenv("AI_HUB_TRAININGS_PER_MINUTE", "3")
    assert api_client.get("/models").status_code == 503
