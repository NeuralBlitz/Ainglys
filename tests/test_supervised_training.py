"""Regressions for authenticated, bounded and leakage-resistant supervised training."""
import math

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import ai_hub.api as api_module
from ai_hub.supervised import SupervisedTrainRequest, stratified_holdout_indices


API_KEY = "supervised-training-tests-key-at-least-32-characters"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AI_HUB_API_KEY", API_KEY)
    monkeypatch.setattr(api_module, "engine", None)
    return TestClient(api_module.app, headers={"X-API-Key": API_KEY})


def example_dataset():
    return {
        "features": [[float(i) / 40.0, float(i % 3)] for i in range(40)],
        "labels": [i % 2 for i in range(40)],
        "hidden_dim": 12,
        "epochs": 2,
        "learning_rate": 0.01,
        "validation_fraction": 0.25,
        "seed": 101,
    }


def test_stratified_holdout_is_disjoint_deterministic_and_balanced():
    labels = [0, 1] * 20
    train, val = stratified_holdout_indices(labels, 0.25, seed=13)
    again = stratified_holdout_indices(labels, 0.25, seed=13)
    assert (train, val) == again
    assert not set(train).intersection(val)
    assert sorted(train + val) == list(range(len(labels)))
    assert len(val) == 10
    assert {labels[index] for index in train} == {0, 1}
    assert {labels[index] for index in val} == {0, 1}


def test_supervised_training_and_prediction(client):
    response = client.post("/train/supervised", json=example_dataset())
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "trained"
    assert result["data_source"] == "user_provided_supervised"
    assert result["model_info"] == {
        "input_dim": 2, "hidden_dim": 12, "output_dim": 2
    }
    assert result["dataset"]["samples"] == 40
    assert result["dataset"]["training_samples"] == 30
    assert result["dataset"]["validation_samples"] == 10
    assert result["dataset"]["split_strategy"] == "stratified_holdout"
    assert result["dataset"]["random_seed"] == 101
    assert 0 <= result["validation"]["accuracy"] <= 1
    assert math.isfinite(result["validation"]["cross_entropy"])
    assert result["validation"]["cross_entropy"] >= 0
    prediction = client.post("/predict", json={"data": [[0.1, 1.0]]})
    assert prediction.status_code == 200
    assert len(prediction.json()["predictions"]) == 1


def test_heldout_samples_are_never_passed_to_training(client, monkeypatch):
    data = example_dataset()
    train_indices, val_indices = stratified_holdout_indices(
        data["labels"], data["validation_fraction"], data["seed"]
    )
    original = api_module.MLEngine.train
    observed = []

    def track_training(self, X, y, *args, **kwargs):
        observed.extend(X[:, 0].tolist())
        return original(self, X, y, *args, **kwargs)

    monkeypatch.setattr(api_module.MLEngine, "train", track_training)
    result = client.post("/train/supervised", json=data)
    assert result.status_code == 200, result.text
    # Every first-column feature is a unique sample identifier.
    observed_identifiers = {round(value * 40) for value in observed}
    assert observed_identifiers == set(train_indices)
    assert observed_identifiers.isdisjoint(val_indices)


@pytest.mark.parametrize("alter", [
    lambda p: p.update(features=p["features"][:10], labels=p["labels"][:10]),
    lambda p: p.update(labels=p["labels"][:-1]),
    lambda p: p.update(features=[row[:1] if i % 2 else row for i, row in enumerate(p["features"])]),
    lambda p: p.update(labels=[0] * 40),
    lambda p: p.update(labels=[i % 2 * 2 for i in range(40)]),
    lambda p: p.update(epochs=25),
    lambda p: p.update(validation_fraction=0.9),
    lambda p: p.update(extra_field="unknown"),
])
def test_bad_supervised_requests_are_rejected(client, alter):
    data = example_dataset()
    alter(data)
    response = client.post("/train/supervised", json=data)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 1e10])
def test_nonfinite_or_extreme_feature_values_are_rejected(bad):
    data = example_dataset()
    data["features"][0][0] = bad
    with pytest.raises(ValidationError):
        SupervisedTrainRequest.model_validate(data)


def test_boolean_class_labels_are_rejected():
    data = example_dataset()
    data["labels"][0] = True
    with pytest.raises(ValidationError):
        SupervisedTrainRequest.model_validate(data)


def test_endpoint_authentication_required(client):
    payload = example_dataset()
    unauthenticated = TestClient(api_module.app)
    assert unauthenticated.post("/train/supervised", json=payload).status_code == 401
    invalid = TestClient(api_module.app, headers={"X-API-Key": "invalid"})
    assert invalid.post("/train/supervised", json=payload).status_code == 401


def test_concurrent_training_is_rejected(client):
    with api_module._training_lock:
        response = client.post("/train/supervised", json=example_dataset())
        assert response.status_code == 429


def test_failed_training_preserves_previous_model(client, monkeypatch):
    trained = client.post("/train/supervised", json=example_dataset())
    assert trained.status_code == 200
    earlier_model = api_module.engine

    def fail_training(*args, **kwargs):
        raise RuntimeError("internal model training diagnostic - must not be exposed")

    monkeypatch.setattr(api_module.MLEngine, "train", fail_training)
    response = client.post("/train/supervised", json=example_dataset())
    assert response.status_code == 500
    assert "internal model training diagnostic" not in response.text
    assert api_module.engine is earlier_model
