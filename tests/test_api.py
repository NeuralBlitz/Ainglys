import pytest
from fastapi.testclient import TestClient
from ai_hub.api import app


TEST_API_KEY = "test-security-api-key-with-32-characters-or-more"

@pytest.fixture(autouse=True)
def configured_api_key(monkeypatch):
    monkeypatch.setenv("AI_HUB_API_KEY", TEST_API_KEY)


client = TestClient(app, headers={"X-API-Key": TEST_API_KEY})


class TestAPI:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "AI Hub API"
        assert data["version"] == "0.1.0"

    def test_status_no_model(self):
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert data["trained"] == False

    def test_train(self):
        request = {"input_dim": 10, "hidden_dim": 20, "output_dim": 5, "epochs": 2}
        response = client.post("/train", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "trained"
        assert data["data_source"] == "synthetic_demo_random_labels"
        assert "training" in data
        assert "evaluation" in data

    def test_predict_without_training(self):
        # Reset by creating new test
        response = client.post("/predict", json={"data": [[0.1, 0.2]]})
        assert response.status_code == 400

    def test_train_and_predict(self):
        # Train first
        train_request = {"input_dim": 3, "hidden_dim": 6, "output_dim": 2, "epochs": 2}
        response = client.post("/train", json=train_request)
        assert response.status_code == 200

        # Then predict with correct dimensions
        predict_request = {"data": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
        response = client.post("/predict", json=predict_request)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2

    def test_save_and_list_models(self):
        # Train a model
        train_request = {
            "input_dim": 10,
            "hidden_dim": 20,
            "output_dim": 5,
            "epochs": 1,
        }
        client.post("/train", json=train_request)

        # Save it
        save_request = {"model_name": "test_model"}
        response = client.post("/save", json=save_request)
        assert response.status_code == 200
        assert response.json()["status"] == "saved"

        # List models
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data

        # Load model
        response = client.post("/load/test_model")
        assert response.status_code == 200
        assert response.json()["status"] == "loaded"

        # Delete model
        response = client.delete("/models/test_model")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"


class TestAPIAuthentication:
    def test_public_root_but_sensitive_status_requires_key(self):
        assert TestClient(app).get("/").status_code == 200
        assert TestClient(app).get("/health").status_code == 200
        assert TestClient(app).get("/status").status_code == 401
        assert client.get("/status").status_code == 200

    def test_wrong_key_rejected(self):
        assert TestClient(app, headers={"X-API-Key": "invalid"}).get("/models").status_code == 401

    def test_fail_closed_when_key_missing(self, monkeypatch):
        monkeypatch.delenv("AI_HUB_API_KEY", raising=False)
        assert client.get("/models").status_code == 503

    def test_reject_model_path_traversal(self):
        response = client.post("/train", json={"input_dim": 3, "hidden_dim": 4, "output_dim": 2, "epochs": 1})
        assert response.status_code == 200
        response = client.post("/save", json={"model_name": "../outside"})
        assert response.status_code == 422
        assert client.post("/load/%2e%2e", json={}).status_code in (404, 422)
        assert client.delete("/models/%2e%2e").status_code in (404, 422)

    def test_training_is_bounded(self):
        assert client.post("/train", json={"input_dim": 100000}).status_code == 422
        assert client.post("/train", json={"epochs": 100000}).status_code == 422
        assert client.post("/train", json={"learning_rate": 0}).status_code == 422

    def test_prediction_batch_is_bounded(self):
        response = client.post("/predict", json={"data": [[0.0] * 600]})
        assert response.status_code == 422


    def test_concurrent_training_is_rejected(self):
        import ai_hub.api as api_module
        assert api_module._training_lock.acquire(blocking=False)
        try:
            response = client.post("/train", json={"epochs": 1})
            assert response.status_code == 429
        finally:
            api_module._training_lock.release()

    def test_nonfinite_prediction_inputs_are_rejected(self):
        from pydantic import ValidationError
        from ai_hub.api import PredictRequest
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValidationError):
                PredictRequest.model_validate({"data": [[bad]]})
