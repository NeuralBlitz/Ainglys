import pytest
from fastapi.testclient import TestClient
from ai_hub.api import app


client = TestClient(app)


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
