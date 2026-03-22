import pytest
import torch
from ai_hub.ml_engine import MLEngine


class TestMLEngine:
    def test_initialization(self):
        engine = MLEngine(input_dim=10, hidden_dim=20, output_dim=5)
        assert engine is not None
        assert not engine.is_trained

    def test_train(self):
        engine = MLEngine(input_dim=10, hidden_dim=20, output_dim=5)
        X = torch.randn(100, 10)
        y = torch.randint(0, 5, (100,))

        result = engine.train(X, y, epochs=2)

        assert "final_loss" in result
        assert "epochs" in result
        assert result["epochs"] == 2
        assert engine.is_trained

    def test_predict(self):
        engine = MLEngine(input_dim=10, hidden_dim=20, output_dim=5)
        X_train = torch.randn(100, 10)
        y_train = torch.randint(0, 5, (100,))
        engine.train(X_train, y_train, epochs=2)

        X_test = torch.randn(10, 10)
        predictions = engine.predict(X_test)

        assert predictions.shape == (10, 5)

    def test_evaluate(self):
        engine = MLEngine(input_dim=10, hidden_dim=20, output_dim=5)
        X = torch.randn(100, 10)
        y = torch.randint(0, 5, (100,))
        engine.train(X, y, epochs=2)

        result = engine.evaluate(X[:10], y[:10])

        assert "accuracy" in result
        assert 0 <= result["accuracy"] <= 1

    def test_save_load(self, tmp_path):
        engine = MLEngine(input_dim=10, hidden_dim=20, output_dim=5)
        X = torch.randn(100, 10)
        y = torch.randint(0, 5, (100,))
        engine.train(X, y, epochs=2)

        model_path = tmp_path / "test_model.pt"
        engine.save(str(model_path))

        new_engine = MLEngine(input_dim=10, hidden_dim=20, output_dim=5)
        new_engine.load(str(model_path))

        assert new_engine.is_trained
