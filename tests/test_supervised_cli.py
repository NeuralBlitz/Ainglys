"""CLI regressions for the supervised dataset training command."""
import json
from unittest.mock import MagicMock

from click.testing import CliRunner

import ai_hub.cli as cli_module


API_KEY = "test-only-cli-api-key-with-32-characters"


def _dataset():
    return {
        "features": [[float(i) / 40, float(i % 3)] for i in range(40)],
        "labels": [i % 2 for i in range(40)],
        "epochs": 2,
    }


def test_train_data_cli_sends_validated_request(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_HUB_API_KEY", API_KEY)
    data_file = tmp_path / "dataset.json"
    data_file.write_text(json.dumps(_dataset()), encoding="utf-8")
    response = MagicMock()
    response.json.return_value = {
        "dataset": {"training_samples": 32, "validation_samples": 8},
        "validation": {"accuracy": 0.8, "cross_entropy": 0.5},
    }
    sent = []

    def post(url, **kwargs):
        sent.append((url, kwargs))
        return response

    monkeypatch.setattr(cli_module._api_session, "post", post)
    result = CliRunner().invoke(
        cli_module.cli, ["model", "train-data", str(data_file), "--api-url", "http://localhost:8000"]
    )
    assert result.exit_code == 0, result.output
    assert len(sent) == 1
    assert sent[0][0] == "http://localhost:8000/train/supervised"
    assert sent[0][1]["headers"]["X-API-Key"] == API_KEY
    assert sent[0][1]["json"]["labels"] == _dataset()["labels"]
    assert "Validation samples" in result.output


def test_train_data_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv("AI_HUB_API_KEY", raising=False)
    data_file = tmp_path / "dataset.json"
    data_file.write_text(json.dumps(_dataset()))
    result = CliRunner().invoke(cli_module.cli, ["model", "train-data", str(data_file)])
    assert result.exit_code != 0
    assert "AI_HUB_API_KEY" in result.output


def test_train_data_refuses_bad_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_HUB_API_KEY", API_KEY)
    data_file = tmp_path / "dataset.json"
    data = _dataset()
    data["labels"] = [0] * 40
    data_file.write_text(json.dumps(data))
    result = CliRunner().invoke(cli_module.cli, ["model", "train-data", str(data_file)])
    assert result.exit_code != 0
    assert "Invalid training dataset" in result.output


def test_train_data_refuses_insecure_remote_api(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_HUB_API_KEY", API_KEY)
    data_file = tmp_path / "dataset.json"
    data_file.write_text(json.dumps(_dataset()))
    for url in ["http://example.com:8000", "http://localhost:8000@evil.example",
                "file:///tmp/data", "http://127.0.0.1:8000?redirect=elsewhere"]:
        result = CliRunner().invoke(
            cli_module.cli, ["model", "train-data", str(data_file), "--api-url", url]
        )
        assert result.exit_code != 0, url
        assert "Use HTTPS" in result.output, url
