"""Unit tests for the supported Ainglys core Compose deployment contract."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.verify_compose import (
    inspect_core_config, missing_experimental_assets, render_compose, main
)


def fixture_document():
    common = {
        "build": {"context": "/tmp/project", "dockerfile": "Dockerfile", "target": "production"},
        "init": True,
        "restart": "unless-stopped",
        "security_opt": ["no-new-privileges:true"],
        "cap_drop": ["ALL"],
        "environment": {"AI_HUB_API_KEY": "example-test-secret-that-is-more-than-32-chars"},
        "volumes": [],
    }
    api = dict(copy.deepcopy(common), **{
        "ports": [{"host_ip": "127.0.0.1", "published": "8000", "target": 8000}],
        "volumes": [{"type": "volume", "target": "/app/models", "source": "models"}],
        "healthcheck": {"test": ["CMD", "curl", "--fail", "http://127.0.0.1:8000/health"]},
    })
    ui = dict(copy.deepcopy(common), **{
        "ports": [{"host_ip": "127.0.0.1", "published": "8501", "target": 8501}],
        "environment": {
            "AI_HUB_API_KEY": "example-test-secret-that-is-more-than-32-chars",
            "AI_HUB_API_URL": "http://api:8000",
        },
        "depends_on": {"api": {"condition": "service_healthy"}},
        "healthcheck": {"test": [
            "CMD", "curl", "--fail", "http://127.0.0.1:8501/_stcore/health"
        ]},
    })
    return {"services": {"api": api, "ui": ui}}


@pytest.mark.parametrize("with_ui", [True, False])
def test_valid_core_config(with_ui):
    assert inspect_core_config(fixture_document(), include_ui=with_ui) == []


@pytest.mark.parametrize("mutate,expected", [
    (lambda c: c["services"]["api"]["ports"][0].update(host_ip="0.0.0.0"), "loopback-only"),
    (lambda c: c["services"]["api"]["ports"][0].pop("host_ip"), "loopback-only"),
    (lambda c: c["services"]["api"]["ports"][0].update(published="8181"), "unexpected published"),
    (lambda c: c["services"]["api"]["environment"].pop("AI_HUB_API_KEY"), "AI_HUB_API_KEY"),
    (lambda c: c["services"]["api"].update(cap_drop=[]), "capabilities"),
    (lambda c: c["services"]["api"].update(security_opt=[]), "no-new-privileges"),
    (lambda c: c["services"]["api"].update(init=False), "init"),
    (lambda c: c["services"]["api"].update(restart=""), "restart"),
    (lambda c: c["services"]["api"].update(healthcheck={}), "health"),
    (lambda c: c["services"]["api"].update(volumes=[]), "named volume"),
    (lambda c: c["services"]["api"].update(
        volumes=[{"type": "bind", "source": "/tmp", "target": "/app/models"}]
    ), "host bind mounts"),
    (lambda c: c["services"].update(redis={}), "Unexpected services"),
])
def test_bad_core_configuration_rejected(mutate, expected):
    data = fixture_document()
    mutate(data)
    assert expected in " ".join(inspect_core_config(data, include_ui=False))


@pytest.mark.parametrize("mutate,expected", [
    (lambda c: c["services"]["ui"]["ports"][0].update(host_ip="0.0.0.0"), "loopback-only"),
    (lambda c: c["services"]["ui"]["environment"].update(AI_HUB_API_URL="https://untrusted"), "internal"),
    (lambda c: c["services"]["ui"].update(depends_on={}), "healthy"),
    (lambda c: c["services"]["ui"].update(build={"target": "development"}), "production"),
    (lambda c: c["services"]["ui"].update(healthcheck={}), "Streamlit healthcheck"),
    (lambda c: c["services"]["ui"]["healthcheck"].update(
        test=["CMD", "curl", "http://127.0.0.1:8000/health"]
    ), "8501"),
])
def test_unsafe_ui_configuration_rejected(mutate, expected):
    data = fixture_document()
    mutate(data)
    assert expected in " ".join(inspect_core_config(data, include_ui=True))


def test_experimental_assets_inventory_is_conservative(tmp_path):
    missing = missing_experimental_assets(tmp_path)
    assert "Dockerfile.ray" in missing
    assert "nginx/nginx.conf" in missing
    assert "monitoring/prometheus.yml" in missing
    assert "init-scripts" in missing
    assert len(missing) >= 5


def test_render_compose_parses_json_and_does_not_print_secrets(monkeypatch, tmp_path):
    from scripts import verify_compose
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout=json.dumps(fixture_document()), stderr="")

    monkeypatch.setattr(verify_compose.subprocess, "run", run)
    assert render_compose(tmp_path, ui=True) == fixture_document()
    assert "--profile" in calls[0] and "ui" in calls[0]
    assert "--format" in calls[0] and "json" in calls[0]


def test_failed_compose_does_not_echo_sensitive_stderr(monkeypatch, tmp_path, capsys):
    from scripts import verify_compose

    def run(*args, **kwargs):
        return SimpleNamespace(
            returncode=1, stdout="",
            stderr="supersecret-config-value: invalid Compose settings",
        )
    monkeypatch.setattr(verify_compose.subprocess, "run", run)
    assert main(["--root", str(tmp_path)]) == 2
    capture = capsys.readouterr()
    assert "supersecret-config-value" not in capture.out + capture.err


def test_preflight_reports_valid_core(monkeypatch, tmp_path, capsys):
    from scripts import verify_compose
    monkeypatch.setattr(verify_compose, "render_compose", lambda root, ui: fixture_document())
    assert main(["--root", str(tmp_path), "--ui"]) == 0
    assert "not proof" in capsys.readouterr().out


def test_experimental_assets_mode_is_non_certifying(tmp_path):
    assert main(["--root", str(tmp_path), "--experimental-assets"]) == 1
