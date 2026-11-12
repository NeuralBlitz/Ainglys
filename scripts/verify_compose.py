"""Validate the *minimal* Ainglys Compose stack before local deployment.

Checks rendered Docker Compose JSON, rather than scanning YAML strings.
Does not assert that containers are running or the experimental full stack works.
Secrets are never printed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


CORE_SERVICES = frozenset({"api", "ui"})
EXPECTED_PORTS = {"api": 8000, "ui": 8501}
ALLOWED_LOOPBACK = {"127.0.0.1", "::1"}
EXPERIMENTAL_ASSETS = (
    "Dockerfile.ray",
    "redis/redis.conf",
    "nginx/nginx.conf",
    "nginx/ssl",
    "monitoring/prometheus.yml",
    "monitoring/grafana/dashboards",
    "monitoring/grafana/datasources",
    "init-scripts",
)


def _published_ports(service: dict[str, Any]) -> list[tuple[str, int, int]]:
    """Convert published Compose port mappings to (host, public, target)."""
    mappings = []
    for mapping in service.get("ports", []):
        if not isinstance(mapping, dict):
            raise ValueError("Expected normalized Docker Compose port objects")
        host = mapping.get("host_ip")
        public = mapping.get("published")
        target = mapping.get("target")
        try:
            mappings.append((str(host or ""), int(public), int(target)))
        except (ValueError, TypeError) as exc:
            raise ValueError("Malformed published port mapping") from exc
    return mappings


def inspect_core_config(config: dict[str, Any], *, include_ui: bool) -> list[str]:
    """Return actionable errors for an unsafe or incomplete core deployment."""
    errors: list[str] = []
    services = config.get("services")
    if not isinstance(services, dict):
        return ["Rendered Compose document has no services object"]

    requested = {"api", "ui"} if include_ui else {"api"}
    for name in requested:
        if name not in services:
            errors.append(f"Missing required service: {name}")

    # Reject unexpectedly exposed services even when their profile is enabled.
    unapproved = set(services) - CORE_SERVICES
    if unapproved:
        errors.append("Unexpected services in core Compose: " + ", ".join(sorted(unapproved)))

    for name in sorted(requested & set(services)):
        service = services[name]
        if not isinstance(service, dict):
            errors.append(f"{name}: invalid service declaration")
            continue

        if not isinstance(service.get("build"), dict):
            errors.append(f"{name}: expected locally built application image")
        elif service["build"].get("target") != "production":
            errors.append(f"{name}: must build the production image target")

        if not service.get("init"):
            errors.append(f"{name}: missing init process")

        capabilities = service.get("cap_drop") or []
        if "ALL" not in capabilities:
            errors.append(f"{name}: all Linux capabilities must be dropped")

        security_options = service.get("security_opt") or []
        if not any(str(option).lower() == "no-new-privileges:true"
                   for option in security_options):
            errors.append(f"{name}: no-new-privileges must be enabled")

        if not service.get("restart"):
            errors.append(f"{name}: restart policy must be configured")

        try:
            published = _published_ports(service)
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
            published = []

        if len(published) != 1:
            errors.append(f"{name}: exactly one loopback port must be published")
        for host, public, target in published:
            expected = EXPECTED_PORTS[name]
            if host not in ALLOWED_LOOPBACK:
                errors.append(f"{name}: published port must be loopback-only")
            if public != expected or target != expected:
                errors.append(f"{name}: unexpected published/target port")

        environment = service.get("environment") or {}
        if not isinstance(environment, dict):
            errors.append(f"{name}: environment must be a mapping")
            environment = {}
        api_key = environment.get("AI_HUB_API_KEY")
        if not isinstance(api_key, str) or len(api_key) < 32:
            errors.append(f"{name}: AI_HUB_API_KEY must be configured with 32+ characters")

        if name == "api":
            health = service.get("healthcheck") or {}
            if not isinstance(health, dict) or not health.get("test") or health.get("disable"):
                errors.append("api: a valid enabled /health check is required")
            else:
                test = health["test"]
                if "/health" not in " ".join(str(x) for x in test if x is not None):
                    errors.append("api: healthcheck must use /health")
            if not any(v.get("type") == "volume" and v.get("target") == "/app/models"
                       for v in service.get("volumes", []) if isinstance(v, dict)):
                errors.append("api: model persistence must use a named volume")
        else:
            ui_health = service.get("healthcheck") or {}
            if not isinstance(ui_health, dict) or not ui_health.get("test") or ui_health.get("disable"):
                errors.append("ui: Streamlit healthcheck must be enabled")
            elif "/_stcore/health" not in " ".join(
                str(x) for x in ui_health["test"] if x is not None
            ) or "8501" not in " ".join(str(x) for x in ui_health["test"]):
                errors.append("ui: healthcheck must query port 8501 /_stcore/health")
            if environment.get("AI_HUB_API_URL") != "http://api:8000":
                errors.append("ui: API destination must use internal service DNS")
            dependency = (service.get("depends_on") or {}).get("api")
            if not isinstance(dependency, dict) or dependency.get("condition") != "service_healthy":
                errors.append("ui: must wait for a healthy API service")

        for volume in service.get("volumes", []) or []:
            if not isinstance(volume, dict) or volume.get("type") == "bind":
                errors.append(f"{name}: host bind mounts are forbidden in core deployment")

    return errors


def missing_experimental_assets(root: Path) -> list[str]:
    """List missing legacy manifest dependencies without claiming readiness."""
    return [name for name in EXPERIMENTAL_ASSETS if not (root / name).exists()]


def render_compose(root: Path, *, ui: bool) -> dict[str, Any]:
    command = ["docker", "compose", "--project-directory", str(root)]
    if ui:
        command += ["--profile", "ui"]
    command += ["-f", str(root / "docker-compose.yml"), "config", "--format", "json"]
    completed = subprocess.run(
        command, cwd=root, capture_output=True, text=True,
        check=False, timeout=30,
    )
    if completed.returncode != 0:
        # Docker can include interpolated secrets in error messages. Avoid
        # displaying stderr/stdout to preserve credential confidentiality.
        raise RuntimeError("Docker Compose config failed; inspect local configuration privately")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Docker Compose did not produce valid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Docker Compose did not produce an object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the Ainglys core Compose safety contract")
    parser.add_argument("--ui", action="store_true", help="Validate the UI profile as well")
    parser.add_argument("--experimental-assets", action="store_true",
                        help="Report missing legacy full-stack files; always non-certifying")
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)
    root = args.root.resolve()

    if args.experimental_assets:
        missing = missing_experimental_assets(root)
        for path in missing:
            print(f"MISSING EXPERIMENTAL ASSET: {path}")
        print("Experimental full-stack deployment is not release-validated")
        return 1

    try:
        config = render_compose(root, ui=args.ui)
        errors = inspect_core_config(config, include_ui=args.ui)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"Preflight could not complete: {type(exc).__name__}", file=sys.stderr)
        return 2

    if errors:
        for issue in errors:
            print(f"FAIL: {issue}", file=sys.stderr)
        return 1
    print("PASS: Ainglys core Compose config meets the declared safety contract")
    print("Note: this is a static preflight, not proof of runtime or production readiness")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
