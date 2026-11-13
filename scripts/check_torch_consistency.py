"""Check reproducible CPU PyTorch agreement across pyproject, uv.lock and Docker.

The production image preinstalls a specific PyTorch CPU wheel. If a package
upgrade changes only uv.lock and pyproject, pip may replace that wheel with a
different build (including a GPU build) during the final image installation.
Fail the CI integrity job before expensive builds when these disagree.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import tomllib

CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
VERSION_PATTERN = re.compile(r"\d+\.\d+\.\d+(?:\.post\d+)?\+cpu\Z")


def check_cpu_torch(root: Path) -> str:
    """Return the exact pinned CPU torch version, otherwise fail closed."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")

    project_deps = project["project"]["dependencies"]
    torch_requirements = [
        dep for dep in project_deps
        if isinstance(dep, str) and re.match(r"^torch\s*(?:[<>=!~]|$)", dep, re.I)
    ]
    if len(torch_requirements) != 1:
        raise ValueError("Root manifest must declare exactly one PyTorch requirement")
    requirement = torch_requirements[0]
    req_match = re.fullmatch(r"torch\s*>=\s*(\d+\.\d+\.\d+)", requirement, re.I)
    if req_match is None:
        raise ValueError("Root torch requirement must use an explicit >= X.Y.Z floor")

    pinned = re.findall(r"torch==([\d.]+\+cpu)", dockerfile)
    if len(pinned) != 1 or not VERSION_PATTERN.fullmatch(pinned[0]):
        raise ValueError("Dockerfile must preinstall one exact torch==X.Y.Z+cpu wheel")
    cpu_pin = pinned[0]
    plain = cpu_pin.partition("+")[0]
    if tuple(map(int, plain.split(".")[:3])) < tuple(map(int, req_match.group(1).split("."))):
        raise ValueError("Docker CPU PyTorch pin is below pyproject minimum version")
    if CPU_INDEX_URL not in dockerfile or "--index-url" not in dockerfile:
        raise ValueError("Docker CPU PyTorch install must use its explicit CPU wheel index")

    uv_sources = project.get("tool", {}).get("uv", {}).get("sources", {}).get("torch")
    if not isinstance(uv_sources, list) or not any(
        isinstance(item, dict) and item.get("index") == "pytorch-cpu"
        and "sys_platform == 'linux'" in str(item.get("marker", ""))
        for item in uv_sources
    ):
        raise ValueError("Root torch source must use the Linux-only PyTorch CPU index")
    indexes = project.get("tool", {}).get("uv", {}).get("index", [])
    if not any(
        isinstance(item, dict) and item.get("name") == "pytorch-cpu"
        and item.get("url") == CPU_INDEX_URL and item.get("explicit") is True
        for item in indexes
    ):
        raise ValueError("PyTorch CPU index must be explicit and match the Docker source")

    packages = lock.get("package", [])
    cpu_records = [
        item for item in packages
        if item.get("name") == "torch"
        and item.get("source", {}).get("registry") == CPU_INDEX_URL
    ]
    if len(cpu_records) != 1 or cpu_records[0].get("version") != cpu_pin:
        raise ValueError(
            "uv.lock Linux CPU torch record must exactly match the Docker pin"
        )
    # Check that the resolved root ai-hub package also references that CPU pin.
    roots = [item for item in packages if item.get("name") == project["project"]["name"]]
    if len(roots) != 1:
        raise ValueError("Root package not found exactly once in uv.lock")
    cpu_links = [
        dep for dep in roots[0].get("dependencies", [])
        if dep.get("name") == "torch"
        and dep.get("source", {}).get("registry") == CPU_INDEX_URL
    ]
    if len(cpu_links) != 1 or cpu_links[0].get("version") != cpu_pin:
        raise ValueError("uv.lock root dependency must select the same Linux CPU wheel")
    return cpu_pin


def check_running_torch(expected: str) -> None:
    """After container startup, ensure pip has not replaced the intended wheel."""
    import torch

    if torch.__version__ != expected:
        raise ValueError("Installed PyTorch version differs from pinned CPU wheel")
    if torch.version.cuda is not None:
        raise ValueError("Production PyTorch has unexpected CUDA support")
    if not torch.isfinite(torch.tensor([1.0])).all().item():
        raise ValueError("Basic PyTorch CPU operation failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Ainglys CPU PyTorch dependency alignment")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--runtime", action="store_true", help="Also check PyTorch installed inside the running image")
    args = parser.parse_args(argv)
    try:
        version = check_cpu_torch(args.root.resolve())
        if args.runtime:
            check_running_torch(version)
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError, ValueError) as exc:
        print(f"CPU PyTorch consistency check failed: {exc}", file=sys.stderr)
        return 1
    print(f"CPU PyTorch consistency passed: {version}" + (" (runtime verified)" if args.runtime else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
