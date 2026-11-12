"""Reproducible, conservative inventory of Ainglys research subprojects.

This static scan measures *repository evidence*, NOT research correctness.
A passing import/shape test is never classified as scientific validation.
No external network access, model execution, or submodule traversal is needed.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

SCHEMA_VERSION = 1
SCIENTIFIC_STATUS = "not_independently_verified"


def _relative(root: Path, file: Path) -> str:
    return file.relative_to(root).as_posix()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _py_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.py")
        if p.is_file() and not p.is_symlink()
        and not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(root).parts)
    )


def _classify_scaffold(path: Path) -> bool:
    """Recognize a tiny two-layer CoreModule pattern, not its scientific quality."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeError, OSError):
        return False
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if len(classes) != 1 or classes[0].name != "CoreModule":
        return False
    methods = {n.name for n in classes[0].body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if methods != {"__init__", "forward"}:
        return False
    if len(path.read_text(encoding="utf-8").splitlines()) > 100:
        return False
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    linear_calls = sum(
        isinstance(n.func, ast.Attribute)
        and n.func.attr == "Linear"
        for n in functions
    )
    return linear_calls == 2 and any(
        isinstance(n.func, ast.Attribute)
        and n.func.attr.lower() == "relu"
        for n in functions
    )


def _test_signal(test_paths: list[Path]) -> str:
    """Distinguish shape smoke coverage from broader tests, without validating either."""
    if not test_paths:
        return "no_tests"
    names: list[str] = []
    for file in test_paths:
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeError, OSError):
            return "uninspectable_tests"
        names.extend(
            n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name.startswith("test_")
        )
    if not names:
        return "no_test_functions"
    if set(names).issubset({"test_initialization", "test_forward"}):
        return "initialization_and_forward_smoke"
    return "additional_test_functions_present"


def inspect_project(root: Path, directory: Path) -> dict[str, Any]:
    name = directory.name
    src_root = directory / "src"
    test_root = directory / "tests"
    source = [
        p for p in _py_files(src_root) if p.name != "__init__.py"
    ] if src_root.is_dir() else []
    tests = [
        p for p in _py_files(test_root)
        if p.name.startswith("test_")
    ] if test_root.is_dir() else []
    core = [p for p in source if p.name == "core.py"]
    only_template = len(source) == 1 and len(core) == 1 and _classify_scaffold(core[0])

    if only_template:
        implementation_signal = "two_layer_core_scaffold"
    elif source:
        implementation_signal = "non_scaffold_source_present"
    else:
        implementation_signal = "no_source_module"

    project_checks = {
        "manifest": (directory / "pyproject.toml").is_file(),
        "readme": (directory / "README.md").is_file(),
        "tests": bool(tests),
        "source": bool(source),
    }
    warnings = [f"missing_{key}" for key, present in project_checks.items() if not present]
    if _test_signal(tests) == "initialization_and_forward_smoke":
        warnings.append("smoke_tests_do_not_demonstrate_scientific_validity")
    if only_template:
        warnings.append("only_generic_two_layer_core_detected")

    return {
        "name": name,
        "path": _relative(root, directory),
        "implementation_signal": implementation_signal,
        "test_signal": _test_signal(tests),
        "scientific_validation": SCIENTIFIC_STATUS,
        "manifest_present": project_checks["manifest"],
        "readme_present": project_checks["readme"],
        "source_files": [_relative(root, file) for file in source],
        "test_files": [_relative(root, file) for file in tests],
        "source_sha256": {_relative(root, p): _digest(p) for p in source},
        "warnings": warnings,
    }


def build_inventory(root: Path, *, commit: str = "") -> dict[str, Any]:
    root = root.resolve()
    projects_root = root / "projects"
    if not projects_root.is_dir():
        raise ValueError("Ainglys projects/ directory does not exist")
    directories = sorted(
        p for p in projects_root.iterdir()
        if p.is_dir() and not p.is_symlink() and not p.name.startswith(".")
    )
    if not directories:
        raise ValueError("No Ainglys research projects found")
    rows = [inspect_project(root, project) for project in directories]
    signals = dict(sorted(Counter(row["implementation_signal"] for row in rows).items()))
    warnings = sum(len(row["warnings"]) for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_kind": "static_repository_evidence_not_scientific_certification",
        "repository": "NeuralBlitz/Ainglys",
        "source_commit": commit or "not_provided",
        "totals": {
            "projects": len(rows),
            "implementation_signals": signals,
            "missing_readmes": sum(not row["readme_present"] for row in rows),
            "missing_manifests": sum(not row["manifest_present"] for row in rows),
            "missing_tests": sum(not row["test_files"] for row in rows),
            "warning_count": warnings,
            "scientifically_verified_by_this_audit": 0,
        },
        "projects": rows,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    total = audit["totals"]
    out = [
        "# Ainglys research evidence inventory",
        "",
        f"Source repository: `{audit['repository']}`  ",
        f"Source commit: `{audit['source_commit']}`  ",
        f"Projects: **{total['projects']}**  ",
        f"Missing READMEs: **{total['missing_readmes']}**  ",
        "",
        "**This is a static evidence inventory, not a scientific validation or novelty assessment.**",
        "A passing shape test demonstrates only the tested tensor contract, not causality,",
        "generalization, benchmark superiority, reproducibility, or deployment readiness.",
        "Source digests and per-project file lists are in the JSON artifact.",
        "",
        "| Project | Source signal | Test evidence | README | Scientific validation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in audit["projects"]:
        out.append(
            f"| `{row['name']}` | {row['implementation_signal']} "
            f"| {row['test_signal']} | {'yes' if row['readme_present'] else 'no'} "
            f"| {row['scientific_validation']} |"
        )
    out.extend([
        "",
        "## Evidence needed before promoting research maturity",
        "",
        "An independent reviewer should identify a falsifiable claim, link an executable",
        "protocol to pinned data and code, document baselines, held-out or out-of-distribution",
        "evaluation, deterministic seeds, uncertainty, ablations, failure modes, and signed-off",
        "reproduction results. Code presence or a passing CI suite alone is insufficient.",
        "",
        "This inventory does not inspect the pinned `Advanced-Research` submodule or any",
        "external repositories.",
        "",
    ])
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory Ainglys research evidence conservatively")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", type=Path, help="Write machine-readable inventory here")
    parser.add_argument("--markdown", type=Path, help="Write Markdown inventory here")
    parser.add_argument("--strict-layout", action="store_true",
                        help="Fail if a project lacks source, manifest or tests")
    args = parser.parse_args(argv)
    try:
        inventory = build_inventory(args.root, commit=os.environ.get("GITHUB_SHA", ""))
    except (ValueError, OSError, UnicodeError, SyntaxError) as exc:
        print(f"Research inventory failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(inventory), encoding="utf-8")

    totals = inventory["totals"]
    print(
        f"Research inventory: {totals['projects']} projects; "
        f"{totals['missing_readmes']} missing READMEs; "
        f"{totals['missing_tests']} missing tests; "
        "zero claims independently validated by this static scan"
    )
    if args.strict_layout and any(
        not row["manifest_present"] or not row["source_files"] or not row["test_files"]
        for row in inventory["projects"]
    ):
        print("Research project layout check failed; see generated inventory", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
