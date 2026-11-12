"""Conservative, deterministic evidence-inventory regression tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research_inventory import build_inventory, main, render_markdown


GENERIC_CORE = """
import torch
import torch.nn as nn

class CoreModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
"""

SHAPE_TEST = """
def test_initialization():
    assert True

def test_forward():
    output = object()
    assert output is not None
"""


def project(root: Path, name: str, *, readme=True, tests=True,
            additional_code=False, manifest=True) -> Path:
    folder = root / "projects" / name
    source = folder / "src" / name
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("", encoding="utf-8")
    (source / "core.py").write_text(GENERIC_CORE, encoding="utf-8")
    if additional_code:
        (source / "specialized.py").write_text(
            "def special_algorithm(x):\n    return x * x\n", encoding="utf-8"
        )
    if manifest:
        (folder / "pyproject.toml").write_text(
            "[project]\nname = \"" + name + "\"\nversion = \"0.1.0\"\n",
            encoding="utf-8",
        )
    if readme:
        (folder / "README.md").write_text("# " + name, encoding="utf-8")
    if tests:
        test_dir = folder / "tests"
        test_dir.mkdir()
        (test_dir / "test_core.py").write_text(SHAPE_TEST, encoding="utf-8")
    return folder


def test_inventory_detects_generic_template_without_scientific_claims(tmp_path):
    project(tmp_path, "sample")
    result = build_inventory(tmp_path, commit="abc123")
    assert result["repository"] == "NeuralBlitz/Ainglys"
    assert result["source_commit"] == "abc123"
    assert result["totals"]["projects"] == 1
    assert result["totals"]["implementation_signals"] == {
        "two_layer_core_scaffold": 1
    }
    row = result["projects"][0]
    assert row["test_signal"] == "initialization_and_forward_smoke"
    assert row["scientific_validation"] == "not_independently_verified"
    assert row["source_sha256"]["projects/sample/src/sample/core.py"] == hashlib.sha256(
        GENERIC_CORE.encode("utf-8")
    ).hexdigest()
    assert "smoke_tests_do_not_demonstrate_scientific_validity" in row["warnings"]


def test_specialized_modules_are_not_mistaken_for_validation(tmp_path):
    project(tmp_path, "projectA", additional_code=True)
    report = build_inventory(tmp_path)
    item = report["projects"][0]
    assert item["implementation_signal"] == "non_scaffold_source_present"
    assert item["scientific_validation"] == "not_independently_verified"
    assert len(item["source_files"]) == 2
    assert report["totals"]["scientifically_verified_by_this_audit"] == 0


def test_missing_docs_tests_and_manifests_are_reported(tmp_path):
    project(tmp_path, "undocumented", readme=False, tests=False, manifest=False)
    item = build_inventory(tmp_path)["projects"][0]
    assert set(item["warnings"]) >= {
        "missing_readme", "missing_manifest", "missing_tests"
    }
    assert item["test_signal"] == "no_tests"
    assert main(["--root", str(tmp_path), "--strict-layout"]) == 1


def test_inventory_is_deterministic_and_sorted(tmp_path):
    project(tmp_path, "zeta")
    project(tmp_path, "alpha")
    first = build_inventory(tmp_path)
    second = build_inventory(tmp_path)
    assert first == second
    assert [x["name"] for x in first["projects"]] == ["alpha", "zeta"]
    assert first["totals"]["projects"] == 2


def test_json_and_markdown_artifacts_are_well_formed(tmp_path):
    project(tmp_path, "demo")
    output = tmp_path / "artifacts" / "research.json"
    summary = tmp_path / "artifacts" / "research.md"
    assert main([
        "--root", str(tmp_path), "--strict-layout",
        "--json", str(output), "--markdown", str(summary),
    ]) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    md = summary.read_text(encoding="utf-8")
    assert data["totals"]["projects"] == 1
    assert "not_independently_verified" in md
    assert "shape test" in md
    assert "Advanced-Research" in md


def test_discovered_only_projects_not_advanced_research_submodule(tmp_path):
    project(tmp_path, "local")
    unrelated = tmp_path / "Advanced-Research" / "other"
    unrelated.mkdir(parents=True)
    (unrelated / "core.py").write_text(GENERIC_CORE)
    assert [p["name"] for p in build_inventory(tmp_path)["projects"]] == ["local"]


def test_extra_test_functions_do_not_promote_scientific_status(tmp_path):
    folder = project(tmp_path, "nontrivial")
    (folder / "tests" / "test_extra.py").write_text(
        "def test_adversarial_case():\n    assert 1 == 1\n"
    )
    item = build_inventory(tmp_path)["projects"][0]
    assert item["test_signal"] == "additional_test_functions_present"
    assert item["scientific_validation"] == "not_independently_verified"


def test_missing_projects_directory_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="projects"):
        build_inventory(tmp_path)
    assert main(["--root", str(tmp_path)]) == 2
