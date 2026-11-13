# Research maturity and validation policy

**Scope:** only the 39 research subprojects within `NeuralBlitz/Ainglys/projects/`. The `Advanced-Research` Git submodule is pinned here but has its own independent provenance and is deliberately not audited by this tool.

## Initial repository evidence (2026-11-12, before README remediation)

A source-tree inspection identified:

| Observable repository signal | Count |
| --- | ---: |
| Subprojects with `pyproject.toml` | 39 |
| Subprojects with at least one `tests/test_*.py` | 39 |
| Subprojects with only a `core.py` implementation module | 36 |
| Subprojects with additional implementation modules | 3 |
| Subprojects without a project-level `README.md` at the initial audit | 7 |

The three packages with additional implementation modules are **ACCA**, **ADAPT**, and **qwen3_coder**. Their existence is not evidence that their research hypotheses have been independently validated. Some additional modules are exploratory or schematic. The remaining packages predominantly expose two-layer `torch.nn.Linear` test scaffolds; their tests usually check construction and tensor output shape.

The missing project READMEs were supplied by the follow-up documentation remediation. The automated `--strict-layout` CI gate now requires source code, a package manifest, tests, **and a project-level README** for every Ainglys subproject. Regenerate the inventory at the exact checkout to verify the new documentation count instead of treating this initial baseline as current.

These are *static observations*, not a complete algorithm review, model evaluation, causal identification analysis, or a reproducibility certificate. Counts should be regenerated whenever the tree changes.

## Generate a per-commit inventory

```bash
python scripts/research_inventory.py --strict-layout \
  --json test-results/research-inventory.json \
  --markdown test-results/research-inventory.md
```

The inventory records each project's source paths, source SHA-256 digests, test files, presence of a manifest and README, implementation signals, and missing-evidence warnings. This produces two deterministic, inspectable files from the checkout. `--strict-layout` causes CI failure if a project loses its manifest, source implementation, or tests.

CI runs the inventory on each PR and `main` commit as part of the extended Python suite. The JSON/Markdown files are uploaded alongside JUnit and coverage evidence in the `extended-tests-python311` artifact. The artifact identifies the GitHub Actions source SHA. Download the artifact from the exact GitHub Actions run being evaluated.

**Important:** The scanner does not execute experiments, calculate independent baselines, review paper references, or award `validated` status. The `scientific_validation` field is always `not_independently_verified`. This is intentional. GitHub CI being green does not make a claimed discovery true.

## Maturity gates for human and independent review

| Stage | Required evidence | What may be claimed |
| --- | --- | --- |
| **Specification** | Defined problem, assumptions, inputs/outputs and failure conditions | A documented proposal |
| **Scaffold** | Importable module, basic forward pass and smoke tests | Software scaffold, not research implementation |
| **Prototype** | Domain-specific algorithm, unit/invariant tests, repeatable execution protocol | An experimental implementation |
| **Evaluated** | Pinned datasets, train/test split provenance, baselines, seeds, ablations, uncertainty and negative results | Performance under the evaluated conditions |
| **Independently reproduced** | Independent execution environment, reviewer, exact commit/data lineage, reproducing results and limitations | A specific reproduced result, not universal scientific truth |
| **Release candidate** | Prior gates plus threat modeling, license clearance, versioned deployment and operational support | Release readiness for a defined use case |

A project may be in a different stage for different claims. A claim about causal discovery, for example, needs controlled interventions or a justified identification strategy and explicit assumptions; an output tensor resembling an adjacency matrix does not demonstrate causality.

## Required experimental record

For a substantive claim, provide a version-controlled, reviewer-inspectable record that contains:

1. **Claim and falsification criterion:** measurable proposition, observable outcome and predefined threshold.
2. **Exact implementation:** commit SHA, dependency lock, hardware/runtime, entrypoint and source/data license.
3. **Evaluation design:** dataset generation or collection, leakage prevention, seeds, splits, held-out units, confounders, controls and comparators.
4. **Experimental evidence:** raw results, evaluation code, uncertainty estimates, confidence intervals where justified, ablations and documented failures.
5. **Review:** independent reviewer, replication instructions, deviations from protocol and reasons for any qualified or rejected claim.
6. **Promotion decision:** dated decision for each specific claim; never promote all projects from a single smoke-test result.

## Immediate Ainglys research priorities

- **ACCA:** evaluate whether the proposed learned adjacency actually recovers known synthetic causal relationships under interventions, not merely an adjacency shape or nonzero gradient. Use controlled baselines and explicit identifiability assumptions.
- **ADAPT:** distinguish dimensionality-reduction and Euclidean interpolation demos from genuine Riemannian geodesics or persistent homology; test against established numerical implementations before making topology claims.
- **qwen3_coder:** inspect `aca.py`, `alf.py` and `fascss.py` separately, identify real executable behavior and construct tests that exercise them rather than only `CoreModule`.
- **Scaffold-only projects:** document intended algorithms and interfaces, then prioritize a few for real implementation instead of claiming that 36 smoke-tested multilayer perceptrons constitute mature research frameworks.
- **Documentation gaps:** add accurate project READMEs to the seven packages that lack one; do not auto-generate scientific claims.

References: [Research validation issue #6](https://github.com/NeuralBlitz/Ainglys/issues/6), [Ainglys technical audit](https://app.notion.com/p/41525ca447c181c39a20e61bd536bd67).

This policy is a research-governance framework, not a substitute for peer review or independent mathematical proof.
