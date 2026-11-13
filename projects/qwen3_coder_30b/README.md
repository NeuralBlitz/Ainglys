# Qwen3 Coder 30B research namespace

**Status:** Early-stage Ainglys research scaffold; **not** a validated reproduction of the named architecture or scientific claims.

## What is actually implemented

- `src/qwen3_coder_30b/core.py` exports `CoreModule(input_dim, hidden_dim, output_dim)`.
- The module is a small PyTorch feed-forward network with two linear layers and a ReLU activation.
- `src/qwen3_coder_30b/__init__.py` exports `CoreModule`; `pyproject.toml` defines the Python package.

Example, after installing the package in a suitable research environment:

```python
import torch
from qwen3_coder_30b import CoreModule

model = CoreModule(input_dim=10, hidden_dim=20, output_dim=5)
outputs = model(torch.randn(2, 10))
assert outputs.shape == (2, 5)
```

## Verification evidence

`tests/test_core.py` checks construction and tensor shape. These tests are **software smoke tests**, not evidence of model equivalence, scientific validity, causal inference, robustness or generalization.

## Explicit limitations

This package does not contain a 30-billion-parameter model, pretrained Qwen Coder weights, or a Qwen-compatible tokenizer/inference stack.

No comparative benchmark, reproducible training protocol, independent model evaluation, dataset provenance, or published scientific validation is provided by this package's current source tree. Its name is a research label and does not imply endorsement or affiliation with a model provider.

## Next engineering and research gates

1. Specify a falsifiable, package-specific research objective and real algorithm interface.
2. Implement domain-specific code separate from the generic `CoreModule`, with tests for algorithmic invariants and known failure cases.
3. Define pinned datasets, baselines, seeds, splits, compute budgets and expected evaluation metrics.
4. Reproduce results independently and report negative outcomes and limits.

See the [Ainglys research maturity policy](../../docs/RESEARCH_MATURITY.md), [issue #6](https://github.com/NeuralBlitz/Ainglys/issues/6), and [root licensing decision](https://github.com/NeuralBlitz/Ainglys/issues/7). Package metadata is not by itself a settled license grant for the entire Ainglys repository.
