# Qwen3 Coder research namespace

**Status:** Experimental Ainglys research code, not a reproduction of the Qwen3 Coder pretrained model or its published capabilities. This directory contains several independent prototypes in addition to a generic model scaffold.

## Implemented modules and interfaces

| File | Observed implementation | Evidence boundary |
| --- | --- | --- |
| `src/qwen3_coder/core.py` | `CoreModule`: two-layer PyTorch MLP with ReLU | Initialization and output-shape smoke tests |
| `src/qwen3_coder/aca.py` | `AttentiveCausalAutomata`: encoder, attention layers, tensor-valued adjacency-like output and decoder | No independent causal identification or intervention validation |
| `src/qwen3_coder/alf.py` | `AdaptiveLearningFramework`: task-indexed linear blocks and an adapter | No demonstrated meta-learning benchmark or generalization evaluation |
| `src/qwen3_coder/fascss.py` | `SelfConfiguringModule` and `FASCSS`: soft-gated branches with gradient-based adaptation | No independent architecture-search, robustness or self-configuration evidence |

Only `CoreModule` is exported from package-level `__init__.py`. Specialized classes may be imported directly from their modules while experimental.

```python
import torch
from qwen3_coder import CoreModule
from qwen3_coder.fascss import FASCSS

baseline = CoreModule(10, 20, 5)
assert baseline(torch.randn(2, 10)).shape == (2, 5)

# The prototype's stacked blocks now handle input_dim != hidden_dim.
prototype = FASCSS(input_dim=10, hidden_dim=20, output_dim=5, num_modules=3)
assert prototype(torch.randn(2, 10)).shape == (2, 5)
```

The examples check tensor contracts only. They are not guarantees of complete training, calibrated outputs or convergence.

## What is not established

- The source tree does **not** provide Qwen3 Coder pretrained weights, inference engine, training dataset, tokenizer or performance parity.
- The repository package description references research papers, but paper-by-paper implementations, source-attributed claims and matching validation results have not been independently verified.
- An adjacency-shaped tensor or trained gating network is not, by itself, evidence of a causal graph, novel learning paradigm, self-improvement or state-of-the-art performance.
- `tests/test_core.py` checks the generic baseline; `tests/test_fascss.py` now separately tests FASCSS stack dimensions, gradients and a local adaptation step. ACA and ALF still lack dedicated evaluations.
- [Ainglys PR #28](https://github.com/NeuralBlitz/Ainglys/pull/28) corrected two software defects: stacked blocks now receive the prior output width, and the PyTorch `nn.Module.modules()` method is no longer shadowed. These passing regression tests establish tensor-contract behavior, **not** a validated adaptive architecture or independent scientific result.

## Priority engineering work

1. Add dedicated tests for each specialized module's outputs, gradients, edge cases and deterministic behavior.
2. Record which hypothesis and reference each module attempts to implement, separating source text from executable behavior.
3. Construct controlled baselines and seeded evaluations for causal discovery, adaptive learning and gated architectures.
4. Require independent scientific replication and an explicit limitations report before promoting any claims.

See [research maturity requirements](../../docs/RESEARCH_MATURITY.md), [research review issue #6](https://github.com/NeuralBlitz/Ainglys/issues/6) and [root licensing decision issue #7](https://github.com/NeuralBlitz/Ainglys/issues/7). This namespace does not assert any endorsement or affiliation with a model provider.
