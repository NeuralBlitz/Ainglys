# PyPI Publication Configuration
# ================================
# This directory contains configuration for publishing packages to PyPI

## Quick Start

### 1. Build Package
```bash
python -m build
```

### 2. Test with TestPyPI
```bash
python -m twine upload --repository testpypi dist/*
```

### 3. Publish to PyPI
```bash
python -m twine upload dist/*
```

## Automated Publishing

Publishing is automated via GitHub Actions when a new tag is pushed:

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

## Package Index

### Core Packages
- `ai-hub` - Main platform
- `acca-framework` - Attentive Causal Automata

### Research Packages
- `acml` - Adaptive Categorical Meta-Learning
- `aether-calc` - Aether-Calculus optimizer
- `gaav` - Granular Arithmetic & Visualization
- `gaft` - Granular Arithmonic Field Theory
- `cnm` - Complex Network Morphogenesis
- `hnma` - Hierarchical Neural Multi-Agent
- `mcaf` - Meta-Cognitive Architecture
- `newnn` - Neuromorphic Edge-Weighted Networks
- `gcaa` - Geometric Cellular Automata

## Version Strategy

We follow Semantic Versioning (SemVer):
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)
