# AI Hub - Advanced ML Research Platform

> **Release status (2026-11-11):** Research software under active security and reliability remediation. Do not expose the API or training services publicly until deployment security and full CI have been independently validated. Historic test and coverage claims below are not current proof of release readiness. See [remediation pull request #2](https://github.com/NeuralBlitz/Ainglys/pull/2).


[![Ainglys CI](https://github.com/NeuralBlitz/Ainglys/actions/workflows/ci.yml/badge.svg)](https://github.com/NeuralBlitz/Ainglys/actions/workflows/ci.yml)
[Research maturity tracking](https://github.com/NeuralBlitz/Ainglys/issues/6)
[Python 3.11+ package requirement](pyproject.toml)
[Root licensing decision pending](https://github.com/NeuralBlitz/Ainglys/issues/7)

A unified AI/ML platform with CLI, REST API, ML components, and comprehensive visualization/interpretability features. Built on the **ACCA** (Attentive Causal Automata) framework and integrated with the Advanced Research orchestration system.

> **Current engineering status:** The core API/UI deployment and configured CI checks are actively maintained; research packages and experimental services are not uniformly validated. See the [live CI evidence](https://github.com/NeuralBlitz/Ainglys/actions/workflows/ci.yml), [research maturity issue](https://github.com/NeuralBlitz/Ainglys/issues/6), and [deployment issue](https://github.com/NeuralBlitz/Ainglys/issues/4).

## Research evidence and maturity inventory

Ainglys currently contains **39 research subprojects**. Many are preliminary two-layer ML scaffolds with initialization/output-shape tests; these smoke tests do not demonstrate a paper's claimed algorithm, causal discovery or scientific reproducibility. The [research maturity policy](docs/RESEARCH_MATURITY.md) identifies evidence standards and next experiments.

To inspect each project's actual source modules, tests, SHA-256 file digests, documentation gaps and conservative implementation signals:

```bash
python scripts/research_inventory.py --strict-layout \
  --json test-results/research-inventory.json \
  --markdown test-results/research-inventory.md
```

CI automatically uploads these inventory files with the extended-suite test evidence for each commit. No project is automatically classified as scientifically validated by this scan. See [research review issue #6](https://github.com/NeuralBlitz/Ainglys/issues/6).

## Overview

AI Hub is a research-oriented machine learning platform that combines:
- **ML Engine**: PyTorch-based training with the ACCA framework
- **REST API**: FastAPI service for model management
- **Async API**: WebSocket-based real-time training and collaboration
- **Rich CLI**: Beautiful terminal interface with progress tracking
- **Advanced Research Integration**: Multi-system orchestration with 87 research packages
- **Visualization Suite**: Interactive Plotly dashboards, training animations, attention visualization
- **Interpretability**: Saliency maps, SHAP/LIME integration, interactive dashboards
- **Jupyter Widgets**: Real-time training widgets for notebooks
- **Distributed Training**: Ray-based multi-node training
- **AutoML**: Grid, Random, Bayesian, and Population-Based Training optimization
- **Model Serving**: Dynamic batching, canary/blue-green deployments
- **Automated Testing**: GitHub Actions smoke CI; full coverage remains unverified

## 🆕 What's New

### February 2026 - Complete Platform

✅ **Phase 1: Core Infrastructure** ✅
- Research Integration Hub (ADAG, ANAS, ACML)
- Model Registry with semantic versioning
- A/B Testing framework
- Experiment Tracking (MLflow, W&B)
- OAuth2/JWT Security
- API Key Management
- Package Template Generator

✅ **Phase 2: Advanced Infrastructure** ✅
- Async API with WebSockets
- Real-time training collaboration
- Ray Distributed Training
- Multi-GPU support
- AutoML (Grid, Random, Bayesian, PBT)
- Model Serving with dynamic batching
- Canary and Blue-Green deployments

✅ **Phase 3: Visualization & Analysis** ✅
- Interactive Plotly dashboards
- Training animations (GIF export)
- Attention heatmaps for transformers
- Model architecture diagrams
- Model comparison tools (radar charts)
- Feature importance visualization
- Confusion matrices & ROC curves
- t-SNE/PCA embedding visualization
- Experiment timeline (Gantt charts)
- Hyperparameter parallel coordinates
- **Interpretability Dashboard**: Saliency maps, Integrated Gradients, SmoothGrad
- **Jupyter Training Widgets**: Real-time metrics, GPU monitoring, interactive controls
- Automated paper generation (LaTeX)
- Dataset versioning (Git-like)

✅ **Phase 4: Production Deployment** ✅
- Docker Compose with 10+ services
- Prometheus/Grafana monitoring
- Nginx load balancer
- Complete CI/CD pipeline

### Platform Statistics
- **21 Core Modules** (~9,500 lines)
- **87 Research Packages** integrated
- **Historical test reports** (current coverage unverified)
- **48 Features** fully implemented
- **Docker deployment** ready

## Features

### Core Capabilities
- **Neural Network Training**: Deep learning with customizable architectures
- **Model Persistence**: Save, load, and version trained models
- **RESTful API**: Complete HTTP API for remote operations
- **Async API**: WebSocket-based real-time training
- **Rich CLI**: Terminal UI with tables, progress bars, and colored output
- **Configuration**: Dot-notation config system with JSON storage
- **Multi-Modal Support**: Integration with research workflow systems

### Advanced Features
- **ACCA Framework**: Attentive Causal Automata for structure learning
- **Geometric Deep Learning**: Riemannian manifold optimization
- **Research Pipeline**: Automated Ideas → Synthesis → Implementation workflow
- **Context Injection**: Priority-based context management for AI systems
- **Learning Analytics**: xAPI/LRS integration for tracking research activities
- **Distributed Training**: Ray-based multi-node, multi-GPU training
- **AutoML**: Multiple optimization strategies (Grid, Random, Bayesian, PBT)
- **Model Serving**: Production-grade serving with A/B testing
- **Advanced Visualization**: Training animations, attention maps, model comparisons
- **Interpretability**: Saliency maps, feature importance, SHAP/LIME integration
- **Jupyter Integration**: Interactive widgets for real-time training monitoring

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AI Hub                                │
├─────────────────────────────────────────────────────────────┤
│  CLI (Rich)  │  REST API (FastAPI)  │  ML Engine (PyTorch) │
├─────────────────────────────────────────────────────────────┤
│  Async API   │  WebSocket Real-Time │  Distributed (Ray)   │
├─────────────────────────────────────────────────────────────┤
│              ACCA Framework (CoreModule)                     │
│         ┌──────────────┬──────────────┐                     │
│         │    MMDG      │   Causal     │                     │
│         │    Graph     │  Discovery   │                     │
│         └──────────────┴──────────────┘                     │
├─────────────────────────────────────────────────────────────┤
│           Advanced Visualization & Interpretability          │
│  ┌─────────────┬──────────────┬──────────────────────┐     │
│  │  Plotly     │  Attention   │   Interpretability   │     │
│  │Dashboards   │Visualization │   Dashboard          │     │
│  └─────────────┴──────────────┴──────────────────────┘     │
├─────────────────────────────────────────────────────────────┤
│              Advanced Research System                        │
│  ┌─────────────┬──────────────┬──────────────────────┐     │
│  │  LRS Agents │   Opencode   │   NeuralBlitz v50    │     │
│  │   (xAPI)    │   (Research) │   (Geometric DL)     │     │
│  └─────────────┴──────────────┴──────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Scoped API client credentials (opt-in)

Ainglys can authenticate up to 32 independently provisioned client keys using an optional `AI_HUB_API_KEYS_JSON` registry of SHA-256 fingerprints and explicit route scopes. Configure read-only, inference, training and model-management clients separately; attempts outside an enrolled client's permissions return HTTP **403**. Invalid/empty keyring configuration fails closed with **503** and **never** falls back to the legacy key.

The existing `AI_HUB_API_KEY` one-key mode remains the default for compatibility. When the scoped registry is configured, that shared key is **not** accepted by the server unless also explicitly enrolled. Per-key request budgets remain process-local.

See [Scoped API Keys: setup, permissions, Docker opt-in and revocation](docs/SCOPED_API_KEYS.md) for safe token provisioning and honest limitations. Full production identity management, distributed quotas, permissions auditing and controlled secret rotation remain work in progress.

## Authenticated request budgets (single-process)

The Ainglys API applies fail-closed request budgets **after** validating `X-API-Key`. Limits are process-local, shared by every client using the same configured key, and reset on key rotation. The unauthenticated `/health` and `/` endpoints are not metered.

| Environment setting | Default | Permitted range |
| --- | ---: | --- |
| `AI_HUB_REQUESTS_PER_MINUTE` | 600 | 1–10,000 |
| `AI_HUB_TRAININGS_PER_MINUTE` | 60 | 1 up to `AI_HUB_REQUESTS_PER_MINUTE` |

The training budget is a subset of the total budget and covers both `POST /train` and `POST /train/supervised`. A rejected attempt returns HTTP **429** with `Retry-After` in seconds. An invalid quota setting returns **503** instead of silently turning off limits. Failed authentication is rejected before quota accounting.

For local development:

```bash
export AI_HUB_REQUESTS_PER_MINUTE=600
export AI_HUB_TRAININGS_PER_MINUTE=60
```

These controls protect **one API process only**. They are not a distributed quota, a multi-user rate policy, an identity system, or a resource-consumption guarantee. For multiworker/public deployments, implement a centralized store or gateway limiter, per-principal accounting, scoped authorization, maximum job-runtime enforcement and operational monitoring before claiming production readiness. See [security issue #3](https://github.com/NeuralBlitz/Ainglys/issues/3).

## API authentication (required)

Sensitive API operations now require a server-configured API key. Set the same strong value in the API deployment and the CLI/Streamlit environment. The server intentionally refuses requests when the key is missing or shorter than 32 characters.

```bash
export AI_HUB_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
```

For Docker Compose production deployment, also provision unique `JWT_SECRET_KEY`, `DB_PASSWORD` and `GRAFANA_PASSWORD` values in a securely managed environment. Do not commit these secrets. Do not run the optional Jupyter or internal data services on public interfaces.

The API currently provides one shared-key authorization boundary, not a production-grade multi-user identity service. Granular roles, quotas, and stronger identity integration remain outstanding.

## Installation

### Quick Install
```bash
pip install -e .
```

### Development Install
```bash
pip install -e ".[dev]"
```

### Advanced Research System
```bash
cd Advanced-Research
pip install -e ".[all]"
```

### Install ACCA Framework
```bash
cd projects/acca
pip install -e .
```

## Quick Start

### CLI Usage

Check system status:
```bash
ai-hub status
```

Train a model locally:
```bash
ai-hub local train --epochs 10 --save-to my_model.pt
```

Make predictions:
```bash
ai-hub local predict my_model.pt --data 0.1 --data 0.2 --data 0.3
```

### API Usage

Start the API server:
```bash
ai-hub api start --port 8000
```

Check API status:
```bash
ai-hub api status
```

Train via API:
```bash
ai-hub model train --epochs 10
```

Save the model:
```bash
ai-hub model save my_model
```

List saved models:
```bash
ai-hub model list
```

Make predictions:
```bash
ai-hub model predict --data 0.1 --data 0.2 --data 0.3
```

### Train from an actual labeled dataset (new)

The legacy `ai-hub model train` command trains on random synthetic labels as a **demo**. To train on your own numeric classification records with a disjoint stratified holdout validation set, supply JSON:

```bash
ai-hub model train-data training-example.json
```

The new authenticated `POST /train/supervised` endpoint requires 16–512 records and 2–32 classes, validates input dimensions and values, and reports the training-set size, held-out validation-set size, accuracy, and cross-entropy. A fixed seed reproduces the index split; a successful run replaces the active in-memory model. No real-world performance claim is implied by the demo data.

Read [Supervised Training: API, CLI, Schema and Limitations](docs/SUPERVISED_TRAINING.md) for sample JSON, secure API configuration, evaluation methodology and scientific limitations.

### Streamlit supervised training dashboard

The Streamlit interface provides a **Supervised Dataset** tab. Upload a JSON dataset in the same schema used by `ai-hub model train-data`. The dashboard validates numeric features and class labels before sending the upload, then displays training/holdout sample counts, held-out accuracy and cross-entropy. It refuses files larger than 1 MB and requires HTTPS for remote API connections (HTTP is allowed on localhost or the internal Docker `api` service).

```bash
docker compose --profile ui up --build -d
# Open http://127.0.0.1:8501
```

The dashboard inherits `AI_HUB_API_KEY` and `AI_HUB_API_URL` from its environment. It is a prototype interface, not a multi-user access-control system; keep it on a trusted local network.

### Direct API Calls

```bash
# Start server
uvicorn ai_hub.api:app --port 8000

# Train
curl -X POST http://localhost:8000/train \
  -H "X-API-Key: $AI_HUB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"epochs": 10, "input_dim": 128, "hidden_dim": 256, "output_dim": 10}'

# Predict
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: $AI_HUB_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"data": [[0.1, 0.2, 0.3]]}'

# List models
curl -H "X-API-Key: $AI_HUB_API_KEY" http://localhost:8000/models
```

### Advanced Visualization Example

```python
from ai_hub.advanced_viz import ModelComparator, TrainingAnimator

# Compare multiple models
comparator = ModelComparator()
comparator.add_model("Model_A", {"accuracy": 0.95, "f1": 0.94})
comparator.add_model("Model_B", {"accuracy": 0.93, "f1": 0.92})
fig = comparator.create_radar_chart()
fig.write_html("model_comparison.html")

# Create training animation
animator = TrainingAnimator()
animator.create_training_animation(history, "training_progress.gif")
```

### Interpretability Dashboard

```python
from ai_hub.interpretability import InterpretabilityDashboard

# Create dashboard
dashboard = InterpretabilityDashboard(model)

# Generate explanations
saliency = dashboard.compute_saliency(input_image)
importance = dashboard.compute_feature_importance(X_train, y_train)

# Create interactive dashboard
dashboard.create_interactive_dashboard(
    input_sample, X_train, y_train,
    save_path="interpretability_dashboard.html"
)
```

### Jupyter Training Widgets

```python
from ai_hub.training_widgets import TrainingDashboard

# Create dashboard
dashboard = TrainingDashboard()
dashboard.display()

# Update during training
for epoch in range(100):
    metrics = trainer.train_epoch()
    dashboard.update_metrics(metrics, epoch, total_epochs=100)
```

## Advanced Research System

### Initialize the Research Framework
```bash
cd Advanced-Research
advanced-research init --config config.yaml
```

### Context Injection Demo
```bash
advanced-research context-demo --user researcher-1
```

### Check Integration Status
```bash
advanced-research integrations-status --config config.yaml
```

### Run Automated Pipeline
```python
from advanced_research.unified.api import UnifiedResearchSystem, SystemMode
from advanced_research.pipeline.automated import AutomatedResearchPipeline, PipelineConfig

# Initialize system
config = {
    "lrs": {"enabled": True, "endpoint": "http://localhost:8080/xapi/"},
    "opencode": {"enabled": True, "workspace": "advanced-research"},
    "neuralblitz": {"enabled": True, "backend": "jax"}
}

system = UnifiedResearchSystem(config)
await system.initialize()

# Create pipeline
pipeline_config = PipelineConfig(
    name="Novel Architecture Research",
    description="Research on geometric deep learning",
    stages=[PipelineStage.IDEATION, PipelineStage.SYNTHESIS, PipelineStage.IMPLEMENTATION],
    auto_transition=True
)

pipeline = AutomatedResearchPipeline(system, pipeline_config)
result = await pipeline.execute_pipeline(
    user_id="researcher-1",
    initial_ideas=["Riemannian attention mechanisms", "Manifold optimization"]
)
```

## Configuration

Configuration is stored in `~/.ai_hub/config.json`.

```bash
# Show configuration
ai-hub config show

# Get a value
ai-hub config get api.port

# Set a value
ai-hub config set api.port 8080
```

Default configuration:
```json
{
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": false
  },
  "model": {
    "default_input_dim": 128,
    "default_hidden_dim": 256,
    "default_output_dim": 10,
    "default_epochs": 10,
    "models_dir": "models"
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "ai_hub.log"
  }
}
```

## ACCA Framework

The **Attentive Causal Automata** framework provides advanced ML capabilities:

### Multi-Modal Dynamic Graph (MMDG)
```python
from acca import MultiModalDynamicGraph

# Create dynamic graph
mmdg = MultiModalDynamicGraph(
    num_nodes=16,
    node_dim=64,
    num_modalities=1,
    attention_heads=8
)

# Forward pass
node_features, attention_weights = mmdg(input_data)
```

### Causal Discovery
```python
from acca import CausalDiscovery

# Learn causal structure
causal = CausalDiscovery(num_variables=16, hidden_dim=64)
adj_matrix = causal(data)
```

### Topology Optimization
```python
from acca import TopologyOptimizer

# Optimize network architecture
topology = TopologyOptimizer(num_nodes=16)
edge_weights = topology()
```

## Research Projects

The root repository tracks **39 local research subprojects** plus a linked Advanced-Research submodule; the historical '87 projects' claim has not been reconciled to validated, independently working packages. Representative concepts include:

### Core Frameworks

| Project | Description | Location |
|---------|-------------|----------|
| **ACCA** | Attentive Causal Automata - Multi-modal dynamic graphs with causal discovery | `projects/acca/` |
| **ACML** | Adaptive Categorical Meta-Learning - Functorial meta-learning with natural gradients | `Advanced-Research/Ideas/projects/acml/` |
| **THGC** | Thermodynamic Hyper-Graph Computation - Non-equilibrium information processing | `projects/glm_47/` |
| **PART** | Polymorphic Adaptive Resonance Theory - Multi-scale optimization with thermodynamic constraints | `projects/deepseek_r1/` |
| **GAFT** | Granular Arithmonic Field Theory - Sub-symbolic spatiotemporal computation | `Advanced-Research/Ideas/projects/gaft/` |

### Optimization & Learning

| Project | Description | Location |
|---------|-------------|----------|
| **Aether-Calc** | Temporal Hyper-Relaxation optimizer for non-convex landscapes | `Advanced-Research/Ideas/projects/aether_calc/` |
| **GAAV** | Granular Arithmetic & Algorithmic Visualization - Uncertainty-aware computation | `Advanced-Research/Ideas/projects/gaav/` |
| **NeuralBlitz** | Geometric deep learning on Riemannian manifolds | `Advanced-Research/src/neuralblitz/` |
| **HoloFlow** | Holographic Flow Networks | `projects/holoflow/` |
| **CNM** | Complex Network Morphogenesis - Network growth through morphogenetic principles | `Advanced-Research/Ideas/projects/cnm/` |

### Advanced Architectures

| Project | Description | Location |
|---------|-------------|----------|
| **ICSEE** | Inter-Cross-Synthesis Engine | `projects/icsee/` |
| **NAM** | Neural Algebraic Manifolds | `projects/nam/` |
| **Tenspec** | Tensor Specification Framework | `projects/tenspec/` |
| **DeepSeek R1** | Research frameworks with formal blueprints | `projects/deepseek_r1/` |
| **HNMA** | Hierarchical Neural Multi-Agent Systems - Multi-agent coordination | `Advanced-Research/Ideas/projects/hnma/` |
| **MCAF** | Meta-Cognitive Architecture Framework - Self-reflective AI systems | `Advanced-Research/Ideas/projects/mcaf/` |
| **NEWNN** | Neuromorphic Edge-Weighted Neural Networks - Biologically-inspired networks | `Advanced-Research/Ideas/projects/newnn/` |
| **GCAA** | Geometric Cellular Automata Architectures - CA on non-Euclidean manifolds | `Advanced-Research/Ideas/projects/gcaa/` |

### All Projects

See `Advanced-Research/Ideas/projects/` for the complete list of **87 research projects** with:
- **5,000+ lines** of novel ML/AI implementation code
- **4,500+ lines** of comprehensive test code
- **170+ tests** with 95%+ pass rate
- Mathematical foundations (category theory, information geometry, field theory)
- research-oriented implementations with type hints and documentation

### Featured New Packages

Recently implemented from research papers:

| Package | Innovation | Status |
|---------|-----------|--------|
| **ACML** | Category-theoretic meta-learning with natural gradients | ✅ Complete + Tests |
| **Aether-Calc** | Temporal Hyper-Relaxation optimizer with entropy feedback | ✅ Complete + Tests |
| **GAAV** | Granular arithmetic G=(v,δ,σ) with uncertainty propagation | ✅ Complete + Tests |
| **GAFT** | Sub-symbolic field computation with particle interactions | ✅ Complete + Tests |
| **CNM** | Network morphogenesis | ✅ Complete + Tests |
| **HNMA** | Hierarchical multi-agent coordination | ✅ Complete + Tests |
| **MCAF** | Meta-cognitive AI systems | ✅ Complete + Tests |
| **NEWNN** | Neuromorphic edge-weighted networks | ✅ Complete + Tests |
| **GCAA** | Geometric cellular automata | ✅ Complete + Tests |

### Quick Installation for Research Projects

```bash
# Install Core Frameworks
cd Advanced-Research/Ideas/projects/acml && pip install -e .  # Meta-Learning
cd ../aether_calc && pip install -e .                         # Optimization
cd ../gaav && pip install -e .                                # Granular Arithmetic
cd ../gaft && pip install -e .                                # Field Theory
cd ../../../projects/acca && pip install -e .                 # Causal Automata

# Install Advanced Architectures
cd ../Advanced-Research/Ideas/projects/cnm && pip install -e .   # Network Morphogenesis
cd ../hnma && pip install -e .                                 # Multi-Agent Systems
cd ../mcaf && pip install -e .                                 # Meta-Cognitive
cd ../newnn && pip install -e .                                # Neuromorphic
cd ../gcaa && pip install -e .                                 # Cellular Automata
```

## File snapshot manager integrity

The separate `DatasetVersionManager` file-snapshot workflow validates manifest paths and SHA-256 digests before **both** restore and diff operations. It denies symlinked sources/targets, excludes hidden credentials and VCS folders, and bounds snapshot size. See [File Snapshot Integrity and Limitations](docs/FILE_SNAPSHOT_INTEGRITY.md). It is not a signed, multi-user, or fully transactional backup service.

## Safe dataset versioning

The `DatasetRepository` storage path now uses only non-executable NumPy, Pandas JSON or standard JSON snapshots. New versions are SHA-256 checked, capped at 32 MiB, and reject unsafe filenames and symlinks. **Historic `.pkl` versions no longer load automatically**, even if stored in a local repository. Read [Safe Dataset Snapshots and Legacy Migration](docs/SAFE_DATASET_SNAPSHOTS.md) before reopening an older dataset registry.

## Deployment preflight (local core stack)

The supported local runtime is the loopback-bound `api` service plus optional `ui` profile. Before running Docker Compose, verify the **rendered** configuration, including credential presence, loopback-only ports, restricted container capabilities, UI dependency ordering and model-volume persistence:

```bash
export AI_HUB_API_KEY='<random credential with 32 or more characters>'
python scripts/verify_compose.py
python scripts/verify_compose.py --ui
docker compose --profile ui up --build -d
```

[Full core deployment preflight guide](docs/COMPOSE_PREFLIGHT.md). The separate `docker-compose.full.experimental.yml` includes unverified services and missing assets and must not be presented as a release-ready stack. A successful preflight is **configuration validation**, not proof of live services or production safety.

## Security scanning and dependency maintenance

Ainglys includes a Python CodeQL analysis workflow (`.github/workflows/codeql.yml`) on pull requests, `main` changes and a weekly schedule. GitHub Dependabot checks the uv dependency lock, ACCA's Python dependencies, GitHub Actions and Docker weekly. The [security policy](SECURITY.md) describes vulnerability reporting and supported development branches.

These are **detection and maintenance measures**, not a guarantee that the research software is production-safe. Branch protection and required-status-check enforcement still need repository settings changes; see [issue #5](https://github.com/NeuralBlitz/Ainglys/issues/5).

## Testing and Verification

The latest completed core smoke workflow prior to this revision passed **23 tests on Python 3.11 and 23 tests on Python 3.12** (November 11, 2026). This establishes a limited clean-install and core API/ACCA check, not full-repository correctness or coverage. New extended tests and lockfile/Compose checks run in GitHub Actions, with their current status visible on the [Actions page](https://github.com/NeuralBlitz/Ainglys/actions).

```bash
# Core regression tests
python -m pytest -q tests/test_api.py tests/test_ml_engine.py \
  projects/acca/tests/test_core.py projects/acca/tests/test_causal_math.py

# Extended local AI Hub and ACCA tests
python -m pytest tests/ projects/acca/tests/ --cov=ai_hub --cov-report=term-missing

# Verify the dependency resolution (uv required)
uv lock --check
```

Prior reports (`TEST_REPORT_FINAL.md`, `TESTING_COMPLETE.md`) are historical; their test counts and coverage claims have not been independently reverified for the current revision. [Follow-up issue #5](https://github.com/NeuralBlitz/Ainglys/issues/5) tracks full-suite assurance.

## API Endpoints

### Basic Endpoints
- `GET /` - API info
- `GET /status` - Check model status
- `POST /train` - Train a new model
- `POST /predict` - Make predictions
- `POST /save` - Save current model
- `POST /load/{name}` - Load a saved model
- `GET /models` - List all models
- `DELETE /models/{name}` - Delete a model

### Integration Hub
- `POST /integration/compose` - Compose architecture
- `GET /integration/report` - Get integration report

### Model Registry
- `POST /registry/models` - Register model
- `GET /registry/models` - List models
- `GET /registry/models/{name}/versions` - List versions
- `POST /registry/ab-test` - Create A/B test

### Experiment Tracking
- `POST /experiments/start` - Start experiment
- `POST /experiments/log` - Log metrics
- `POST /experiments/end` - End experiment
- `GET /experiments/compare` - Compare runs

### Security
- `POST /auth/token` - Get access token
- `POST /auth/api-key` - Generate API key
- `GET /auth/verify` - Verify token

### Async/WebSocket
- `POST /async/train` - Start async training
- `GET /async/status/{job_id}` - Check training status
- `WS /ws/train/{job_id}` - Real-time training updates
- `WS /ws/collaborate/{room_id}` - Real-time collaboration

## Project Structure

```
.
├── src/ai_hub/              # AI Hub core (21 modules)
│   ├── api.py               # FastAPI application
│   ├── async_api.py         # Async API & WebSockets
│   ├── cli.py               # Rich CLI
│   ├── config.py            # Configuration management
│   ├── integration.py       # Research integration
│   ├── experiment_tracking.py # Experiment tracking
│   ├── model_registry.py    # Model versioning
│   ├── security.py          # Authentication
│   ├── distributed.py       # Ray distributed training
│   ├── automl.py            # Hyperparameter optimization
│   ├── serving.py           # Model serving
│   ├── visualization.py     # Basic visualization
│   ├── advanced_viz.py      # Advanced visualization
│   ├── interpretability.py  # Model interpretability
│   ├── training_widgets.py  # Jupyter widgets
│   ├── paper_generator.py   # LaTeX paper generation
│   ├── dataset_versioning.py # Dataset versioning
│   ├── logger.py            # Logging setup
│   ├── ml_engine.py         # ML engine
│   └── __init__.py          # Package initialization
│
├── tests/                   # Comprehensive test suites
│   ├── test_api.py          # API tests
│   ├── test_ml_engine.py    # ML engine tests
│   ├── test_phase1_comprehensive.py # Phase 1 tests
│   ├── test_phase2_comprehensive.py # Phase 2 tests
│   ├── test_phase3_comprehensive.py # Phase 3 tests
│   └── benchmarks/          # Performance benchmarks
│
├── examples/                # Complete usage examples
│   ├── phase1_complete_examples.py
│   ├── phase2_advanced_examples.py
│   ├── phase3_advanced_visualization.py
│   ├── phase3_interpretability_examples.py
│   └── phase3_jupyter_widgets_examples.py
│
├── projects/                # Research projects (40+ packages)
│   ├── acca/               # ACCA framework + tests
│   ├── holoflow/           # Holographic Flow + tests
│   ├── glm_47/             # GLM-4.7 frameworks + tests
│   ├── deepseek_r1/        # DeepSeek R1 + tests
│   └── ...                 # 36 more research packages
│
├── Advanced-Research/       # Research orchestration system
│   ├── src/advanced_research/
│   │   ├── unified/api.py   # Main orchestration API
│   │   ├── workflow/        # Research workflows
│   │   ├── pipeline/        # Automated pipelines
│   │   ├── neuralblitz/     # Geometric computation
│   │   └── core/            # Integrations & context
│   └── Ideas/projects/      # 48 research packages + tests
│       ├── acml/           # Category-theoretic meta-learning
│       ├── aether_calc/    # Aether-Calculus optimizer
│       ├── gaav/           # Granular arithmetic
│       ├── gaft/           # Field theory computation
│       ├── cnm/            # Network morphogenesis
│       ├── hnma/           # Multi-agent systems
│       ├── mcaf/           # Meta-cognitive architecture
│       ├── newnn/          # Neuromorphic networks
│       ├── gcaa/           # Cellular automata
│       └── ...             # 39 more packages
│
├── docker-compose.yml       # Production deployment
├── Dockerfile               # Main container
├── monitoring/              # Prometheus & Grafana
├── docs/                    # Documentation
├── models/                  # Saved models
├── pyproject.toml
└── README.md
```

**Statistics:**
- **87 packages** total
- **170+ tests** (95%+ passing)
- **9,500+ lines** of implementation
- **5,000+ lines** of tests
- **Type hint coverage not yet measured**
- **Comprehensive documentation**

## Docker Deployment

### Minimal local deployment (supported configuration)

Set a strong API key in your shell or secret manager, then start the API:

```bash
export AI_HUB_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
docker compose config --quiet
docker compose up --build -d
curl --fail http://127.0.0.1:8000/health
curl -H "X-API-Key: $AI_HUB_API_KEY" http://127.0.0.1:8000/models
```

The default `docker-compose.yml` (also available as `compose.core.yml`) exposes the API **only on loopback** at port 8000 and uses a persistent model volume. The optional Streamlit UI is enabled with `docker compose --profile ui up --build -d` and is exposed on loopback port 8501.

The previous larger 10-service stack is preserved in `docker-compose.full.experimental.yml`, **not production validated**. It references additional infrastructure components and configuration assets that still require implementation and verification. Do not use it for a public deployment. See [deployment tracking issue #4](https://github.com/NeuralBlitz/Ainglys/issues/4).

The application currently demonstrates training with synthetic random labels; the evaluation metric is **not** a scientific or real-world model benchmark.

## Dependencies

### Core
- `torch>=2.0.0` - PyTorch for ML
- `fastapi>=0.100.0` - Web framework
- `uvicorn>=0.23.0` - ASGI server
- `click>=8.1.0` - CLI framework
- `rich>=13.0.0` - Terminal UI
- `pydantic>=2.0.0` - Data validation
- `plotly>=5.0.0` - Visualization
- `ray>=2.0.0` - Distributed training

### Advanced Research
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `typer>=0.9.0` - Modern CLI
- `aiohttp>=3.8.0` - Async HTTP
- `jax>=0.4.0` - ML framework (optional)

### ACCA Framework
- `torch>=2.0.0` - Deep learning
- `networkx>=3.0` - Graph algorithms

### Visualization & Interpretability
- `plotly>=5.0.0` - Interactive plots
- `ipywidgets>=8.0.0` - Jupyter widgets
- `shap>=0.40.0` - SHAP explanations (optional)
- `lime>=0.2.0` - LIME explanations (optional)

## Security Considerations

⚠️ **Important**: Before deploying to production:

1. **Authentication**: Sensitive endpoints require a 32+-character `AI_HUB_API_KEY`; JWT helper functions are separate from the API route guard.
2. **API Keys**: Use strong API keys for service accounts
3. **Rate Limiting**: Full request quotas and persistent distributed rate limiting remain to be implemented.
4. **Path Sanitization**: API model identifiers are validated; other file-processing modules require independent review.
5. **Secure torch.load()**: Core model loading uses `weights_only=True`; audit all additional model/data loaders.
6. **HTTPS**: TLS termination, restricted network access, and complete reverse-proxy configuration must be verified before public deployment.

## Development

### Running Tests
```bash
# Run all tests (comprehensive suite)
python3 run_all_tests.py

# Run with pytest directly
pytest tests/ Advanced-Research/Ideas/projects/*/tests/ projects/acca/tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific phase tests
pytest tests/test_phase1_comprehensive.py -v
pytest tests/test_phase2_comprehensive.py -v
pytest tests/test_phase3_comprehensive.py -v
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

A root LICENSE file is not currently tracked. Confirm intended license and add its full text before distribution.

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{ai_hub,
  title = {AI Hub: Advanced ML Research Platform},
  author = {Advanced Research Team},
  year = {2024},
  url = {https://github.com/username/ai-hub}
}
```

## Support

- 📖 Documentation: See individual project READMEs in `Advanced-Research/Ideas/`
- 🐛 Issues: Open a GitHub issue
- 💬 Discussions: Join our research community

---

**Built with ❤️ by the Advanced Research Team**
