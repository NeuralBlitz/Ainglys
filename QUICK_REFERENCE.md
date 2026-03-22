# AI Hub - Quick Reference Guide

## 🚀 Quick Commands

### Installation
```bash
# Basic install
pip install -e .

# Development install with all dependencies
pip install -e ".[dev]"

# Install specific research packages
cd Advanced-Research/Ideas/projects/acml && pip install -e .
cd ../aether_calc && pip install -e .
cd ../gaav && pip install -e .
```

### CLI Commands
```bash
# System status
ai-hub status

# Train a model
ai-hub local train --epochs 100 --save-to my_model.pt

# Make predictions
ai-hub local predict my_model.pt --data 0.1 --data 0.2

# Start API server
ai-hub api start --port 8000

# Model management
ai-hub model list
ai-hub model save my_model
ai-hub model load my_model

# Configuration
ai-hub config show
ai-hub config set api.port 8080
```

### Docker Deployment
```bash
# Start all services
docker-compose up -d

# Scale Ray workers
docker-compose up -d --scale ray-worker=4

# View logs
docker-compose logs -f ai-hub-api

# Stop all
docker-compose down
```

## 📊 Phase-by-Phase Quick Start

### Phase 1: Core Infrastructure
```python
from ai_hub.integration import ResearchIntegration
from ai_hub.model_registry import get_registry
from ai_hub.experiment_tracking import ExperimentTracker

# Compose architecture
hub = ResearchIntegration()
model = hub.compose_architecture_pipeline(784, 10, "MNIST")

# Register model
registry = get_registry()
version = registry.register_model(model, "classifier", metrics={"acc": 0.95})

# Track experiments
tracker = ExperimentTracker("experiment_1", use_mlflow=True)
tracker.start_run()
tracker.log_param("lr", 0.001)
tracker.log_metric("accuracy", 0.95)
tracker.end_run()
```

### Phase 2: Advanced Infrastructure
```python
from ai_hub.distributed import DistributedTrainer
from ai_hub.automl import HyperparameterOptimizer
from ai_hub.serving import ModelServer

# Distributed training
trainer = DistributedTrainer(num_workers=4, use_gpu=True)
results = trainer.train_distributed(model_fn, dataset_fn)

# AutoML optimization
optimizer = HyperparameterOptimizer(strategy="bayesian")
best = optimizer.optimize(model_fn, train_fn, search_space)

# Model serving
server = ModelServer(model_path="model.pt")
server.deploy(strategy="canary", traffic_percentage=10)
```

### Phase 3: Visualization & Analysis
```python
from ai_hub.advanced_viz import ModelComparator, TrainingAnimator
from ai_hub.interpretability import InterpretabilityDashboard
from ai_hub.training_widgets import TrainingDashboard

# Compare models
comparator = ModelComparator()
comparator.add_model("A", {"acc": 0.95})
comparator.add_model("B", {"acc": 0.93})
comparator.create_radar_chart().write_html("comparison.html")

# Model interpretability
dashboard = InterpretabilityDashboard(model)
saliency = dashboard.compute_saliency(input_image)
dashboard.create_interactive_dashboard(input, X_train, y_train)

# Jupyter widgets (in notebook)
dashboard = TrainingDashboard()
dashboard.display()
for epoch in range(100):
    metrics = trainer.train_epoch()
    dashboard.update_metrics(metrics, epoch)
```

## 🧪 Testing

### Run All Tests
```bash
# All tests
pytest tests/ -v

# Specific phases
pytest tests/test_phase1_comprehensive.py -v
pytest tests/test_phase2_comprehensive.py -v
pytest tests/test_phase3_comprehensive.py -v

# Integration tests
pytest tests/test_integration_all_phases.py -v

# With coverage
pytest tests/ --cov=ai_hub --cov-report=html
```

### Verify Deployment
```bash
# Quick verification
python verify_deployment.py

# Full verification with tests
python verify_deployment.py --full

# Verify specific phase
python verify_deployment.py --phase 3
```

## 📚 Examples

### Run Examples
```bash
# Phase 1
python examples/phase1_complete_examples.py

# Phase 2
python examples/phase2_advanced_examples.py

# Phase 3
python examples/phase3_advanced_visualization.py
python examples/phase3_interpretability_examples.py
python examples/phase3_jupyter_widgets_examples.py

# Complete workflow
python examples/master_complete_example.py
```

## 🔧 Configuration

### Environment Variables
```bash
# Model Registry
MODEL_REGISTRY_DIR=models/registry

# Experiment Tracking
MLFLOW_TRACKING_URI=http://localhost:5000
WANDB_PROJECT=my_project

# Security
JWT_SECRET_KEY=your-secret-key
API_RATE_LIMIT=100/hour

# Ray
RAY_ADDRESS=auto
RAY_NUM_CPUS=4
RAY_NUM_GPUS=1

# Serving
SERVING_BATCH_SIZE=32
SERVING_MAX_LATENCY_MS=100
```

### Config File (~/.ai_hub/config.json)
```json
{
  "api": {
    "host": "0.0.0.0",
    "port": 8000
  },
  "model": {
    "default_epochs": 10,
    "models_dir": "models"
  },
  "logging": {
    "level": "INFO"
  }
}
```

## 🌐 API Endpoints

### Basic
- `GET /` - API info
- `GET /status` - System status
- `POST /train` - Train model
- `POST /predict` - Make prediction
- `GET /models` - List models

### Phase 1
- `POST /integration/compose` - Compose architecture
- `POST /registry/models` - Register model
- `POST /experiments/start` - Start experiment
- `POST /auth/token` - Get access token

### Phase 2
- `POST /async/train` - Async training
- `WS /ws/train/{id}` - Training updates
- `POST /distributed/train` - Distributed training
- `POST /automl/optimize` - AutoML
- `POST /serve/deploy` - Deploy model

### Phase 3
- `POST /viz/dashboard/create` - Create dashboard
- `POST /interpret/saliency` - Compute saliency
- `POST /paper/create` - Generate paper
- `POST /dataset/commit` - Commit dataset version

## 🐛 Troubleshooting

### Import Errors
```bash
# If research packages not found
export PYTHONPATH="${PYTHONPATH}:$(pwd)/Advanced-Research/Ideas/projects"

# Or install them
cd Advanced-Research && pip install -e ".[all]"
```

### Docker Issues
```bash
# Reset Docker environment
docker-compose down -v
docker-compose up -d --build

# Check logs
docker-compose logs ai-hub-api
```

### Test Failures
```bash
# Check specific test
pytest tests/test_api.py::test_train_endpoint -v --tb=short

# Run with debugger
pytest tests/test_api.py -v --pdb
```

## 📖 Documentation

### Key Documents
- `README.md` - Main overview
- `COMPLETE_IMPLEMENTATION_SUMMARY.md` - Full platform details
- `docs/PHASE1_README.md` - Phase 1 documentation
- `docs/PHASE2_README.md` - Phase 2 documentation
- `docs/PHASE3_README.md` - Phase 3 documentation
- `docs/ARCHITECTURE.md` - System architecture
- `PROJECT_COMPLETE.md` - Project completion summary

### API Documentation
```bash
# Start API server
ai-hub api start

# View docs at:
# http://localhost:8000/docs (Swagger)
# http://localhost:8000/redoc (ReDoc)
```

## 🎯 Common Workflows

### Research → Production Pipeline
```python
# 1. Research phase - compare architectures
from ai_hub.advanced_viz import ModelComparator
comparator = ModelComparator()
comparator.add_model("Arch_A", metrics_a)
comparator.add_model("Arch_B", metrics_b)

# 2. Select best and register
from ai_hub.model_registry import get_registry
registry = get_registry()
version = registry.register_model(best_model, "production")

# 3. Deploy
from ai_hub.serving import ModelServer
server = ModelServer(model_path)
server.deploy(strategy="canary")

# 4. Monitor
from ai_hub.training_widgets import TrainingDashboard
dashboard = TrainingDashboard()
```

### Collaborative Research
```python
# 1. Version dataset
from ai_hub.dataset_versioning import DatasetVersionManager
manager = DatasetVersionManager("./data")
manager.commit("Initial dataset", "researcher_a@lab.com")

# 2. Train model
from ai_hub.experiment_tracking import ExperimentTracker
tracker = ExperimentTracker("collab_experiment")
tracker.start_run()
# ... train ...
tracker.end_run()

# 3. Share results
from ai_hub.paper_generator import PaperGenerator
paper = PaperGenerator()
paper.add_section("Results", "Achieved 95% accuracy")
paper.generate_latex()
```

## 📞 Support

- **Issues**: Check `verify_deployment.py` output
- **Tests**: Run `pytest tests/ -v`
- **Examples**: See `examples/` directory
- **Docs**: Read `COMPLETE_IMPLEMENTATION_SUMMARY.md`

---

**AI Hub Platform v1.0 - Complete 4-Phase Implementation**
21 Modules | 87 Packages | 170+ Tests | 48 Features
