#!/bin/bash
# AI Hub Platform - Master Command Reference
# Save this as ai-hub-commands.sh and source it: source ai-hub-commands.sh

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           AI Hub Platform - Command Reference                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# DEVELOPMENT COMMANDS
# ============================================================================
echo "📦 DEVELOPMENT"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Install and Setup:"
echo "  pip install -e .                                    # Install AI Hub"
echo "  pip install -e \".[dev]\"                            # With dev dependencies"
echo ""
echo "Development Server:"
echo "  ai-hub api start                                    # Start API server"
echo "  ai-hub api start --port 8000 --reload              # With auto-reload"
echo "  ai-hub api start --host 0.0.0.0 --port 8080        # Custom host/port"
echo ""
echo "Docker Development:"
echo "  docker-compose up -d                                # Start all services"
echo "  docker-compose --profile dev up -d                 # With Jupyter Lab"
echo "  docker-compose --profile monitoring up -d          # With monitoring"
echo "  docker-compose logs -f ai-hub-api                  # View API logs"
echo "  docker-compose down                                 # Stop all services"
echo ""

# ============================================================================
# MODEL TRAINING COMMANDS
# ============================================================================
echo "🧠 MODEL TRAINING"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Basic Training:"
echo "  ai-hub model train --epochs 100                    # Train with defaults"
echo "  ai-hub model train --input-dim 784 --hidden-dim 256 --output-dim 10"
echo "  ai-hub local train --epochs 50 --save-to model.pt  # Local training"
echo ""
echo "Async Training:"
echo "  ai-hub async train --epochs 100                    # Async with WebSocket"
echo "  # Connect to: ws://localhost:8000/ws/train/{job_id}"
echo ""
echo "Distributed Training:"
echo "  ai-hub distributed train --workers 4               # Distributed on Ray"
echo "  ai-hub distributed train --workers 8 --gpu         # With GPU"
echo ""
echo "AutoML Optimization:"
echo "  ai-hub automl optimize                             # Default Bayesian"
echo "  ai-hub automl optimize --strategy random --iterations 50"
echo "  ai-hub automl optimize --strategy grid"
echo "  ai-hub automl optimize --strategy pbt --metric accuracy --mode max"
echo ""

# ============================================================================
# MODEL SERVING COMMANDS
# ============================================================================
echo "🌐 MODEL SERVING"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Start Serving:"
echo "  ai-hub serve start --port 8080                     # Start server"
echo "  ai-hub serve start --model my_model --version v1.0"
echo ""
echo "Monitor Serving:"
echo "  ai-hub serve metrics                               # View metrics"
echo ""
echo "Predictions:"
echo "  curl -X POST http://localhost:8080/serve/predict \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"data\": [0.1, 0.2, 0.3], \"model_name\": \"my_model\"}'"
echo ""

# ============================================================================
# MODEL REGISTRY COMMANDS
# ============================================================================
echo "📦 MODEL REGISTRY"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "List Models:"
echo "  ai-hub registry list                               # List all models"
echo "  ai-hub registry versions my_model                  # List versions"
echo ""
echo "A/B Testing:"
echo "  # Setup via Python API:"
echo "  registry.setup_ab_test('model', 'v1.0', 'v2.0', 0.2)"
echo ""

# ============================================================================
# EXPERIMENT TRACKING COMMANDS
# ============================================================================
echo "📊 EXPERIMENT TRACKING"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "List Experiments:"
echo "  ai-hub experiment list                             # List all runs"
echo ""
echo "Python API:"
echo "  from ai_hub.experiment_tracking import ExperimentTracker"
echo "  tracker = ExperimentTracker('my_exp', use_mlflow=True)"
echo "  tracker.start_run()"
echo "  tracker.log_param('lr', 0.001)"
echo "  tracker.log_metric('accuracy', 0.95)"
echo "  tracker.end_run()"
echo ""

# ============================================================================
# RESEARCH INTEGRATION COMMANDS
# ============================================================================
echo "🔬 RESEARCH INTEGRATION"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Compose Architecture:"
echo "  ai-hub integration compose                         # Compose ADAG→ANAS→ACML"
echo "  ai-hub integration compose --input-dim 128 --output-dim 10"
echo ""
echo "Advanced Research CLI:"
echo "  cd Advanced-Research"
echo "  advanced-research init --config config.yaml        # Initialize project"
echo "  advanced-research validate --package my_package    # Validate package"
echo "  advanced-research test --all                       # Test all packages"
echo "  advanced-research list                            # List all packages"
echo ""

# ============================================================================
# PACKAGE DEVELOPMENT COMMANDS
# ============================================================================
echo "📦 PACKAGE DEVELOPMENT"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Create New Package:"
echo "  ai-hub create package --name 'quantum_ml'          # Create package"
echo "  ai-hub create package -n 'my_pkg' -t 'meta-learning'"
echo "  ai-hub create package -n 'opt_pkg' -t 'optimization'"
echo ""
echo "Package Types:"
echo "  - general (default)"
echo "  - meta-learning"
echo "  - optimization"
echo "  - architecture"
echo "  - field-theory"
echo ""
echo "After Creating:"
echo "  cd Advanced-Research/Ideas/projects/my_pkg"
echo "  pip install -e ."
echo "  pytest tests/ -v"
echo ""

# ============================================================================
# VISUALIZATION COMMANDS
# ============================================================================
echo "📈 VISUALIZATION"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Launch Dashboard:"
echo "  ai-hub dashboard                                    # Interactive dashboard"
echo ""
echo "Python API:"
echo "  from ai_hub.visualization import DashboardManager"
echo "  dashboard = DashboardManager()"
echo "  plot = dashboard.metrics_viz.create_training_dashboard('run_123')"
echo "  DashboardManager.export_plot(plot, 'training.html')"
echo ""

# ============================================================================
# TESTING COMMANDS
# ============================================================================
echo "🧪 TESTING"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Run Tests:"
echo "  pytest tests/ -v                                   # All tests"
echo "  pytest tests/test_phase1_comprehensive.py -v      # Phase 1 tests"
echo "  pytest tests/test_phase2_comprehensive.py -v      # Phase 2 tests"
echo "  pytest tests/test_api.py -v                       # API tests"
echo ""
echo "With Coverage:"
echo "  pytest tests/ --cov=ai_hub --cov-report=html"
echo "  pytest tests/ --cov=ai_hub --cov-report=xml"
echo ""
echo "Research Package Tests:"
echo "  cd Advanced-Research/Ideas/projects/acml && pytest tests/ -v"
echo "  cd Advanced-Research/Ideas/projects/adag && pytest tests/ -v"
echo "  cd Advanced-Research/Ideas/projects/anas && pytest tests/ -v"
echo ""
echo "Docker Testing:"
echo "  docker-compose --profile testing up                # Run in container"
echo ""

# ============================================================================
# CONFIGURATION COMMANDS
# ============================================================================
echo "⚙️  CONFIGURATION"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "View Config:"
echo "  ai-hub config show                                 # Show all config"
echo "  ai-hub config get api.port                         # Get specific value"
echo ""
echo "Set Config:"
echo "  ai-hub config set api.port 8080                    # Set value"
echo "  ai-hub config set training.batch_size 64"
echo ""
echo "System Status:"
echo "  ai-hub status                                      # Check system status"
echo "  ai-hub api status                                  # Check API status"
echo ""

# ============================================================================
# MONITORING & DEBUGGING
# ============================================================================
echo "🔍 MONITORING & DEBUGGING"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Docker Monitoring:"
echo "  docker-compose ps                                  # List containers"
echo "  docker-compose logs -f ai-hub-api                  # API logs"
echo "  docker-compose logs -f model-server                # Serving logs"
echo "  docker-compose top                                 # Process list"
echo ""
echo "Monitoring Dashboards:"
echo "  http://localhost:3000                              # Grafana"
echo "  http://localhost:9090                              # Prometheus"
echo "  http://localhost:8265                              # Ray Dashboard"
echo "  http://localhost:5000                              # MLflow UI"
echo ""
echo "API Documentation:"
echo "  http://localhost:8000/docs                         # Swagger UI"
echo "  http://localhost:8000/redoc                        # ReDoc"
echo ""

# ============================================================================
# UTILITY COMMANDS
# ============================================================================
echo "🛠️  UTILITY"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Paper Generation:"
echo "  # Python API:"
echo "  from ai_hub.paper_generator import ExperimentToPaper"
echo "  paper_gen = ExperimentToPaper(tracker)"
echo "  paper = paper_gen.generate_from_experiment('my_exp')"
echo "  paper.save('paper.tex')"
echo ""
echo "Dataset Versioning:"
echo "  # Python API:"
echo "  from ai_hub.dataset_versioning import DatasetRepository"
echo "  repo = DatasetRepository()"
echo "  version = repo.commit(data, 'my_dataset')"
echo "  old_data = repo.checkout('v1_20240216_120000')"
echo ""
echo "Security:"
echo "  from ai_hub.security import authenticate_user, create_access_token"
echo "  user = authenticate_user(db, 'admin', 'password')"
echo "  token = create_access_token({'sub': user.username})"
echo ""

# ============================================================================
# EXAMPLES
# ============================================================================
echo "💡 EXAMPLES"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Run Examples:"
echo "  python examples/phase1_complete_examples.py        # Phase 1 examples"
echo "  python examples/phase2_advanced_examples.py        # Phase 2 examples"
echo ""

# ============================================================================
# COMPLETE WORKFLOW EXAMPLE
# ============================================================================
echo "🔄 COMPLETE WORKFLOW EXAMPLE"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "1. Start platform:"
echo "   docker-compose up -d"
echo ""
echo "2. Create research package:"
echo "   ai-hub create package --name 'my_experiment' --type 'meta-learning'"
echo ""
echo "3. Train with AutoML:"
echo "   ai-hub automl optimize --strategy bayesian --iterations 100"
echo ""
echo "4. Register best model:"
echo "   # (Automatically done via Python API)"
echo ""
echo "5. Start serving:"
echo "   ai-hub serve start --model my_model"
echo ""
echo "6. Monitor:"
echo "   ai-hub serve metrics"
echo "   # Or visit http://localhost:3000 (Grafana)"
echo ""

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  For detailed documentation: docs/ or http://localhost:8000/docs"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
