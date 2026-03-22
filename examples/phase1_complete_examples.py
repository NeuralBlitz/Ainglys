"""
Phase 1 Complete Examples and Documentation

This file provides comprehensive examples for all Phase 1 components:
1. Research Integration Hub
2. Model Registry
3. Experiment Tracking
4. Security
5. Full integration workflows
"""

import torch
import torch.nn as nn
from pathlib import Path

print("=" * 70)
print("PHASE 1: COMPLETE USAGE EXAMPLES")
print("=" * 70)

# ============================================================================
# EXAMPLE 1: Research Integration Hub
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 1: Research Integration Hub")
print("=" * 70)

from ai_hub.integration import ResearchIntegration, IntegrationConfig

# Create integration hub
config = IntegrationConfig(
    adag_enabled=True,
    anas_enabled=True,
    acml_enabled=True,
    tracking_enabled=True,
    output_dir="outputs/integration_example",
)

hub = ResearchIntegration(config)

# Compose architecture through full pipeline
print("\n🚀 Composing architecture through ADAG → ANAS → ACML pipeline...")
architecture = hub.compose_architecture_pipeline(
    input_dim=784,  # MNIST size
    output_dim=10,  # 10 classes
    task_description="MNIST digit classification",
)

print(f"✅ Created architecture: {architecture}")

# Test the architecture
x = torch.randn(4, 784)
y = architecture(x)
print(f"✅ Test forward pass: input {x.shape} → output {y.shape}")

# Get integration report
report = hub.get_integration_report()
print(f"\n📊 Integration Report:")
print(f"   Available components: {len(report['available_components'])}")
for comp in report["available_components"]:
    print(f"     • {comp}")

# ============================================================================
# EXAMPLE 2: Model Registry with Versioning
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 2: Model Registry with Versioning")
print("=" * 70)

from ai_hub.model_registry import ModelRegistry, get_registry

# Create registry
registry = ModelRegistry(registry_dir="outputs/registry_example")

# Create a simple model
model = nn.Sequential(
    nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 10)
)

# Register first version
print("\n📝 Registering model v1.0.0...")
version1 = registry.register_model(
    model=model,
    model_name="mnist_classifier",
    metrics={"accuracy": 0.92, "f1_score": 0.91, "training_time": 120},
    tags=["baseline", "production"],
    description="Initial MNIST classifier model",
)

print(f"✅ Registered version: {version1.version_id}")
print(f"   Checksum: {version1.checksum}")
print(f"   Created: {version1.created_at}")

# Register second version (improved)
print("\n📝 Registering model v1.0.1 (improved)...")
version2 = registry.register_model(
    model=model,
    model_name="mnist_classifier",
    metrics={"accuracy": 0.95, "f1_score": 0.94, "training_time": 135},
    tags=["improved"],
    description="Improved MNIST classifier with better accuracy",
    parent_version="1.0.0",
)

print(f"✅ Registered version: {version2.version_id}")

# List all versions
versions = registry.list_versions("mnist_classifier")
print(f"\n📋 All versions ({len(versions)}):")
for v in versions:
    print(f"   • {v.version_id}: acc={v.metrics.get('accuracy', 'N/A')}")

# Get model card
card = registry.get_model_card("mnist_classifier", "1.0.0")
print(f"\n📄 Model Card:")
print(f"   Name: {card.model_name}")
print(f"   Parameters: {card.num_parameters:,}")
print(f"   Size: {card.model_size_mb:.2f} MB")

# ============================================================================
# EXAMPLE 3: A/B Testing
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 3: A/B Testing")
print("=" * 70)

# Create A/B test
print("\n🧪 Setting up A/B test...")
test_id = registry.create_ab_test(
    test_name="mnist_model_comparison",
    model_a=("mnist_classifier", "1.0.0"),
    model_b=("mnist_classifier", "1.0.1"),
    traffic_split=0.3,  # 30% to new model
)

print(f"✅ A/B test created: {test_id}")

# Simulate routing requests
print("\n🎯 Simulating request routing:")
for i in range(5):
    user_id = f"user_{i}"
    model_name, version = registry.route_request(test_id, user_id)
    print(f"   User {user_id} → {model_name} v{version}")

# ============================================================================
# EXAMPLE 4: Experiment Tracking
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Experiment Tracking")
print("=" * 70)

from ai_hub.experiment_tracking import ExperimentTracker

# Create tracker (without external dependencies for demo)
tracker = ExperimentTracker(
    experiment_name="mnist_training",
    tracking_uri=None,
    use_mlflow=False,
    use_wandb=False,
    local_dir="outputs/experiments_example",
)

# Start experiment
print("\n🔬 Starting experiment...")
run = tracker.start_run(run_name="baseline_run", tags=["mnist", "baseline"])
print(f"✅ Run started: {run.run_id}")

# Log hyperparameters
tracker.log_params(
    {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "optimizer": "Adam",
        "model_architecture": "SimpleMLP",
    }
)

# Log hyperparameters organized
tracker.log_hyperparameters(
    {"lr": 0.001, "batch_size": 32, "hidden_dim": 256, "dropout": 0.2}
)

# Simulate training and log metrics
print("\n📈 Simulating training...")
for epoch in range(10):
    # Simulate improving metrics
    train_loss = 1.0 / (epoch + 1)
    val_loss = 1.2 / (epoch + 1)
    accuracy = 0.5 + (epoch * 0.04)

    tracker.log_metric("train_loss", train_loss, step=epoch)
    tracker.log_metric("val_loss", val_loss, step=epoch)
    tracker.log_metric("accuracy", accuracy, step=epoch)

    if epoch % 3 == 0:
        print(f"   Epoch {epoch}: loss={train_loss:.4f}, acc={accuracy:.2%}")

# End experiment
tracker.end_run()
print(f"✅ Experiment completed")

# Get run history
runs = tracker.get_run_history()
print(f"\n📊 Total runs: {len(runs)}")

# Compare runs (if multiple)
if len(runs) >= 1:
    run_ids = [r.run_id for r in runs]
    comparison = tracker.compare_runs(run_ids)
    print(f"\n📊 Comparison available for {len(comparison['runs'])} runs")

# Find best run
best = tracker.get_best_run("accuracy", mode="max")
if best:
    print(f"\n🏆 Best run: {best.run_id[:8]}...")
    print(f"   Best accuracy: {best.metrics['accuracy'][-1]:.2%}")

# ============================================================================
# EXAMPLE 5: Security & Authentication
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 5: Security & Authentication")
print("=" * 70)

from ai_hub.security import (
    authenticate_user,
    create_access_token,
    verify_token,
    generate_api_key,
    check_rate_limit,
    fake_users_db,
)

# User authentication
print("\n🔐 Testing authentication...")

# Valid login
user = authenticate_user(fake_users_db, "admin", "admin123")
if user:
    print(f"✅ Admin authenticated: {user.username}")
    print(f"   Role: {user.role}")
    print(f"   Scopes: {', '.join(user.scopes)}")

# Invalid login
user = authenticate_user(fake_users_db, "admin", "wrong_password")
if not user:
    print(f"❌ Invalid credentials rejected")

# JWT Token creation
print("\n🔑 Creating JWT token...")
token = create_access_token(
    data={"sub": "researcher", "scopes": ["read", "write", "train"]}, expires_delta=None
)
print(f"✅ Token created: {token[:50]}...")

# Token verification
token_data = verify_token(token)
print(f"✅ Token verified for user: {token_data.username}")
print(f"   Scopes: {', '.join(token_data.scopes)}")

# API Key generation
print("\n🔑 Generating API key...")
api_key = generate_api_key("researcher", "production_key")
print(f"✅ API key generated: {api_key[:20]}...")

# Rate limiting
print("\n⏱️  Testing rate limiting...")
client_id = "test_client"
for i in range(12):
    allowed = check_rate_limit(client_id, max_requests=10, window_seconds=60)
    status = "✅ Allowed" if allowed else "❌ Blocked"
    if i >= 9:  # Only show last few
        print(f"   Request {i + 1}: {status}")

# ============================================================================
# EXAMPLE 6: Complete Integration Workflow
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 6: Complete Integration Workflow")
print("=" * 70)

print("\n🔄 Demonstrating complete workflow:")
print("   1. Create model using Integration Hub")
print("   2. Track training with Experiment Tracker")
print("   3. Register best model in Registry")
print("   4. Set up A/B testing")

# Step 1: Create model
hub = ResearchIntegration()
model = hub.compose_architecture_pipeline(784, 10, "MNIST classifier")
print(f"\n   ✅ Step 1: Model created")

# Step 2: Track training
tracker = ExperimentTracker("integration_workflow", use_mlflow=False, use_wandb=False)
tracker.start_run()
tracker.log_param("model_type", "composed")
for epoch in range(5):
    tracker.log_metric("accuracy", 0.8 + epoch * 0.03, step=epoch)
tracker.end_run()
print(f"   ✅ Step 2: Training tracked")

# Step 3: Register model
workflow_registry = ModelRegistry(registry_dir="outputs/workflow_registry")
version = workflow_registry.register_model(
    model=model, model_name="workflow_model", metrics={"accuracy": 0.95}
)
print(f"   ✅ Step 3: Model registered as v{version.version_id}")

# Step 4: A/B test setup
workflow_registry.register_model(model, "workflow_model", version="v2.0")
test_id = workflow_registry.create_ab_test(
    "workflow_ab",
    ("workflow_model", version.version_id),
    ("workflow_model", "v2.0"),
    0.5,
)
print(f"   ✅ Step 4: A/B test configured")

print("\n" + "=" * 70)
print("✅ ALL PHASE 1 EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nKey Features Demonstrated:")
print("  • Research Integration Hub (ADAG+ANAS+ACML)")
print("  • Model Registry with Semantic Versioning")
print("  • A/B Testing and Traffic Splitting")
print("  • Experiment Tracking (MLflow/W&B compatible)")
print("  • Security (OAuth2/JWT/API Keys)")
print("  • Rate Limiting")
print("  • Complete End-to-End Workflows")
print("\nAll outputs saved to: outputs/")
print("=" * 70)
