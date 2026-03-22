"""
Comprehensive Phase 1 Test Suite

Tests all Phase 1 core modules:
- Research Integration Hub
- Model Registry
- Experiment Tracking
- Security
- Full integration between components
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Import Phase 1 modules
from ai_hub.integration import ResearchIntegration, IntegrationConfig, UnifiedExperiment
from ai_hub.model_registry import ModelRegistry, ModelVersion, ModelCard, get_registry
from ai_hub.experiment_tracking import ExperimentTracker, ExperimentRun, AutoTracker
from ai_hub.security import (
    authenticate_user,
    create_access_token,
    verify_token,
    generate_api_key,
    check_rate_limit,
    Token,
    User,
)


class TestResearchIntegrationHub:
    """Test Research Integration Hub."""

    def test_integration_initialization(self):
        """Test hub initialization."""
        config = IntegrationConfig(
            adag_enabled=True, anas_enabled=True, acml_enabled=True
        )
        hub = ResearchIntegration(config)

        assert hub.config == config
        assert hub.config.output_dir == "outputs/integration"

    def test_compose_architecture_pipeline(self):
        """Test full architecture composition pipeline."""
        hub = ResearchIntegration()

        architecture = hub.compose_architecture_pipeline(
            input_dim=128, output_dim=10, task_description="Test classification"
        )

        assert architecture is not None
        assert isinstance(architecture, nn.Module)

        # Test forward pass
        x = torch.randn(4, 128)
        output = architecture(x)
        assert output.shape == (4, 10)

    def test_integration_report(self):
        """Test integration report generation."""
        hub = ResearchIntegration()

        # Compose something first to populate history
        hub.compose_architecture_pipeline(64, 5)

        report = hub.get_integration_report()

        assert "available_components" in report
        assert "config" in report
        assert report["config"]["adag_enabled"] == True

    def test_cross_framework_pipeline(self):
        """Test creating cross-framework pipelines."""
        hub = ResearchIntegration()

        pipeline = hub.create_cross_framework_pipeline(
            source_framework="adag",
            target_framework="anas",
            transformation_type="functorial",
        )

        assert callable(pipeline)

        # Test pipeline execution
        result = pipeline({"test": "data"})
        assert result is not None


class TestModelRegistry:
    """Test Model Registry with versioning and A/B testing."""

    @pytest.fixture
    def temp_registry(self):
        """Create temporary registry for testing."""
        temp_dir = tempfile.mkdtemp()
        registry = ModelRegistry(registry_dir=temp_dir)
        yield registry
        shutil.rmtree(temp_dir)

    def test_register_model(self, temp_registry):
        """Test model registration."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        version = temp_registry.register_model(
            model=model,
            model_name="test_model",
            metrics={"accuracy": 0.95, "loss": 0.05},
            tags=["v1", "production"],
        )

        assert version.version_id == "1.0.0"
        assert version.model_name == "test_model"
        assert version.metrics["accuracy"] == 0.95

    def test_semantic_versioning(self, temp_registry):
        """Test automatic semantic versioning."""
        model = nn.Linear(10, 5)

        # First version
        v1 = temp_registry.register_model(model, "versioned_model")
        assert v1.version_id == "1.0.0"

        # Second version
        v2 = temp_registry.register_model(model, "versioned_model")
        assert v2.version_id == "1.0.1"

    def test_load_model(self, temp_registry):
        """Test model loading."""
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))

        temp_registry.register_model(model, "loadable_model", version="v1.0")

        # Load without class
        state_dict = temp_registry.load_model("loadable_model", "v1.0")
        assert isinstance(state_dict, dict)

    def test_ab_test_creation(self, temp_registry):
        """Test A/B test setup."""
        model = nn.Linear(10, 5)

        # Register two versions
        temp_registry.register_model(model, "ab_model", version="v1.0")
        temp_registry.register_model(model, "ab_model", version="v2.0")

        # Create A/B test
        test_id = temp_registry.create_ab_test(
            test_name="model_comparison",
            model_a=("ab_model", "v1.0"),
            model_b=("ab_model", "v2.0"),
            traffic_split=0.3,
        )

        assert test_id.startswith("ab_model_comparison_")

        # Verify config
        config = temp_registry.get_ab_test_config(test_id)
        assert config["traffic_split"] == 0.3
        assert config["model_a"]["version"] == "v1.0"

    def test_ab_test_routing(self, temp_registry):
        """Test A/B test request routing."""
        model = nn.Linear(10, 5)

        temp_registry.register_model(model, "route_model", version="v1.0")
        temp_registry.register_model(model, "route_model", version="v2.0")

        test_id = temp_registry.create_ab_test(
            "routing_test",
            ("route_model", "v1.0"),
            ("route_model", "v2.0"),
            traffic_split=0.5,
        )

        # Test routing (should be deterministic for same user)
        route1 = temp_registry.route_request(test_id, "user_123")
        route2 = temp_registry.route_request(test_id, "user_123")
        assert route1 == route2

    def test_model_card_creation(self, temp_registry):
        """Test automatic model card creation."""
        model = nn.Sequential(nn.Linear(100, 50), nn.Linear(50, 10))

        temp_registry.register_model(
            model, "card_model", version="v1.0", description="Test model with card"
        )

        card = temp_registry.get_model_card("card_model", "v1.0")

        assert card is not None
        assert card.model_name == "card_model"
        assert card.version == "v1.0"
        assert card.architecture == "Sequential"
        assert card.num_parameters > 0

    def test_model_lineage(self, temp_registry):
        """Test model lineage tracking."""
        model = nn.Linear(10, 5)

        # Create lineage
        v1 = temp_registry.register_model(model, "lineage_model", version="v1.0")
        v2 = temp_registry.register_model(
            model, "lineage_model", version="v2.0", parent_version="v1.0"
        )

        lineage = temp_registry.get_lineage("lineage_model", "v2.0")

        assert lineage["version"] == "v2.0"
        assert lineage["parent"]["version_id"] == "v1.0"


class TestExperimentTracking:
    """Test Experiment Tracking with MLflow/W&B."""

    @pytest.fixture
    def temp_tracker(self):
        """Create temporary tracker."""
        temp_dir = tempfile.mkdtemp()
        tracker = ExperimentTracker(
            experiment_name="test_experiment",
            local_dir=temp_dir,
            use_mlflow=False,  # Don't require MLflow for tests
            use_wandb=False,  # Don't require W&B for tests
        )
        yield tracker
        shutil.rmtree(temp_dir)

    def test_start_run(self, temp_tracker):
        """Test starting experiment run."""
        run = temp_tracker.start_run(run_name="test_run")

        assert run is not None
        assert run.experiment_name == "test_experiment"
        assert run.status == "running"
        assert run.run_id is not None

    def test_log_params(self, temp_tracker):
        """Test parameter logging."""
        temp_tracker.start_run()

        temp_tracker.log_param("learning_rate", 0.001)
        temp_tracker.log_param("batch_size", 32)

        assert temp_tracker.active_run.params["learning_rate"] == 0.001
        assert temp_tracker.active_run.params["batch_size"] == 32

    def test_log_metrics(self, temp_tracker):
        """Test metric logging."""
        temp_tracker.start_run()

        temp_tracker.log_metric("accuracy", 0.95, step=0)
        temp_tracker.log_metric("accuracy", 0.96, step=1)
        temp_tracker.log_metric("loss", 0.1, step=0)

        assert len(temp_tracker.active_run.metrics["accuracy"]) == 2
        assert temp_tracker.active_run.metrics["accuracy"][0][1] == 0.95

    def test_end_run(self, temp_tracker):
        """Test ending experiment run."""
        temp_tracker.start_run()
        temp_tracker.log_metric("accuracy", 0.95)

        temp_tracker.end_run()

        assert temp_tracker.active_run is None
        assert len(temp_tracker.runs) == 1
        assert temp_tracker.runs[0].status == "completed"

    def test_compare_runs(self, temp_tracker):
        """Test run comparison."""
        # Run 1
        temp_tracker.start_run()
        temp_tracker.log_metric("accuracy", 0.95)
        temp_tracker.end_run()

        # Run 2
        temp_tracker.start_run()
        temp_tracker.log_metric("accuracy", 0.97)
        temp_tracker.end_run()

        run_ids = [r.run_id for r in temp_tracker.runs]
        comparison = temp_tracker.compare_runs(run_ids)

        assert "runs" in comparison
        assert "metric_comparison" in comparison

    def test_get_best_run(self, temp_tracker):
        """Test finding best run."""
        # Run 1 - worse
        temp_tracker.start_run()
        temp_tracker.log_metric("accuracy", 0.95)
        temp_tracker.end_run()

        # Run 2 - better
        temp_tracker.start_run()
        temp_tracker.log_metric("accuracy", 0.98)
        temp_tracker.end_run()

        best = temp_tracker.get_best_run("accuracy", mode="max")

        assert best is not None
        assert best.metrics["accuracy"][-1] == 0.98


class TestSecurity:
    """Test Security with OAuth2/JWT."""

    def test_password_verification(self):
        """Test password hashing and verification."""
        from ai_hub.security import verify_password

        # Test with known hash
        password = "test123"
        hashed = "9bc34549d565d2c9b66fb5c7c2e6a3b6"  # MD5 of test123

        result = verify_password(password, hashed)
        assert result == True

        result = verify_password("wrong", hashed)
        assert result == False

    def test_authenticate_user(self):
        """Test user authentication."""
        from ai_hub.security import authenticate_user, fake_users_db

        # Valid credentials
        user = authenticate_user(fake_users_db, "admin", "admin123")
        assert user is not None
        assert user.username == "admin"
        assert user.role == "admin"

        # Invalid credentials
        user = authenticate_user(fake_users_db, "admin", "wrong")
        assert user is None

        # Non-existent user
        user = authenticate_user(fake_users_db, "nobody", "pass")
        assert user is None

    def test_token_creation_and_verification(self):
        """Test JWT token lifecycle."""
        # Create token
        token = create_access_token(
            data={"sub": "testuser", "scopes": ["read", "write"]}
        )

        assert token is not None
        assert isinstance(token, str)

        # Verify token
        token_data = verify_token(token)

        assert token_data.username == "testuser"
        assert "read" in token_data.scopes

    def test_api_key_generation(self):
        """Test API key generation."""
        from ai_hub.security import generate_api_key, api_keys_db

        api_key = generate_api_key("testuser", "my_key")

        assert api_key.startswith("ak_")
        assert api_key in api_keys_db
        assert api_keys_db[api_key]["username"] == "testuser"
        assert api_keys_db[api_key]["active"] == True

    def test_rate_limiting(self):
        """Test rate limit checking."""
        identifier = "test_client"

        # First requests should be allowed
        for i in range(5):
            allowed = check_rate_limit(identifier, max_requests=10)
            assert allowed == True

        # Should still be allowed (under limit)
        allowed = check_rate_limit(identifier, max_requests=10)
        assert allowed == True

    def test_user_roles_and_scopes(self):
        """Test user roles and permission scopes."""
        from ai_hub.security import get_user, fake_users_db

        # Admin user
        admin = get_user(fake_users_db, "admin")
        assert admin.role == "admin"
        assert "admin" in admin.scopes
        assert "delete" in admin.scopes

        # Researcher user
        researcher = get_user(fake_users_db, "researcher")
        assert researcher.role == "researcher"
        assert "train" in researcher.scopes
        assert "admin" not in researcher.scopes

        # Regular user
        user = get_user(fake_users_db, "user")
        assert user.role == "user"
        assert "read" in user.scopes
        assert "write" not in user.scopes


class TestIntegration:
    """Integration tests between Phase 1 components."""

    def test_end_to_end_workflow(self):
        """Test complete workflow: train → track → register → serve."""
        # This tests the integration between all Phase 1 components

        # 1. Start experiment tracking
        tracker = ExperimentTracker(
            "integration_test", use_mlflow=False, use_wandb=False
        )
        tracker.start_run()

        # 2. Create model
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        # 3. Log parameters
        tracker.log_param("input_dim", 10)
        tracker.log_param("hidden_dim", 20)

        # 4. Simulate training
        for epoch in range(3):
            loss = 1.0 / (epoch + 1)
            tracker.log_metric("loss", loss, step=epoch)

        # 5. End experiment
        tracker.end_run()

        # 6. Register model
        temp_dir = tempfile.mkdtemp()
        try:
            registry = ModelRegistry(registry_dir=temp_dir)
            version = registry.register_model(
                model=model, model_name="integrated_model", metrics={"final_loss": 0.33}
            )

            assert version.version_id == "1.0.0"

            # 7. Verify model can be loaded
            loaded = registry.load_model("integrated_model", "1.0.0")
            assert loaded is not None

        finally:
            shutil.rmtree(temp_dir)

    def test_integration_with_security(self):
        """Test integration hub with security."""
        from ai_hub.security import authenticate_user, fake_users_db

        # Authenticate researcher
        user = authenticate_user(fake_users_db, "researcher", "research123")
        assert user is not None

        # Researcher should be able to use integration hub
        hub = ResearchIntegration()

        # Compose architecture (requires appropriate permissions in real scenario)
        model = hub.compose_architecture_pipeline(64, 5)
        assert model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
