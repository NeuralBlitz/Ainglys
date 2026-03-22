"""
Enhanced Model Registry with Versioning, A/B Testing, and Model Cards
"""

import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn


@dataclass
class ModelVersion:
    """Represents a specific model version."""

    version_id: str
    model_name: str
    created_at: str
    checksum: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    tags: List[str]
    parent_version: Optional[str] = None
    description: str = ""


@dataclass
class ModelCard:
    """
    Model Card for documentation and transparency.

    Based on Google's Model Cards paper.
    """

    model_name: str
    version: str
    description: str
    author: str
    created_date: str

    # Model Details
    architecture: str
    num_parameters: int
    model_size_mb: float

    # Intended Use
    intended_use: List[str]
    factors: List[str]

    # Metrics
    performance_metrics: Dict[str, float]
    evaluation_data: str
    training_data: str

    # Ethics & Safety
    ethical_considerations: str
    caveats: str

    # Lineage
    parent_models: List[str]
    derived_models: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ModelRegistry:
    """
    Enhanced Model Registry with versioning, A/B testing, and lineage tracking.

    Features:
    - Semantic versioning (major.minor.patch)
    - Git LFS integration for model storage
    - A/B test configuration
    - Model lineage tracking
    """

    def __init__(self, registry_dir: str = "models/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.versions_dir = self.registry_dir / "versions"
        self.cards_dir = self.registry_dir / "cards"
        self.ab_tests_dir = self.registry_dir / "ab_tests"

        for d in [self.versions_dir, self.cards_dir, self.ab_tests_dir]:
            d.mkdir(exist_ok=True)

        # In-memory index
        self.index: Dict[str, List[ModelVersion]] = {}
        self._load_index()

    def _load_index(self):
        """Load registry index from disk."""
        index_file = self.registry_dir / "registry_index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                data = json.load(f)
                for model_name, versions in data.items():
                    self.index[model_name] = [ModelVersion(**v) for v in versions]

    def _save_index(self):
        """Save registry index to disk."""
        index_file = self.registry_dir / "registry_index.json"
        data = {}
        for model_name, versions in self.index.items():
            data[model_name] = [
                {k: v for k, v in asdict(ver).items()} for ver in versions
            ]
        with open(index_file, "w") as f:
            json.dump(data, f, indent=2)

    def _compute_checksum(self, filepath: str) -> str:
        """Compute MD5 checksum of model file."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def register_model(
        self,
        model: nn.Module,
        model_name: str,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        parent_version: Optional[str] = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model: PyTorch model to register
            model_name: Name of the model
            version: Semantic version (auto-generated if None)
            metrics: Performance metrics
            parameters: Hyperparameters
            tags: List of tags
            description: Version description
            parent_version: Parent version for lineage

        Returns:
            ModelVersion object
        """
        # Generate version if not provided
        if version is None:
            version = self._generate_version(model_name)

        # Save model
        version_dir = self.versions_dir / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Compute checksum
        checksum = self._compute_checksum(str(model_path))

        # Create version record
        model_version = ModelVersion(
            version_id=version,
            model_name=model_name,
            created_at=datetime.now().isoformat(),
            checksum=checksum,
            metrics=metrics or {},
            parameters=parameters or {},
            tags=tags or [],
            parent_version=parent_version,
            description=description,
        )

        # Update index
        if model_name not in self.index:
            self.index[model_name] = []
        self.index[model_name].append(model_version)

        self._save_index()

        # Create model card
        self._create_model_card(model, model_name, version, metrics)

        return model_version

    def _generate_version(self, model_name: str) -> str:
        """Generate next semantic version."""
        if model_name not in self.index or not self.index[model_name]:
            return "1.0.0"

        # Get latest version and increment patch
        latest = self.index[model_name][-1].version_id
        parts = latest.split(".")
        parts[2] = str(int(parts[2]) + 1)
        return ".".join(parts)

    def _create_model_card(
        self,
        model: nn.Module,
        model_name: str,
        version: str,
        metrics: Optional[Dict[str, float]],
    ):
        """Create model card for the version."""
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (
            1024 * 1024
        )

        card = ModelCard(
            model_name=model_name,
            version=version,
            description=f"Model {model_name} version {version}",
            author="AI Hub",
            created_date=datetime.now().isoformat(),
            architecture=type(model).__name__,
            num_parameters=num_params,
            model_size_mb=model_size,
            intended_use=["research", "experimentation"],
            factors=["architecture_search", "meta_learning"],
            performance_metrics=metrics or {},
            evaluation_data="test_dataset",
            training_data="train_dataset",
            ethical_considerations="Research use only",
            caveats="Experimental model",
            parent_models=[],
            derived_models=[],
        )

        card_path = self.cards_dir / f"{model_name}_{version}_card.json"
        card.save(str(card_path))

    def load_model(
        self, model_name: str, version: str, model_class: Optional[type] = None
    ) -> nn.Module:
        """
        Load a specific model version.

        Args:
            model_name: Name of the model
            version: Version to load
            model_class: Class to instantiate (optional)

        Returns:
            Loaded model
        """
        model_path = self.versions_dir / model_name / version / "model.pt"

        if not model_path.exists():
            raise ValueError(f"Model {model_name} version {version} not found")

        state_dict = torch.load(model_path, map_location="cpu")

        if model_class is not None:
            model = model_class()
            model.load_state_dict(state_dict)
            return model

        # Return state dict if class not provided
        return state_dict

    def get_model_card(self, model_name: str, version: str) -> Optional[ModelCard]:
        """Get model card for a version."""
        card_path = self.cards_dir / f"{model_name}_{version}_card.json"

        if not card_path.exists():
            return None

        with open(card_path, "r") as f:
            data = json.load(f)
            return ModelCard(**data)

    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.index.keys())

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model."""
        return self.index.get(model_name, [])

    def compare_versions(self, model_name: str, version_a: str, version_b: str) -> Dict:
        """Compare two model versions."""
        versions = self.index.get(model_name, [])

        ver_a = next((v for v in versions if v.version_id == version_a), None)
        ver_b = next((v for v in versions if v.version_id == version_b), None)

        if not ver_a or not ver_b:
            raise ValueError("Version not found")

        return {
            "version_a": asdict(ver_a),
            "version_b": asdict(ver_b),
            "metric_diff": {
                k: ver_b.metrics.get(k, 0) - ver_a.metrics.get(k, 0)
                for k in set(ver_a.metrics) | set(ver_b.metrics)
            },
        }

    # A/B Testing Support

    def create_ab_test(
        self,
        test_name: str,
        model_a: Tuple[str, str],  # (model_name, version)
        model_b: Tuple[str, str],
        traffic_split: float = 0.5,
        metrics: Optional[List[str]] = None,
    ) -> str:
        """
        Create A/B test configuration.

        Args:
            test_name: Name of the test
            model_a: Control model (name, version)
            model_b: Treatment model (name, version)
            traffic_split: Percentage of traffic to model_b
            metrics: Metrics to track

        Returns:
            Test ID
        """
        test_config = {
            "test_id": f"ab_{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "name": test_name,
            "model_a": {"name": model_a[0], "version": model_a[1]},
            "model_b": {"name": model_b[0], "version": model_b[1]},
            "traffic_split": traffic_split,
            "metrics": metrics or ["accuracy", "latency"],
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "results": {},
        }

        test_path = self.ab_tests_dir / f"{test_config['test_id']}.json"
        with open(test_path, "w") as f:
            json.dump(test_config, f, indent=2)

        return test_config["test_id"]

    def get_ab_test_config(self, test_id: str) -> Optional[Dict]:
        """Get A/B test configuration."""
        test_path = self.ab_tests_dir / f"{test_id}.json"

        if not test_path.exists():
            return None

        with open(test_path, "r") as f:
            return json.load(f)

    def route_request(self, test_id: str, user_id: str) -> Tuple[str, str]:
        """
        Route request to model A or B based on traffic split.

        Args:
            test_id: A/B test ID
            user_id: User identifier for consistent routing

        Returns:
            (model_name, version)
        """
        config = self.get_ab_test_config(test_id)
        if not config:
            raise ValueError(f"A/B test {test_id} not found")

        # Deterministic routing based on user_id hash
        hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        bucket = hash_val % 100

        if bucket < config["traffic_split"] * 100:
            return (config["model_b"]["name"], config["model_b"]["version"])
        else:
            return (config["model_a"]["name"], config["model_a"]["version"])

    def log_ab_result(self, test_id: str, model: str, metrics: Dict[str, float]):
        """Log result from A/B test."""
        config = self.get_ab_test_config(test_id)
        if not config:
            return

        if model not in config["results"]:
            config["results"][model] = []

        config["results"][model].append(
            {"timestamp": datetime.now().isoformat(), "metrics": metrics}
        )

        test_path = self.ab_tests_dir / f"{test_id}.json"
        with open(test_path, "w") as f:
            json.dump(config, f, indent=2)

    def get_lineage(self, model_name: str, version: str) -> Dict:
        """Get model lineage (parent and child versions)."""
        versions = self.index.get(model_name, [])
        target = next((v for v in versions if v.version_id == version), None)

        if not target:
            return {}

        # Find parent
        parent = None
        if target.parent_version:
            parent = next(
                (v for v in versions if v.version_id == target.parent_version), None
            )

        # Find children
        children = [v for v in versions if v.parent_version == version]

        return {
            "model_name": model_name,
            "version": version,
            "parent": asdict(parent) if parent else None,
            "children": [asdict(c) for c in children],
        }

    def export_for_git_lfs(self, model_name: str, version: str, export_dir: str):
        """
        Export model for Git LFS tracking.

        Creates symlinks and .gitattributes entries.
        """
        model_path = self.versions_dir / model_name / version / "model.pt"
        export_path = Path(export_dir) / f"{model_name}_{version}.pt"

        # Copy model
        shutil.copy(str(model_path), str(export_path))

        # Create .gitattributes entry
        gitattrs_path = Path(export_dir) / ".gitattributes"
        with open(gitattrs_path, "a") as f:
            f.write(f"{model_name}_{version}.pt filter=lfs diff=lfs merge=lfs -text\n")

        return export_path


# Singleton instance
_registry = None


def get_registry(registry_dir: str = "models/registry") -> ModelRegistry:
    """Get or create singleton registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(registry_dir)
    return _registry
