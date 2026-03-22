"""
Dataset Versioning and Management

Provides Git-like versioning for datasets with:
- Version control
- Lineage tracking
- Differential updates
- Remote storage backends
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import pickle


@dataclass
class DatasetVersion:
    """Represents a dataset version."""

    version_id: str
    parent_version: Optional[str]
    created_at: str
    checksum: str
    num_samples: int
    features: List[str]
    metadata: Dict[str, Any]
    tags: List[str]
    description: str


@dataclass
class DatasetDiff:
    """Represents changes between dataset versions."""

    added_samples: int
    removed_samples: int
    modified_samples: int
    feature_changes: List[str]


class DatasetRepository:
    """
    Git-like version control for datasets.

    Features:
    - Version snapshots
    - Branch management
    - Diffs between versions
    - Tagging
    - Remote synchronization
    """

    def __init__(self, repo_path: str = "datasets/registry"):
        self.repo_path = Path(repo_path)
        self.repo_path.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.data_dir = self.repo_path / "data"
        self.versions_dir = self.repo_path / "versions"
        self.branches_dir = self.repo_path / "branches"

        for d in [self.data_dir, self.versions_dir, self.branches_dir]:
            d.mkdir(exist_ok=True)

        # In-memory index
        self.versions: Dict[str, DatasetVersion] = {}
        self.branches: Dict[str, str] = {}  # branch_name -> current_version
        self.current_branch = "main"

        self._load_index()

    def _load_index(self):
        """Load repository index."""
        index_file = self.repo_path / "index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                data = json.load(f)
                self.versions = {
                    k: DatasetVersion(**v) for k, v in data.get("versions", {}).items()
                }
                self.branches = data.get("branches", {"main": None})
                self.current_branch = data.get("current_branch", "main")

    def _save_index(self):
        """Save repository index."""
        index_file = self.repo_path / "index.json"
        data = {
            "versions": {k: asdict(v) for k, v in self.versions.items()},
            "branches": self.branches,
            "current_branch": self.current_branch,
        }
        with open(index_file, "w") as f:
            json.dump(data, f, indent=2)

    def _compute_checksum(self, data_path: Path) -> str:
        """Compute checksum of dataset."""
        hash_md5 = hashlib.md5()

        if data_path.is_file():
            with open(data_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        else:
            # For directories, hash all files
            for file_path in sorted(data_path.rglob("*")):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)

        return hash_md5.hexdigest()[:16]

    def commit(
        self,
        data: Any,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Commit a new dataset version.

        Args:
            data: Dataset to version (numpy array, pandas DataFrame, etc.)
            name: Dataset name
            description: Version description
            tags: List of tags
            metadata: Additional metadata

        Returns:
            Version ID
        """
        # Save data
        version_id = (
            f"v{len(self.versions) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        data_path = self.data_dir / f"{name}_{version_id}.pkl"

        with open(data_path, "wb") as f:
            pickle.dump(data, f)

        # Compute checksum
        checksum = self._compute_checksum(data_path)

        # Get parent version
        parent_version = self.branches.get(self.current_branch)

        # Infer dataset info
        num_samples = len(data) if hasattr(data, "__len__") else 0
        features = list(data.columns) if hasattr(data, "columns") else []

        # Create version
        version = DatasetVersion(
            version_id=version_id,
            parent_version=parent_version,
            created_at=datetime.now().isoformat(),
            checksum=checksum,
            num_samples=num_samples,
            features=features,
            metadata=metadata or {},
            tags=tags or [],
            description=description,
        )

        self.versions[version_id] = version
        self.branches[self.current_branch] = version_id

        self._save_index()

        print(f"✅ Committed {name} version {version_id}")
        print(f"   Samples: {num_samples}, Features: {len(features)}")

        return version_id

    def checkout(self, version_id: str) -> Any:
        """Checkout a specific dataset version."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        # Find data file
        data_files = list(self.data_dir.glob(f"*{version_id}.pkl"))
        if not data_files:
            raise ValueError(f"Data file for version {version_id} not found")

        with open(data_files[0], "rb") as f:
            data = pickle.load(f)

        return data

    def log(self, max_entries: int = 10) -> List[DatasetVersion]:
        """Get version history (like git log)."""
        versions = []
        current = self.branches.get(self.current_branch)

        while current and len(versions) < max_entries:
            version = self.versions.get(current)
            if not version:
                break
            versions.append(version)
            current = version.parent_version

        return versions

    def diff(self, version_a: str, version_b: str) -> DatasetDiff:
        """Compute diff between two versions."""
        ver_a = self.versions.get(version_a)
        ver_b = self.versions.get(version_b)

        if not ver_a or not ver_b:
            raise ValueError("Version not found")

        return DatasetDiff(
            added_samples=ver_b.num_samples - ver_a.num_samples,
            removed_samples=0,  # Would need actual data comparison
            modified_samples=0,
            feature_changes=[],
        )

    def create_branch(self, branch_name: str, from_version: Optional[str] = None):
        """Create a new branch."""
        if from_version is None:
            from_version = self.branches.get(self.current_branch)

        self.branches[branch_name] = from_version
        self._save_index()

        print(f"✅ Created branch '{branch_name}' from {from_version or 'scratch'}")

    def switch_branch(self, branch_name: str):
        """Switch to a different branch."""
        if branch_name not in self.branches:
            raise ValueError(f"Branch {branch_name} not found")

        self.current_branch = branch_name
        self._save_index()

        print(f"✅ Switched to branch '{branch_name}'")

    def tag(self, version_id: str, tag_name: str):
        """Tag a version."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        self.versions[version_id].tags.append(tag_name)
        self._save_index()

        print(f"✅ Tagged {version_id} as '{tag_name}'")

    def list_versions(self, tag_filter: Optional[str] = None) -> List[DatasetVersion]:
        """List all versions, optionally filtered by tag."""
        versions = list(self.versions.values())

        if tag_filter:
            versions = [v for v in versions if tag_filter in v.tags]

        return sorted(versions, key=lambda x: x.created_at, reverse=True)


class DatasetLineage:
    """Track dataset lineage and provenance."""

    def __init__(self, repository: DatasetRepository):
        self.repo = repository

    def get_lineage(self, version_id: str) -> Dict[str, Any]:
        """Get full lineage of a dataset version."""
        version = self.repo.versions.get(version_id)
        if not version:
            return {}

        lineage = {
            "version_id": version_id,
            "created_at": version.created_at,
            "parent": version.parent_version,
            "ancestors": [],
            "transformations": version.metadata.get("transformations", []),
        }

        # Trace back through parents
        current = version.parent_version
        while current:
            parent = self.repo.versions.get(current)
            if parent:
                lineage["ancestors"].append(
                    {
                        "version_id": current,
                        "created_at": parent.created_at,
                        "description": parent.description,
                    }
                )
                current = parent.parent_version
            else:
                break

        return lineage

    def find_common_ancestor(self, version_a: str, version_b: str) -> Optional[str]:
        """Find common ancestor of two versions."""
        ancestors_a = set()

        # Collect all ancestors of version_a
        current = version_a
        while current:
            ancestors_a.add(current)
            version = self.repo.versions.get(current)
            current = version.parent_version if version else None

        # Check ancestors of version_b
        current = version_b
        while current:
            if current in ancestors_a:
                return current
            version = self.repo.versions.get(current)
            current = version.parent_version if version else None

        return None


# Export
__all__ = ["DatasetRepository", "DatasetVersion", "DatasetDiff", "DatasetLineage"]
