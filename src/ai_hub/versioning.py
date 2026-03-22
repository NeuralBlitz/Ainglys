import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib


class ModelVersioning:
    """Model versioning system."""

    def __init__(self, versions_dir: str = "models/versions"):
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.versions_dir / "versions.json"
        self._load_metadata()

    def _load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"versions": []}

    def _save_metadata(self):
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _compute_hash(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def save_version(
        self,
        model_path: str,
        version_name: Optional[str] = None,
        description: str = "",
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Save a new model version."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = version_name or f"v{timestamp}"

        version_dir = self.versions_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        dest_path = version_dir / "model.pt"
        shutil.copy2(model_path, dest_path)

        model_hash = self._compute_hash(model_path)

        version_info = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "metrics": metrics or {},
            "hash": model_hash,
            "model_file": str(dest_path),
        }

        self.metadata["versions"].append(version_info)
        self._save_metadata()

        return version

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all versions."""
        return self.metadata.get("versions", [])

    def get_version(self, version: str) -> Optional[Dict[str, Any]]:
        """Get a specific version."""
        for v in self.metadata.get("versions", []):
            if v["version"] == version:
                return v
        return None

    def load_version(self, version: str) -> Optional[str]:
        """Get the path to a model version."""
        v = self.get_version(version)
        return v["model_file"] if v else None

    def delete_version(self, version: str) -> bool:
        """Delete a version."""
        versions = self.metadata.get("versions", [])
        self.metadata["versions"] = [v for v in versions if v["version"] != version]
        self._save_metadata()

        version_dir = self.versions_dir / version
        if version_dir.exists():
            shutil.rmtree(version_dir)
            return True
        return False

    def compare_versions(self, version1: str, version2: str) -> Optional[Dict]:
        """Compare two versions."""
        v1 = self.get_version(version1)
        v2 = self.get_version(version2)

        if not v1 or not v2:
            return None

        return {
            "version1": v1,
            "version2": v2,
            "metrics_diff": {
                k: v2.get("metrics", {}).get(k, 0) - v1.get("metrics", {}).get(k, 0)
                for k in set(v1.get("metrics", {}).keys())
                | set(v2.get("metrics", {}).keys())
            },
        }
