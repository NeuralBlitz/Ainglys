"""File-based dataset snapshots with reproducible, non-pickle version history.

This compatibility manager handles ordinary files within one designated directory.
It does not replace DatasetRepository's tensor/pandas-specific versioning interface.
"""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone
from uuid import uuid4


_VERSION = re.compile(r"^[0-9a-f]{32}$")


class DatasetVersionManager:
    """Snapshot and restore files without executing serialized Python objects."""

    def __init__(self, repo_path: str | Path):
        self.repo_path = str(repo_path)
        self.root = Path(repo_path).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.storage = self.root / ".dataset-versions"
        self.storage.mkdir(exist_ok=True)

    def _version_dir(self, version: str) -> Path:
        if not _VERSION.fullmatch(version):
            raise ValueError("Invalid dataset version ID")
        folder = self.storage / version
        if not (folder / "manifest.json").is_file():
            raise ValueError("Unknown dataset version")
        return folder

    def _tracked_files(self):
        for file in sorted(self.root.rglob("*")):
            if not file.is_file() or file.is_symlink():
                continue
            relative = file.relative_to(self.root)
            if relative.parts[0] == ".dataset-versions":
                continue
            yield file, relative

    def _safe_target(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Unsafe dataset path")
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root) or target == self.root:
            raise ValueError("Dataset target escapes repository")
        return target

    def _manifest(self, version: str) -> tuple[Path, dict]:
        folder = self._version_dir(version)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest.get("files"), dict):
            raise ValueError("Malformed dataset manifest")
        return folder, manifest

    def commit(self, message: str, author: str) -> str:
        """Record tracked file contents and metadata as a new immutable snapshot."""
        version = uuid4().hex
        folder = self.storage / version
        folder.mkdir()
        files = {}
        for source, relative in self._tracked_files():
            dest = folder / "files" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            files[relative.as_posix()] = hashlib.sha256(dest.read_bytes()).hexdigest()
        metadata = {
            "version": version,
            "message": str(message),
            "author": str(author),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
        }
        (folder / "manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        return version

    def list_versions(self) -> list[str]:
        """List saved versions in creation order."""
        versions = []
        for folder in self.storage.iterdir():
            if folder.is_dir() and _VERSION.fullmatch(folder.name):
                try:
                    _, manifest = self._manifest(folder.name)
                    versions.append((manifest["created_at"], folder.name))
                except (ValueError, KeyError, OSError, json.JSONDecodeError):
                    continue
        return [version for _, version in sorted(versions)]

    def checkout(self, version: str) -> None:
        """Restore snapshot files; verify hashes before overwriting any target."""
        folder, manifest = self._manifest(version)
        pending = []
        for relative, checksum in manifest["files"].items():
            target = self._safe_target(relative)
            source = folder / "files" / relative
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"Missing dataset snapshot file: {relative}")
            if hashlib.sha256(source.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Dataset snapshot checksum mismatch: {relative}")
            pending.append((source, target))
        for source, target in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    def diff(self, old_version: str, new_version: str) -> str:
        """Generate a unified text diff for changed tracked files."""
        old_dir, old_meta = self._manifest(old_version)
        new_dir, new_meta = self._manifest(new_version)
        lines = []
        for relative in sorted(set(old_meta["files"]) | set(new_meta["files"])):
            self._safe_target(relative)
            old_file = old_dir / "files" / relative
            new_file = new_dir / "files" / relative
            old_text = old_file.read_text(errors="replace").splitlines(keepends=True) if old_file.is_file() else []
            new_text = new_file.read_text(errors="replace").splitlines(keepends=True) if new_file.is_file() else []
            lines.extend(difflib.unified_diff(
                old_text, new_text, fromfile=f"{old_version}/{relative}",
                tofile=f"{new_version}/{relative}"
            ))
        return "".join(lines)
