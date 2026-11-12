"""Checksum-verified file snapshots within a designated Ainglys directory.

No unpickling or execution of saved content. Treat version manifests as
untrusted metadata: validate paths and digest each file before restore or diff.
This is local snapshot storage, not a multi-user transactional database.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone
from uuid import uuid4

_VERSION = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_FILES = 1000
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_DIFF_INPUT_BYTES = 1024 * 1024
MAX_DIFF_OUTPUT_BYTES = 4 * 1024 * 1024


class DatasetVersionManager:
    """Version and restore ordinary files, refusing symlink/manifest escapes."""

    def __init__(self, repo_path: str | Path):
        self.repo_path = str(repo_path)
        self.root = Path(repo_path).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.storage = self.root / ".dataset-versions"
        self._check_storage()
        self.storage.mkdir(exist_ok=True)
        self._check_storage()

    def _check_storage(self) -> None:
        if self.storage.is_symlink():
            raise ValueError("Snapshot storage cannot be a symlink")

    @staticmethod
    def _relative_path(relative: str) -> Path:
        if not isinstance(relative, str) or not relative or len(relative) > 1024:
            raise ValueError("Unsafe dataset path")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Unsafe dataset path")
        # Dotfiles and VCS internals are not automatically included; avoid
        # accidentally snapshotting .env credentials or .git metadata.
        if any(part.startswith(".") for part in path.parts):
            raise ValueError("Hidden or internal files cannot be snapshotted")
        if "\x00" in relative:
            raise ValueError("Unsafe dataset path")
        return path

    @staticmethod
    def _reject_symlink_components(base: Path, relative: Path) -> Path:
        candidate = base
        for component in relative.parts:
            candidate = candidate / component
            if candidate.is_symlink():
                raise ValueError("Symlinked snapshot path is not permitted")
        return candidate

    def _safe_target(self, relative: str) -> Path:
        path = self._relative_path(relative)
        target = self._reject_symlink_components(self.root, path)
        if not target.resolve().is_relative_to(self.root) or target == self.root:
            raise ValueError("Dataset target escapes repository")
        return target

    def _version_dir(self, version: str) -> Path:
        if not isinstance(version, str) or not _VERSION.fullmatch(version):
            raise ValueError("Invalid dataset version ID")
        self._check_storage()
        folder = self.storage / version
        if folder.is_symlink():
            raise ValueError("Snapshot version directory is symlinked")
        manifest = folder / "manifest.json"
        if manifest.is_symlink() or not manifest.is_file():
            raise ValueError("Unknown or unsafe dataset version")
        return folder

    def _tracked_files(self):
        for file in sorted(self.root.rglob("*")):
            relative = file.relative_to(self.root)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if file.is_symlink() or not file.is_file():
                continue
            self._safe_target(relative.as_posix())
            yield file, relative

    def _manifest(self, version: str) -> tuple[Path, dict]:
        folder = self._version_dir(version)
        manifest_path = folder / "manifest.json"
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("Snapshot manifest exceeds size limit")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Malformed dataset manifest") from exc
        if not isinstance(manifest, dict) or manifest.get("version") != version:
            raise ValueError("Snapshot manifest version mismatch")
        files = manifest.get("files")
        if not isinstance(files, dict) or len(files) > MAX_FILES:
            raise ValueError("Malformed snapshot file inventory")
        for relative, checksum in files.items():
            self._safe_target(relative)
            if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
                raise ValueError("Invalid snapshot digest")
        return folder, manifest

    def _read_snapshot(self, folder: Path, relative: str, checksum: str) -> bytes:
        file_path = self._relative_path(relative)
        source_root = folder / "files"
        if source_root.is_symlink():
            raise ValueError("Snapshot files root cannot be symlinked")
        source = self._reject_symlink_components(source_root, file_path)
        if not source.resolve().is_relative_to(source_root.resolve()):
            raise ValueError("Snapshot source escapes version directory")
        if not source.is_file() or source.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Snapshot missing or oversized: {relative}")
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError(f"Dataset snapshot checksum mismatch: {relative}")
        return payload

    def commit(self, message: str, author: str) -> str:
        """Snapshot ordinary files and record their SHA-256 hashes."""
        self._check_storage()
        version = uuid4().hex
        folder = self.storage / version
        folder.mkdir()
        files: dict[str, str] = {}
        total_size = 0
        try:
            for source, relative in self._tracked_files():
                if len(files) >= MAX_FILES:
                    raise ValueError("Snapshot contains too many files")
                size = source.stat().st_size
                if size > MAX_FILE_BYTES or total_size + size > MAX_TOTAL_BYTES:
                    raise ValueError("Snapshot file or total size limit exceeded")
                payload = source.read_bytes()
                total_size += len(payload)
                if len(payload) > MAX_FILE_BYTES or total_size > MAX_TOTAL_BYTES:
                    raise ValueError("Snapshot file or total size limit exceeded")
                dest = folder / "files" / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("xb") as output:
                    output.write(payload)
                files[relative.as_posix()] = hashlib.sha256(payload).hexdigest()
            metadata = {
                "version": version,
                "message": str(message),
                "author": str(author),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "files": files,
            }
            serialized = json.dumps(metadata, indent=2, sort_keys=True)
            if len(serialized.encode("utf-8")) > MAX_MANIFEST_BYTES:
                raise ValueError("Snapshot manifest exceeds size limit")
            (folder / "manifest.json").write_text(serialized, encoding="utf-8")
        except Exception:
            # Do not leave half-complete snapshots that appear as valid versions.
            import shutil
            shutil.rmtree(folder)
            raise
        return version

    def list_versions(self) -> list[str]:
        """List well-formed manifests in creation order."""
        self._check_storage()
        versions = []
        for folder in self.storage.iterdir():
            if folder.is_dir() and not folder.is_symlink() and _VERSION.fullmatch(folder.name):
                try:
                    _, manifest = self._manifest(folder.name)
                    versions.append((manifest["created_at"], folder.name))
                except (ValueError, KeyError, OSError, json.JSONDecodeError, TypeError):
                    continue
        return [version for _, version in sorted(versions)]

    def checkout(self, version: str) -> None:
        """Verify all source files first, then restore them to the working tree.

        Do not promise atomic multi-file restoration: I/O failures can occur
        during the write phase. The versioned originals are never overwritten.
        """
        folder, manifest = self._manifest(version)
        pending: list[tuple[str, bytes]] = []
        total = 0
        for relative, checksum in manifest["files"].items():
            payload = self._read_snapshot(folder, relative, checksum)
            total += len(payload)
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Snapshot total size limit exceeded")
            pending.append((relative, payload))

        for relative, payload in pending:
            target = self._safe_target(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target = self._safe_target(relative)
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(target, flags, 0o600)
            with os.fdopen(fd, "wb") as output:
                output.write(payload)

    def diff(self, old_version: str, new_version: str) -> str:
        """Diff checksum-verified text files without following snapshot symlinks."""
        old_dir, old_meta = self._manifest(old_version)
        new_dir, new_meta = self._manifest(new_version)
        changes: list[str] = []
        total_chars = 0
        for relative in sorted(set(old_meta["files"]) | set(new_meta["files"])):
            self._safe_target(relative)
            before = (
                self._read_snapshot(old_dir, relative, old_meta["files"][relative])
                if relative in old_meta["files"] else b""
            )
            after = (
                self._read_snapshot(new_dir, relative, new_meta["files"][relative])
                if relative in new_meta["files"] else b""
            )
            if max(len(before), len(after)) > MAX_DIFF_INPUT_BYTES:
                raise ValueError("Text diff input exceeds size limit")
            lines = list(difflib.unified_diff(
                before.decode("utf-8", errors="replace").splitlines(keepends=True),
                after.decode("utf-8", errors="replace").splitlines(keepends=True),
                fromfile=f"{old_version}/{relative}",
                tofile=f"{new_version}/{relative}",
            ))
            total_chars += sum(len(line) for line in lines)
            if total_chars > MAX_DIFF_OUTPUT_BYTES:
                raise ValueError("Text diff output exceeds size limit")
            changes.extend(lines)
        return "".join(changes)
