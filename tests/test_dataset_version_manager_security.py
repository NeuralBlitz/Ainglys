"""Adversarial checks for DatasetVersionManager filesystem snapshots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ai_hub.dataset_version_manager import DatasetVersionManager, MAX_FILE_BYTES


def manifest_path(manager, version):
    return manager.storage / version / "manifest.json"


def alter_manifest(manager, version, callback):
    path = manifest_path(manager, version)
    data = json.loads(path.read_text(encoding="utf-8"))
    callback(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_snapshot_roundtrip_diff_and_restart(tmp_path):
    manager = DatasetVersionManager(tmp_path)
    file = tmp_path / "measurements.csv"
    file.write_text("value\n1\n", encoding="utf-8")
    v1 = manager.commit("First", "tester")
    file.write_text("value\n2\n", encoding="utf-8")
    v2 = manager.commit("Second", "tester")
    assert v1 != v2
    assert set(manager.list_versions()) == {v1, v2}
    delta = manager.diff(v1, v2)
    assert "-1" in delta and "+2" in delta
    restarted = DatasetVersionManager(tmp_path)
    restarted.checkout(v1)
    assert file.read_text(encoding="utf-8") == "value\n1\n"
    # The versioned source remains intact after restore.
    restarted.checkout(v2)
    assert file.read_text(encoding="utf-8") == "value\n2\n"


def test_hidden_files_and_internal_repository_are_not_snapshotted(tmp_path):
    manager = DatasetVersionManager(tmp_path)
    (tmp_path / "allowed.txt").write_text("ok")
    (tmp_path / ".env").write_text("SECRET=do-not-snapshot")
    nested = tmp_path / ".git"
    nested.mkdir()
    (nested / "config").write_text("private")
    v = manager.commit("No secrets", "tester")
    data = json.loads(manifest_path(manager, v).read_text())
    assert list(data["files"]) == ["allowed.txt"]


@pytest.mark.parametrize("relative", [
    "../outside.txt",
    "/tmp/outside.txt",
    ".dataset-versions/corrupt.txt",
    ".env",
    ".git/config",
    "",
])
def test_malicious_manifest_cannot_target_internal_or_external_files(tmp_path, relative):
    manager = DatasetVersionManager(tmp_path)
    (tmp_path / "ordinary.txt").write_text("safe")
    version = manager.commit("Base", "tester")

    def rewrite(data):
        data["files"] = {relative: hashlib.sha256(b"tamper").hexdigest()}

    alter_manifest(manager, version, rewrite)
    with pytest.raises(ValueError):
        manager.checkout(version)
    with pytest.raises(ValueError):
        manager.diff(version, version)
    assert (tmp_path / "ordinary.txt").read_text() == "safe"


def test_checksum_change_is_refused_during_checkout_and_diff(tmp_path):
    manager = DatasetVersionManager(tmp_path)
    source = tmp_path / "record.txt"
    source.write_text("untampered")
    version = manager.commit("Base", "tester")
    snap = manager.storage / version / "files" / "record.txt"
    snap.write_text("altered")
    with pytest.raises(ValueError, match="checksum"):
        manager.checkout(version)
    with pytest.raises(ValueError, match="checksum"):
        manager.diff(version, version)
    assert source.read_text() == "untampered"


def test_symlinked_source_file_is_refused_even_when_hash_matches(tmp_path):
    manager = DatasetVersionManager(tmp_path / "repo")
    source = manager.root / "record.txt"
    source.write_text("safe")
    version = manager.commit("Base", "tester")
    snap = manager.storage / version / "files" / "record.txt"
    outside = tmp_path / "outside.txt"
    outside.write_text("safe")
    snap.unlink()
    snap.symlink_to(outside)
    with pytest.raises(ValueError, match="Symlink"):
        manager.checkout(version)
    with pytest.raises(ValueError, match="Symlink"):
        manager.diff(version, version)


def test_symlinked_source_directory_is_refused(tmp_path):
    manager = DatasetVersionManager(tmp_path / "repo")
    folder = manager.root / "reports"
    folder.mkdir()
    (folder / "record.txt").write_text("safe")
    version = manager.commit("Base", "tester")
    source_dir = manager.storage / version / "files" / "reports"
    (source_dir / "record.txt").unlink()
    source_dir.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "record.txt").write_text("safe")
    source_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        manager.checkout(version)
    with pytest.raises(ValueError, match="Symlink"):
        manager.diff(version, version)


def test_symlinked_target_directory_cannot_redirect_restore(tmp_path):
    manager = DatasetVersionManager(tmp_path / "repo")
    folder = manager.root / "reports"
    folder.mkdir()
    (folder / "record.txt").write_text("safe")
    version = manager.commit("Base", "tester")
    (folder / "record.txt").unlink()
    folder.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "record.txt"
    protected.write_text("do-not-overwrite")
    folder.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        manager.checkout(version)
    assert protected.read_text() == "do-not-overwrite"


def test_symlinked_storage_directory_is_refused(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (root / ".dataset-versions").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        DatasetVersionManager(root)


def test_symlinked_version_directory_is_refused(tmp_path):
    manager = DatasetVersionManager(tmp_path)
    (tmp_path / "data.txt").write_text("safe")
    version = manager.commit("Base", "tester")
    folder = manager.storage / version
    shadow = tmp_path / "shadow"
    folder.rename(shadow)
    folder.symlink_to(shadow, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        manager.checkout(version)


def test_invalid_manifest_version_and_digest_refused(tmp_path):
    manager = DatasetVersionManager(tmp_path)
    (tmp_path / "file.txt").write_text("safe")
    version = manager.commit("Base", "tester")
    alter_manifest(manager, version, lambda m: m.update(version="0" * 32))
    with pytest.raises(ValueError, match="version"):
        manager.checkout(version)

    alter_manifest(manager, version, lambda m: m.update(
        version=version, files={"file.txt": "not-a-sha256"}
    ))
    with pytest.raises(ValueError, match="digest"):
        manager.checkout(version)


def test_oversized_snapshot_file_rejected_and_incomplete_version_cleaned(tmp_path):
    manager = DatasetVersionManager(tmp_path)
    file = tmp_path / "huge.dat"
    with file.open("wb") as out:
        out.truncate(MAX_FILE_BYTES + 1)
    with pytest.raises(ValueError, match="size limit"):
        manager.commit("Large", "tester")
    assert manager.list_versions() == []


def test_binary_or_large_text_diff_is_bounded(tmp_path):
    manager = DatasetVersionManager(tmp_path)
    file = tmp_path / "long.txt"
    file.write_text("a" * (1024 * 1024 + 1))
    first = manager.commit("First", "tester")
    file.write_text("b" * (1024 * 1024 + 1))
    second = manager.commit("Second", "tester")
    with pytest.raises(ValueError, match="diff input exceeds"):
        manager.diff(first, second)
