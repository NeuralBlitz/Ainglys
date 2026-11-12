"""Security and round-trip invariants for non-executable dataset snapshots."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from ai_hub.dataset_versioning import DatasetRepository, DatasetVersion
from ai_hub.safe_dataset_codec import (
    MAX_DATASET_BYTES, decode_dataset, encode_dataset,
)


def test_numpy_roundtrip_is_non_pickle_and_checksum_is_sha256(tmp_path):
    repo = DatasetRepository(str(tmp_path))
    values = np.arange(30, dtype=np.float64).reshape(10, 3)
    version = repo.commit(values, "measurements", description="Numeric experiment")
    saved = repo.versions[version]
    assert saved.storage_format == "npy"
    assert saved.data_file.endswith(".npy")
    assert len(saved.checksum) == 64
    assert not list((tmp_path / "data").glob("*.pkl"))
    np.testing.assert_array_equal(repo.checkout(version), values)

    reloaded = DatasetRepository(str(tmp_path))
    np.testing.assert_array_equal(reloaded.checkout(version), values)
    assert reloaded.versions[version].description == "Numeric experiment"


def test_dataframe_roundtrip(tmp_path):
    repo = DatasetRepository(str(tmp_path))
    frame = pd.DataFrame({"response": [2, 4, 8], "label": ["a", "b", "c"]})
    version = repo.commit(frame, "responses")
    loaded = repo.checkout(version)
    assert isinstance(loaded, pd.DataFrame)
    pd.testing.assert_frame_equal(loaded, frame, check_dtype=False)
    assert repo.versions[version].storage_format == "dataframe-json"


def test_json_structured_dataset_roundtrip_and_unique_versions(tmp_path):
    repo = DatasetRepository(str(tmp_path))
    original = [{"sample": [0.1, 0.2], "label": 0}, {"sample": [0.3, 0.4], "label": 1}]
    first = repo.commit(original, "examples", metadata={"project": "Ainglys"}, tags=["safe"])
    second = repo.commit({"new": True}, "examples")
    assert first != second
    assert repo.checkout(first) == original
    assert repo.checkout(second) == {"new": True}
    assert repo.versions[first].metadata["project"] == "Ainglys"
    assert len(repo.log()) == 2


@pytest.mark.parametrize("name", ["", "../other", "..", "model/test", ".hidden", "x" * 65])
def test_dataset_name_validation_prevents_path_traversal(tmp_path, name):
    repo = DatasetRepository(str(tmp_path))
    with pytest.raises(ValueError):
        repo.commit({"a": 1}, name)
    assert not list((tmp_path / "data").iterdir())


def test_reject_numpy_object_arrays_and_unsupported_objects(tmp_path):
    repo = DatasetRepository(str(tmp_path))
    with pytest.raises(TypeError):
        repo.commit(np.array([{"x": 1}], dtype=object), "objects")
    with pytest.raises(TypeError):
        repo.commit(object(), "instance")


def test_tampered_file_is_rejected_before_decode(tmp_path):
    repo = DatasetRepository(str(tmp_path))
    version = repo.commit(np.array([1, 2, 3]), "values")
    file_path = tmp_path / "data" / repo.versions[version].data_file
    with file_path.open("ab") as out:
        out.write(b"untrusted trailing bytes")
    with pytest.raises(ValueError, match="checksum"):
        repo.checkout(version)


def test_symlink_snapshot_is_rejected(tmp_path):
    repo = DatasetRepository(str(tmp_path / "repository"))
    version = repo.commit({"ok": True}, "safe")
    target = repo.data_dir / repo.versions[version].data_file
    backup = tmp_path / "untrusted.json"
    backup.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(backup)
    with pytest.raises(ValueError, match="symlink"):
        repo.checkout(version)


def test_path_in_index_cannot_escape_snapshot_directory(tmp_path):
    repo = DatasetRepository(str(tmp_path))
    version = repo.commit({"ok": True}, "safe")
    repo.versions[version].data_file = "../outside.json"
    with pytest.raises(ValueError, match="file name"):
        repo.checkout(version)


def test_unsupported_snapshot_format_is_rejected(tmp_path):
    repo = DatasetRepository(str(tmp_path))
    version = repo.commit({"ok": True}, "safe")
    repo.versions[version].storage_format = "pickle"
    with pytest.raises(ValueError, match="format"):
        repo.checkout(version)


def test_legacy_pickle_snapshot_is_never_unpickled(tmp_path):
    marker = tmp_path / "should-not-be-created"

    class Payload:
        def __reduce__(self):
            from pathlib import Path
            return (Path.write_text, (marker, "executed"))

    repo = DatasetRepository(str(tmp_path))
    version = "v1_20200101_000000"
    pickle_blob = pickle.dumps(Payload())
    (repo.data_dir / f"legacy_{version}.pkl").write_bytes(pickle_blob)
    repo.versions[version] = DatasetVersion(
        version_id=version, parent_version=None,
        created_at="2020-01-01T00:00:00", checksum=hashlib.md5(pickle_blob).hexdigest()[:16],
        num_samples=1, features=[], metadata={}, tags=[], description="legacy pickle",
    )
    repo._save_index()

    reopened = DatasetRepository(str(tmp_path))
    with pytest.raises(ValueError, match="pickle"):
        reopened.checkout(version)
    assert not marker.exists()


def test_snapshot_max_size_rejection():
    with pytest.raises(ValueError, match="limit"):
        decode_dataset(b"x" * (MAX_DATASET_BYTES + 1), "json")


def test_json_rejects_nan_without_writing_snapshot(tmp_path):
    repo = DatasetRepository(str(tmp_path))
    with pytest.raises(ValueError):
        repo.commit({"not_finite": float("nan")}, "bad")
    assert not list((tmp_path / "data").iterdir())


def test_object_array_loader_rejects_pickle_payload():
    # NumPy object arrays are pickle-based and must never load on checkout.
    buffer = __import__("io").BytesIO()
    np.save(buffer, np.array([{"x": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="Object arrays cannot be loaded"):
        decode_dataset(buffer.getvalue(), "npy")
