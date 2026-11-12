# File snapshot integrity and restore constraints

Ainglys has two separate dataset versioning systems. `DatasetRepository` stores typed NumPy, Pandas JSON and ordinary JSON data (see [safe dataset snapshots](SAFE_DATASET_SNAPSHOTS.md)). `DatasetVersionManager` snapshots **ordinary files** in one designated working directory and restores or diffs them without decoding Python objects.

## Security rules

- A snapshot's `.dataset-versions/<version>/manifest.json` is treated as **untrusted input**, even when the snapshot belongs to the local repository.
- Before checkout or diff, the manager validates the version ID, manifest size and structure, manifest version identifier, each relative file path and SHA-256 digest.
- Every snapshot source is checked for missing files, symlinked path components, size limits and SHA-256 mismatch **before** its content is exposed to a diff or written to the workspace.
- Restore targets cannot leave the designated root, enter hidden/internal directories, or traverse symlinks. Each restored file is opened with a no-follow flag where supported.
- Snapshot creation excludes dotfiles and hidden directories, including `.env` files, `.git` and the internal `.dataset-versions` store. It also skips symlinked files. This prevents accidental copying of common credentials and VCS metadata.
- Size limits: 1,000 files; 32 MiB per file; 128 MiB total; 1 MiB manifest; 1 MiB per input file for text diff; 4 MiB maximum diff output. Very large projects need a purpose-built artifact store instead.

## Usage

```python
from pathlib import Path
from ai_hub.dataset_versioning import DatasetVersionManager

root = Path("datasets/experiment-files")
root.mkdir(parents=True, exist_ok=True)
(root / "measurements.csv").write_text("sample,value\n1,2\n")

snapshots = DatasetVersionManager(root)
before = snapshots.commit("Original data", "researcher")
(root / "measurements.csv").write_text("sample,value\n1,3\n")
after = snapshots.commit("Adjusted observation", "researcher")

print(snapshots.diff(before, after))
snapshots.checkout(before)
```

## Limitations and threat model

- This is **not** a cryptographically signed or immutable repository. Anyone able to replace both manifest and content can substitute an alternate snapshot with internally consistent hashes.
- Multi-file checkout is **not atomic**: all source files are prevalidated before writes, but an I/O failure during restoration can leave some targets updated and others not. Back up important working data before checkout.
- Concurrent hostile filesystem mutation between path validation and file opening is not fully prevented; the manager assumes the snapshot directory is not concurrently writable by untrusted users. Use filesystem permissions or isolated service accounts to enforce that boundary.
- Hidden files are intentionally excluded, so this is not a full directory backup system.
- String diffs are best-effort decoded as UTF-8 with replacement, not a binary-aware data comparison. Oversized diffs raise a validation error.
- Storage quotas, encrypted content, remote synchronization, signed provenance, journaled transactions and audited multi-user access remain future work.

Tests in `tests/test_dataset_version_manager_security.py` cover safe round-trips and adversarial cases including manifest traversal, symlinked restore targets and sources, tampering, size limits and hidden-secret exclusions. 
