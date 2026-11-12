# Safe dataset snapshots and legacy pickle migration

Ainglys's `DatasetRepository` now stores new dataset versions in **non-executable** formats. It no longer unpickles repository data during normal `checkout()` operations.

## Supported snapshot formats

| Source data | Snapshot | Loading behavior |
| --- | --- | --- |
| NumPy arrays with non-object dtypes | `.npy` | `numpy.load(..., allow_pickle=False)` |
| Pandas DataFrames with string column labels | `.frame.json` | Pandas JSON split orientation |
| JSON-compatible dictionaries, lists and scalar values | `.json` | Standard JSON parser |

The repository refuses arbitrary Python objects and NumPy object arrays. The saved payload is bounded to 32 MiB. Dataset identifiers must contain only letters, digits, hyphens and underscores (1–64 characters). Files are opened exclusively for creation to avoid accidental overwrites.

Each new version records its exact relative filename, format and SHA-256 checksum in `index.json`. Checkouts reject missing files, symlinks, invalid paths/formats, oversized payloads and checksum mismatches **before decoding**. The SHA-256 value provides accidental or unauthorized modification detection *when the index is trusted*. It is not a signature: an attacker who can modify both the index and the data can replace both.

## Usage

```python
import numpy as np
import pandas as pd
from ai_hub.dataset_versioning import DatasetRepository

repo = DatasetRepository("datasets/research")
array_version = repo.commit(np.array([[1.0, 2.0], [3.0, 4.0]]), "experiment_01")
array = repo.checkout(array_version)

frame_version = repo.commit(
    pd.DataFrame({"feature": [1, 2], "label": ["a", "b"]}),
    "tabular_dataset",
    description="Initial trusted import",
)
frame = repo.checkout(frame_version)

json_version = repo.commit({"features": [[0.1, 0.2]], "labels": [1]}, "json_example")
print(repo.checkout(json_version))
```

DataFrame JSON serialization preserves the tabular values but does **not** guarantee exact dtype or timezone round-tripping for every Pandas extension dtype. Verify the schema for scientific reproducibility, especially datetime, categorical and nullable dtypes.

## Breaking security change: historic pickle snapshots

Older `DatasetRepository` versions saved `.pkl` files and called `pickle.load()` on checkout. The new reader deliberately raises `ValueError` for those versions, rather than running untrusted code. An old `index.json` without the new `storage_format` and `data_file` fields is recognized as **legacy pickle** and cannot be loaded through the normal application API.

For old datasets:
1. Prefer re-exporting from the **original trusted source** as NumPy numeric arrays, Pandas DataFrames or JSON.
2. If no source exists, only recover a legacy pickle in a separately isolated, offline, disposable environment after independently establishing its provenance. Do not open untrusted pickle files.
3. Once recovered, inspect the resulting data and commit it into a **new** Ainglys repository snapshot using the safe formats above. Keep provenance and transformation metadata externally or in the `metadata` mapping.
4. Verify the new snapshot round-trips, passes integrity checks and reproduces the required analytical results.

There is intentionally **no in-process unsafe pickle fallback** and no public unpickling migration endpoint.

## Remaining hardening

The file-based `DatasetVersionManager` uses its own checksummed snapshots. Neither manager is an adversarial multi-user data lake: filesystem permissions, index authenticity, atomic transaction recovery, disk quotas, storage encryption, signed manifests, schema normalization and dataset provenance still need further work. Avoid using an untrusted snapshot index from an externally writable directory.
