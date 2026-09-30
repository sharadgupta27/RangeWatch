"""Immutable per-model-version snapshot of the data a model was trained on.

Occurrences are append-only in PostGIS, but a model's *training set* cannot be re-derived from
them exactly: it depends on the native-range polygon at training time (effective labels), on
feature values for the bioclim version used, and on seeded thinning / background sampling.
Each model version therefore stores, next to `model.pkl`:

- `training_data.parquet` — the exact MaxEnt input: every presence (with its source/record id)
  and background point, with coordinates and predictor values.
- `occurrence_manifest.parquet` — every valid record considered, with the ingested and the
  effective (polygon-derived) range label; this is what severity/spread and MESS were built on.

Both are content-addressed by SHA-256 (recorded in the model's reproducibility metadata and
logged to MLflow), so a version can be audited or re-trained bit-for-bit later.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRAINING_DATA_FILE = "training_data.parquet"
OCCURRENCE_MANIFEST_FILE = "occurrence_manifest.parquet"


@dataclass(frozen=True, slots=True)
class SnapshotFile:
    path: Path
    sha256: str
    n_rows: int
    bytes: int

    def describe(self, uri: str) -> dict[str, Any]:
        return {**asdict(self), "path": uri}


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, expected_sha256: str) -> bool:
    """True if the snapshot on disk is byte-identical to the one recorded at training time."""
    return path.exists() and file_sha256(path) == expected_sha256


def _write(df: pd.DataFrame, path: Path) -> SnapshotFile:
    df.to_parquet(path, index=False)
    return SnapshotFile(path, file_sha256(path), len(df), path.stat().st_size)


def write_training_snapshot(
    vdir: Path,
    presences: pd.DataFrame,
    background_lon: np.ndarray,
    background_lat: np.ndarray,
    background_x: pd.DataFrame,
    predictors: Sequence[str],
    occurrences: pd.DataFrame,
    effective_labels: pd.Series,
) -> dict[str, SnapshotFile]:
    preds = list(predictors)
    pres = presences[["source", "record_id", "longitude", "latitude", *preds]].copy()
    pres.insert(0, "role", "presence")
    bg = pd.DataFrame(
        {
            "role": "background",
            "source": pd.Series([None] * len(background_lon), dtype="string"),
            "record_id": pd.Series([pd.NA] * len(background_lon), dtype="Int64"),
            "longitude": np.asarray(background_lon, dtype="float64"),
            "latitude": np.asarray(background_lat, dtype="float64"),
        }
    )
    bg = pd.concat([bg, background_x[preds].reset_index(drop=True)], axis=1)
    pres = pres.astype({"source": "string", "record_id": "Int64"})
    training = pd.concat([pres, bg], ignore_index=True)

    wanted = ("source", "record_id", "longitude", "latitude", "event_date")
    cols = [c for c in wanted if c in occurrences]
    manifest = occurrences[cols].copy()
    manifest["range_label_ingested"] = occurrences["range_label"].to_numpy()
    manifest["range_label_effective"] = np.asarray(effective_labels)
    if "event_date" in manifest:
        manifest["event_date"] = manifest["event_date"].astype("string")
    manifest = manifest.astype({"source": "string", "record_id": "Int64"})
    manifest = manifest.sort_values(["source", "record_id"], kind="stable").reset_index(drop=True)

    vdir.mkdir(parents=True, exist_ok=True)
    return {
        "training_data": _write(training, vdir / TRAINING_DATA_FILE),
        "occurrence_manifest": _write(manifest, vdir / OCCURRENCE_MANIFEST_FILE),
    }
