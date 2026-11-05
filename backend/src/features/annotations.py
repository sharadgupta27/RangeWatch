"""Expert record annotations (U11): effective state per record and how training applies it.

Annotations are append-only (`record_annotations`); the latest per (record, kind) is the
effective one. Training applies them after the data-quality tests:

  * `exclude` / `cultivated` = "true" → the record is left out of training;
  * `verified` = "true" → the record is kept even if an excluded quality test flagged it;
  * `range_label` → overrides the polygon-derived effective label (native / introduced /
    unknown), which also decides Model A vs B.

The effective table is snapshotted per model version (training_snapshot.ANNOTATIONS_FILE) and
summarised in the version's reproducibility block, so a retrain is auditable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.domain import RecordAnnotation

ANNOTATION_KINDS = ("range_label", "exclude", "cultivated", "verified")
RANGE_LABEL_VALUES = ("native", "introduced", "unknown")
BOOLEAN_VALUES = ("true", "false")
EFFECTIVE_COLUMNS = (
    "source",
    "record_id",
    "range_label",
    "exclude",
    "cultivated",
    "verified",
    "n_annotations",
    "last_author",
    "last_ts",
)


def validate_annotation(kind: str, value: str) -> str:
    v = value.strip().lower()
    allowed = RANGE_LABEL_VALUES if kind == "range_label" else BOOLEAN_VALUES
    if kind not in ANNOTATION_KINDS:
        raise ValueError(f"kind must be one of {ANNOTATION_KINDS}")
    if v not in allowed:
        raise ValueError(f"{kind} value must be one of {allowed}, not {value!r}")
    return v


def effective_annotations(annotations: Sequence[RecordAnnotation]) -> pd.DataFrame:
    """One row per annotated record: the latest value of each kind (None = never set)."""
    if not annotations:
        return pd.DataFrame(columns=list(EFFECTIVE_COLUMNS))
    df = pd.DataFrame(
        [
            {
                "source": a.source,
                "record_id": int(a.record_id),
                "kind": a.kind,
                "value": a.value,
                "author": a.author,
                "ts": pd.Timestamp(a.created_ts),
            }
            for a in annotations
        ]
    ).sort_values("ts", kind="stable")
    latest = df.groupby(["source", "record_id", "kind"], sort=False).last().reset_index()
    wide = latest.pivot(index=["source", "record_id"], columns="kind", values="value")
    meta = df.groupby(["source", "record_id"]).agg(
        n_annotations=("kind", "size"), last_author=("author", "last"), last_ts=("ts", "max")
    )
    out = wide.join(meta).reset_index()
    for kind in ANNOTATION_KINDS:
        if kind not in out:
            out[kind] = None
    for kind in ("exclude", "cultivated", "verified"):
        out[kind] = out[kind].map(lambda v: None if pd.isna(v) else v == "true")
    out["range_label"] = out["range_label"].map(lambda v: None if pd.isna(v) else v)
    out["last_ts"] = out["last_ts"].astype("string")
    return out[list(EFFECTIVE_COLUMNS)].sort_values(["source", "record_id"]).reset_index(drop=True)


def _lookup(occ: pd.DataFrame, eff: pd.DataFrame) -> pd.DataFrame:
    keys = pd.DataFrame(
        {"source": occ["source"].astype(str), "record_id": occ["record_id"].astype("int64")}
    )
    return keys.merge(eff.astype({"record_id": "int64"}), on=["source", "record_id"], how="left")


def apply_exclusions(
    occ: pd.DataFrame, quality_excluded: np.ndarray, eff: pd.DataFrame
) -> tuple[np.ndarray, dict[str, Any]]:
    """Training exclusion mask after expert annotations, and what changed."""
    excluded = np.asarray(quality_excluded, dtype=bool).copy()
    if eff.empty:
        return excluded, _summary(eff)
    m = _lookup(occ, eff)
    user_exclude = (m["exclude"] == True).to_numpy()  # noqa: E712  (None-aware)
    cultivated = (m["cultivated"] == True).to_numpy()  # noqa: E712
    verified = (m["verified"] == True).to_numpy()  # noqa: E712
    rescued = excluded & verified & ~user_exclude & ~cultivated
    excluded = (excluded & ~verified) | user_exclude | cultivated
    summary = _summary(eff)
    summary.update(
        n_excluded_by_annotation=int((user_exclude | cultivated).sum()),
        n_excluded_suspicious=int(user_exclude.sum()),
        n_excluded_cultivated=int(cultivated.sum()),
        n_kept_verified=int(rescued.sum()),
    )
    return excluded, summary


def apply_range_labels(
    occ: pd.DataFrame, labels: pd.Series, eff: pd.DataFrame
) -> tuple[pd.Series, int]:
    """Effective labels with expert overrides; returns the number of labels changed."""
    if eff.empty or eff["range_label"].isna().all():
        return labels, 0
    m = _lookup(occ, eff)
    override = m["range_label"].to_numpy()
    has = pd.notna(override)
    out = labels.copy().to_numpy(dtype=object)
    changed = int((has & (out != override)).sum())
    out[has] = override[has]
    return pd.Series(out, index=labels.index, name=labels.name), changed


def _summary(eff: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_records_annotated": len(eff),
        "n_annotations": int(eff["n_annotations"].sum()) if len(eff) else 0,
        "n_excluded_by_annotation": 0,
        "n_excluded_suspicious": 0,
        "n_excluded_cultivated": 0,
        "n_kept_verified": 0,
        "n_label_overrides": 0,
        "rules": "exclude / cultivated = true → left out of training; verified = true → kept "
        "despite an excluded data-quality test; range_label overrides the polygon label",
    }
