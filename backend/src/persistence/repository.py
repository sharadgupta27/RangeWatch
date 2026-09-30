"""Repository protocol + an in-memory implementation (used by tests and local dry-runs).

Occurrence data is append-only: `append_occurrences` inserts only unseen (source, record_id)
keys and `fill_features` only fills bio* columns that are still NULL. Only species_registry,
native_ranges, model_versions (severity recompute) and jobs rows are updated in place.
"""

from __future__ import annotations

import copy
import threading
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, replace
from datetime import timedelta
from typing import Any, Protocol

import pandas as pd

from src.domain import (
    BIOCLIM_BANDS,
    Job,
    JobStatus,
    ModelVersionRecord,
    NativeRange,
    OccurrenceRecord,
    SpeciesRecord,
    utcnow,
)

OCCURRENCE_COLUMNS: tuple[str, ...] = (
    "source",
    "record_id",
    "longitude",
    "latitude",
    "event_date",
    "coordinate_uncertainty_m",
    "basis_of_record",
    "country_code",
    "range_label",
    "inat_crossref_id",
    "ingested_ts",
    *BIOCLIM_BANDS,
    "bioclim_version",
)


STALE_JOB_MESSAGE = "Marked failed: no progress heartbeat (worker lost or task crashed)"


class NotFoundError(KeyError):
    pass


class Repository(Protocol):
    # species registry
    def get_species(self, taxon_key: int) -> SpeciesRecord | None: ...
    def list_species(self) -> list[SpeciesRecord]: ...
    def create_species(self, rec: SpeciesRecord) -> SpeciesRecord: ...
    def update_species(self, taxon_key: int, **fields: Any) -> SpeciesRecord: ...

    # occurrences (append-only)
    def append_occurrences(self, records: Sequence[OccurrenceRecord]) -> int: ...
    def load_occurrences(self, taxon_key: int) -> pd.DataFrame: ...
    def occurrence_counts(self, taxon_key: int) -> dict[str, int]: ...
    def inat_crossref_ids(self, taxon_key: int) -> set[int]: ...
    def source_counts(self, taxon_key: int) -> dict[str, int]: ...
    def fill_features(
        self, taxon_key: int, keys: pd.DataFrame, features: pd.DataFrame, bioclim_version: str
    ) -> int: ...
    def page_occurrences(
        self,
        taxon_key: int,
        offset: int,
        limit: int,
        source: str | None = None,
        range_label: str | None = None,
    ) -> tuple[pd.DataFrame, int]: ...
    def occurrence_timeline(self, taxon_key: int) -> pd.DataFrame: ...

    # native range
    def get_native_range(self, taxon_key: int) -> NativeRange | None: ...
    def save_native_range(self, nr: NativeRange) -> NativeRange: ...

    # model lineage
    def add_model_version(self, mv: ModelVersionRecord) -> None: ...
    def list_model_versions(self, taxon_key: int) -> list[ModelVersionRecord]: ...
    def get_model_version(self, taxon_key: int, version: int) -> ModelVersionRecord | None: ...
    def update_model_severity(
        self, taxon_key: int, version: int, severity: dict[str, Any]
    ) -> None: ...
    def update_model_artifacts(
        self, taxon_key: int, version: int, artifacts: dict[str, Any]
    ) -> None: ...

    # jobs
    def create_job(self, job: Job) -> Job: ...
    def update_job(self, job_id: uuid.UUID, **fields: Any) -> Job: ...
    def get_job(self, job_id: uuid.UUID) -> Job | None: ...
    def list_jobs(self, taxon_key: int | None = None, limit: int = 20) -> list[Job]: ...
    def fail_stale_jobs(self, older_than: timedelta) -> list[int | None]: ...

    # per-species mutual exclusion (one pipeline / projection run at a time)
    def species_lock(
        self, taxon_key: int, on_wait: Callable[[], None] | None = None
    ) -> AbstractContextManager[None]: ...


def records_to_frame(records: Sequence[OccurrenceRecord]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in records])
    for b in BIOCLIM_BANDS:
        df[b] = float("nan")
    df["bioclim_version"] = None
    df["ingested_ts"] = utcnow()
    return df


class InMemoryRepository:
    """Thread-safe in-memory repository implementing the Repository protocol."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._species: dict[int, SpeciesRecord] = {}
        self._occ: dict[int, pd.DataFrame] = {}
        self._native: dict[int, NativeRange] = {}
        self._models: dict[int, dict[int, ModelVersionRecord]] = {}
        self._jobs: dict[uuid.UUID, Job] = {}
        self._species_locks: dict[int, threading.Lock] = {}

    # ---------------------------------------------------------- species
    def get_species(self, taxon_key: int) -> SpeciesRecord | None:
        with self._lock:
            rec = self._species.get(taxon_key)
            return copy.deepcopy(rec) if rec else None

    def list_species(self) -> list[SpeciesRecord]:
        with self._lock:
            return [copy.deepcopy(r) for r in self._species.values()]

    def create_species(self, rec: SpeciesRecord) -> SpeciesRecord:
        with self._lock:
            if rec.taxon_key in self._species:
                raise ValueError(f"Species {rec.taxon_key} already registered")
            self._species[rec.taxon_key] = copy.deepcopy(rec)
            self._occ[rec.taxon_key] = pd.DataFrame(columns=list(OCCURRENCE_COLUMNS))
            return copy.deepcopy(rec)

    def update_species(self, taxon_key: int, **fields: Any) -> SpeciesRecord:
        with self._lock:
            rec = self._species.get(taxon_key)
            if rec is None:
                raise NotFoundError(taxon_key)
            rec = replace(rec, **fields, updated_ts=utcnow())
            self._species[taxon_key] = rec
            return copy.deepcopy(rec)

    # ------------------------------------------------------ occurrences
    def append_occurrences(self, records: Sequence[OccurrenceRecord]) -> int:
        if not records:
            return 0
        with self._lock:
            inserted = 0
            new = records_to_frame(records)
            for key, grp in new.groupby("taxon_key"):
                key = int(key)
                if key not in self._species:
                    raise NotFoundError(key)
                cur = self._occ[key]
                seen = set(zip(cur["source"], cur["record_id"].astype("int64"), strict=True))
                grp = grp.drop_duplicates(["source", "record_id"])
                mask = [
                    (s, int(r)) not in seen
                    for s, r in zip(grp["source"], grp["record_id"], strict=True)
                ]
                add = grp.loc[mask, list(OCCURRENCE_COLUMNS)]
                if len(add):
                    self._occ[key] = (
                        add.copy() if cur.empty else pd.concat([cur, add], ignore_index=True)
                    )
                inserted += len(add)
            return inserted

    def load_occurrences(self, taxon_key: int) -> pd.DataFrame:
        with self._lock:
            return self._occ.get(taxon_key, pd.DataFrame(columns=list(OCCURRENCE_COLUMNS))).copy()

    def occurrence_counts(self, taxon_key: int) -> dict[str, int]:
        df = self.load_occurrences(taxon_key)
        vc = df["range_label"].value_counts() if not df.empty else pd.Series(dtype=int)
        return {
            "total": len(df),
            "native": int(vc.get("native", 0)),
            "introduced": int(vc.get("introduced", 0)),
            "unknown": int(vc.get("unknown", 0)),
        }

    def source_counts(self, taxon_key: int) -> dict[str, int]:
        df = self.load_occurrences(taxon_key)
        if df.empty:
            return {"gbif": 0, "gbif_from_inaturalist": 0, "inaturalist_direct": 0}
        gbif = df["source"] == "GBIF"
        return {
            "gbif": int(gbif.sum()),
            "gbif_from_inaturalist": int((gbif & df["inat_crossref_id"].notna()).sum()),
            "inaturalist_direct": int((df["source"] == "iNaturalist").sum()),
        }

    def inat_crossref_ids(self, taxon_key: int) -> set[int]:
        df = self.load_occurrences(taxon_key)
        return {int(v) for v in df["inat_crossref_id"].dropna()} if not df.empty else set()

    def fill_features(
        self, taxon_key: int, keys: pd.DataFrame, features: pd.DataFrame, bioclim_version: str
    ) -> int:
        with self._lock:
            df = self._occ[taxon_key]
            idx = pd.MultiIndex.from_frame(
                df[["source", "record_id"]].astype({"record_id": "int64"})
            )
            key_idx = pd.MultiIndex.from_frame(
                keys[["source", "record_id"]].astype({"record_id": "int64"})
            )
            pos = idx.get_indexer(key_idx)
            n = 0
            for i, p in enumerate(pos):
                if p < 0 or pd.notna(df.at[p, "bio1"]):
                    continue
                for c in features.columns:
                    df.at[p, c] = features.iloc[i][c]
                df.at[p, "bioclim_version"] = bioclim_version
                n += 1
            return n

    def page_occurrences(
        self,
        taxon_key: int,
        offset: int,
        limit: int,
        source: str | None = None,
        range_label: str | None = None,
    ) -> tuple[pd.DataFrame, int]:
        df = self.load_occurrences(taxon_key)
        if source:
            df = df[df["source"] == source]
        if range_label:
            df = df[df["range_label"] == range_label]
        df = df.sort_values(["ingested_ts", "record_id"], ascending=[False, True])
        return df.iloc[offset : offset + limit].reset_index(drop=True), len(df)

    def occurrence_timeline(self, taxon_key: int) -> pd.DataFrame:
        df = self.load_occurrences(taxon_key)
        if df.empty:
            return pd.DataFrame(columns=["year", "range_label", "n"])
        years = pd.to_datetime(df["event_date"], errors="coerce").dt.year
        out = (
            df.assign(year=years)
            .dropna(subset=["year"])
            .groupby(["year", "range_label"])
            .size()
            .reset_index(name="n")
        )
        out["year"] = out["year"].astype(int)
        return out

    # ------------------------------------------------------ native range
    def get_native_range(self, taxon_key: int) -> NativeRange | None:
        with self._lock:
            nr = self._native.get(taxon_key)
            return copy.deepcopy(nr) if nr else None

    def save_native_range(self, nr: NativeRange) -> NativeRange:
        with self._lock:
            if nr.taxon_key not in self._species:
                raise NotFoundError(nr.taxon_key)
            self._native[nr.taxon_key] = copy.deepcopy(nr)
            return copy.deepcopy(nr)

    # --------------------------------------------------- model lineage
    def add_model_version(self, mv: ModelVersionRecord) -> None:
        with self._lock:
            versions = self._models.setdefault(mv.taxon_key, {})
            if mv.model_version in versions:
                raise ValueError("model_version already exists")
            versions[mv.model_version] = copy.deepcopy(mv)

    def list_model_versions(self, taxon_key: int) -> list[ModelVersionRecord]:
        with self._lock:
            vs = self._models.get(taxon_key, {})
            return [copy.deepcopy(vs[k]) for k in sorted(vs, reverse=True)]

    def get_model_version(self, taxon_key: int, version: int) -> ModelVersionRecord | None:
        with self._lock:
            mv = self._models.get(taxon_key, {}).get(version)
            return copy.deepcopy(mv) if mv else None

    def update_model_severity(self, taxon_key: int, version: int, severity: dict[str, Any]) -> None:
        with self._lock:
            self._models[taxon_key][version].severity = copy.deepcopy(severity)

    def update_model_artifacts(
        self, taxon_key: int, version: int, artifacts: dict[str, Any]
    ) -> None:
        """Add derived artifacts (e.g. new scenario projections); the model itself is unchanged."""
        with self._lock:
            self._models[taxon_key][version].artifacts = copy.deepcopy(artifacts)

    # --------------------------------------------------------- jobs
    def create_job(self, job: Job) -> Job:
        with self._lock:
            self._jobs[job.job_id] = copy.deepcopy(job)
            return copy.deepcopy(job)

    def update_job(self, job_id: uuid.UUID, **fields: Any) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise NotFoundError(job_id)
            job = replace(job, **fields, updated_ts=utcnow())
            self._jobs[job_id] = job
            return copy.deepcopy(job)

    def get_job(self, job_id: uuid.UUID) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def list_jobs(self, taxon_key: int | None = None, limit: int = 20) -> list[Job]:
        with self._lock:
            jobs = [j for j in self._jobs.values() if taxon_key is None or j.taxon_key == taxon_key]
            jobs.sort(key=lambda j: j.created_ts, reverse=True)
            return [copy.deepcopy(j) for j in jobs[:limit]]

    def fail_stale_jobs(self, older_than: timedelta) -> list[int | None]:
        """Fail active jobs silent for longer than `older_than`; returns their taxon keys."""
        with self._lock:
            cutoff = utcnow() - older_than
            reaped: list[int | None] = []
            for jid, job in self._jobs.items():
                active = job.status in (JobStatus.QUEUED, JobStatus.RUNNING)
                if active and job.updated_ts < cutoff:
                    self._jobs[jid] = replace(
                        job, status=JobStatus.FAILED, message=STALE_JOB_MESSAGE, updated_ts=utcnow()
                    )
                    reaped.append(job.taxon_key)
            return reaped

    @contextmanager
    def species_lock(
        self, taxon_key: int, on_wait: Callable[[], None] | None = None
    ) -> Iterator[None]:
        with self._lock:
            lock = self._species_locks.setdefault(taxon_key, threading.Lock())
        if not lock.acquire(blocking=False):
            if on_wait is not None:
                on_wait()
            lock.acquire()
        try:
            yield
        finally:
            lock.release()
