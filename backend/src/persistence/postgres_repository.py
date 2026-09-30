"""PostgreSQL + PostGIS implementation of the Repository protocol."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import fields as dc_fields
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from src.domain import (
    BIOCLIM_BANDS,
    Job,
    JobKind,
    JobStatus,
    ModelVersionRecord,
    NativeRange,
    OccurrenceRecord,
    SpeciesRecord,
    SpeciesStatus,
)
from src.persistence.repository import STALE_JOB_MESSAGE, NotFoundError

_SPECIES_COLS = [f.name for f in dc_fields(SpeciesRecord)]
_JSON_SPECIES_COLS = {"model_metrics", "severity_config"}
_JOB_COLS = [f.name for f in dc_fields(Job)]
_BIO_SELECT = ", ".join(BIOCLIM_BANDS)


def _json_default(o: Any) -> Any:
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, uuid.UUID):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def _dumps(v: Any) -> str | None:
    return None if v is None else json.dumps(v, default=_json_default)


# Advisory-lock key space for per-species runs (first int of the two-int lock key).
_SPECIES_LOCK_NAMESPACE = 0x5D4D
_SPECIES_LOCK_POLL_S = 2.0


class PostgresRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    @classmethod
    def from_url(cls, url: str) -> PostgresRepository:
        return cls(create_engine(url, pool_pre_ping=True, future=True))

    # ---------------------------------------------------------- species
    def _row_to_species(self, row: Any) -> SpeciesRecord:
        d = dict(row._mapping)
        d["status"] = SpeciesStatus(d["status"])
        return SpeciesRecord(**{k: d[k] for k in _SPECIES_COLS if k in d})

    def get_species(self, taxon_key: int) -> SpeciesRecord | None:
        with self.engine.connect() as c:
            row = c.execute(
                text("SELECT * FROM species_registry WHERE taxon_key = :k"), {"k": taxon_key}
            ).first()
        return self._row_to_species(row) if row else None

    def list_species(self) -> list[SpeciesRecord]:
        with self.engine.connect() as c:
            rows = c.execute(text("SELECT * FROM species_registry ORDER BY updated_ts DESC"))
            return [self._row_to_species(r) for r in rows]

    def create_species(self, rec: SpeciesRecord) -> SpeciesRecord:
        key = int(rec.taxon_key)
        params = {k: getattr(rec, k) for k in _SPECIES_COLS}
        params["status"] = str(rec.status)
        for k in _JSON_SPECIES_COLS:
            params[k] = _dumps(params[k])
        cols = ", ".join(_SPECIES_COLS)
        vals = ", ".join(
            f"CAST(:{k} AS jsonb)" if k in _JSON_SPECIES_COLS else f":{k}" for k in _SPECIES_COLS
        )
        with self.engine.begin() as c:
            c.execute(text(f"INSERT INTO species_registry ({cols}) VALUES ({vals})"), params)
            # One occurrence partition per species (key is an int, safe to inline).
            c.execute(
                text(
                    f"CREATE TABLE IF NOT EXISTS occurrences_{key} "
                    f"PARTITION OF occurrences FOR VALUES IN ({key})"
                )
            )
        return rec

    def update_species(self, taxon_key: int, **fields: Any) -> SpeciesRecord:
        unknown = set(fields) - set(_SPECIES_COLS)
        if unknown:
            raise ValueError(f"Unknown species fields: {unknown}")
        sets, params = [], {"k": taxon_key}
        for k, v in fields.items():
            if k in _JSON_SPECIES_COLS:
                sets.append(f"{k} = CAST(:{k} AS jsonb)")
                params[k] = _dumps(v)
            else:
                sets.append(f"{k} = :{k}")
                params[k] = str(v) if isinstance(v, SpeciesStatus) else v
        sets.append("updated_ts = now()")
        with self.engine.begin() as c:
            res = c.execute(
                text(f"UPDATE species_registry SET {', '.join(sets)} WHERE taxon_key = :k"),
                params,
            )
            if res.rowcount == 0:
                raise NotFoundError(taxon_key)
        rec = self.get_species(taxon_key)
        assert rec is not None
        return rec

    # ------------------------------------------------------ occurrences
    def append_occurrences(self, records: Sequence[OccurrenceRecord]) -> int:
        if not records:
            return 0
        rows = [
            {
                "taxon_key": r.taxon_key,
                "source": r.source,
                "record_id": r.record_id,
                "inat_crossref_id": r.inat_crossref_id,
                "lon": r.longitude,
                "lat": r.latitude,
                "event_date": r.event_date,
                "unc": r.coordinate_uncertainty_m,
                "bor": r.basis_of_record,
                "cc": r.country_code,
                "label": r.range_label,
            }
            for r in records
        ]
        sql = text("""
            INSERT INTO occurrences (taxon_key, source, record_id, inat_crossref_id, geom,
                event_date, coordinate_uncertainty_m, basis_of_record, country_code, range_label)
            VALUES (:taxon_key, :source, :record_id, :inat_crossref_id,
                ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), :event_date, :unc, :bor, :cc, :label)
            ON CONFLICT DO NOTHING
            """)
        keys = {r.taxon_key for r in records}
        with self.engine.begin() as c:
            before = sum(self._count(c, k) for k in keys)
            for i in range(0, len(rows), 5000):
                c.execute(sql, rows[i : i + 5000])
            after = sum(self._count(c, k) for k in keys)
        return after - before

    @staticmethod
    def _count(conn: Any, taxon_key: int) -> int:
        return int(
            conn.execute(
                text("SELECT count(*) FROM occurrences WHERE taxon_key = :k"), {"k": taxon_key}
            ).scalar_one()
        )

    def load_occurrences(self, taxon_key: int) -> pd.DataFrame:
        sql = text(f"""
            SELECT source, record_id, ST_X(geom) AS longitude, ST_Y(geom) AS latitude,
                   event_date, coordinate_uncertainty_m, basis_of_record, country_code,
                   range_label, inat_crossref_id, ingested_ts, {_BIO_SELECT}, bioclim_version
            FROM occurrences WHERE taxon_key = :k
            """)
        with self.engine.connect() as c:
            return pd.read_sql(sql, c, params={"k": taxon_key})

    def occurrence_counts(self, taxon_key: int) -> dict[str, int]:
        sql = text("""
            SELECT count(*) AS total,
                   count(*) FILTER (WHERE range_label = 'native') AS native,
                   count(*) FILTER (WHERE range_label = 'introduced') AS introduced,
                   count(*) FILTER (WHERE range_label = 'unknown') AS unknown
            FROM occurrences WHERE taxon_key = :k
            """)
        with self.engine.connect() as c:
            row = c.execute(sql, {"k": taxon_key}).one()
        return {k: int(v) for k, v in row._mapping.items()}

    def source_counts(self, taxon_key: int) -> dict[str, int]:
        sql = text("""
            SELECT count(*) FILTER (WHERE source = 'GBIF') AS gbif,
                   count(*) FILTER (WHERE source = 'GBIF' AND inat_crossref_id IS NOT NULL)
                       AS gbif_from_inaturalist,
                   count(*) FILTER (WHERE source = 'iNaturalist') AS inaturalist_direct
            FROM occurrences WHERE taxon_key = :k
            """)
        with self.engine.connect() as c:
            row = c.execute(sql, {"k": taxon_key}).one()
        return {k: int(v) for k, v in row._mapping.items()}

    def inat_crossref_ids(self, taxon_key: int) -> set[int]:
        with self.engine.connect() as c:
            rows = c.execute(
                text(
                    "SELECT inat_crossref_id FROM occurrences "
                    "WHERE taxon_key = :k AND inat_crossref_id IS NOT NULL"
                ),
                {"k": taxon_key},
            )
            return {int(r[0]) for r in rows}

    def fill_features(
        self, taxon_key: int, keys: pd.DataFrame, features: pd.DataFrame, bioclim_version: str
    ) -> int:
        sets = ", ".join(f"{b} = :{b}" for b in features.columns)
        sql = text(f"""
            UPDATE occurrences SET {sets}, bioclim_version = :v
            WHERE taxon_key = :k AND source = :source AND record_id = :record_id
              AND bio1 IS NULL
            """)
        payload = []
        for (_, key), (_, feat) in zip(keys.iterrows(), features.iterrows(), strict=True):
            row = {b: (None if pd.isna(feat[b]) else float(feat[b])) for b in features.columns}
            row.update(
                k=taxon_key,
                v=bioclim_version,
                source=key["source"],
                record_id=int(key["record_id"]),
            )
            payload.append(row)
        if not payload:
            return 0
        with self.engine.begin() as c:
            res = c.execute(sql, payload)
            return int(res.rowcount or 0)

    def page_occurrences(
        self,
        taxon_key: int,
        offset: int,
        limit: int,
        source: str | None = None,
        range_label: str | None = None,
    ) -> tuple[pd.DataFrame, int]:
        where = ["taxon_key = :k"]
        params: dict[str, Any] = {"k": taxon_key, "off": offset, "lim": limit}
        if source:
            where.append("source = :source")
            params["source"] = source
        if range_label:
            where.append("range_label = :label")
            params["label"] = range_label
        w = " AND ".join(where)
        with self.engine.connect() as c:
            total = int(
                c.execute(text(f"SELECT count(*) FROM occurrences WHERE {w}"), params).scalar_one()
            )
            df = pd.read_sql(
                text(f"""
                    SELECT source, record_id, ST_X(geom) AS longitude, ST_Y(geom) AS latitude,
                           event_date, coordinate_uncertainty_m, basis_of_record, country_code,
                           range_label, ingested_ts
                    FROM occurrences WHERE {w}
                    ORDER BY ingested_ts DESC, record_id
                    OFFSET :off LIMIT :lim
                    """),
                c,
                params=params,
            )
        return df, total

    def occurrence_timeline(self, taxon_key: int) -> pd.DataFrame:
        sql = text("""
            SELECT EXTRACT(YEAR FROM event_date)::int AS year, range_label, count(*)::int AS n
            FROM occurrences WHERE taxon_key = :k AND event_date IS NOT NULL
            GROUP BY 1, 2 ORDER BY 1
            """)
        with self.engine.connect() as c:
            return pd.read_sql(sql, c, params={"k": taxon_key})

    # ------------------------------------------------------ native range
    def get_native_range(self, taxon_key: int) -> NativeRange | None:
        sql = text("""
            SELECT taxon_key, ST_AsGeoJSON(geom)::jsonb AS geometry, status, source, note,
                   updated_ts, confirmed_ts
            FROM native_ranges WHERE taxon_key = :k
            """)
        with self.engine.connect() as c:
            row = c.execute(sql, {"k": taxon_key}).first()
        return NativeRange(**dict(row._mapping)) if row else None

    def save_native_range(self, nr: NativeRange) -> NativeRange:
        sql = text("""
            INSERT INTO native_ranges (taxon_key, geom, status, source, note, updated_ts,
                                       confirmed_ts)
            VALUES (:k, ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(:g), 4326)), :status, :source,
                    :note, now(), :confirmed_ts)
            ON CONFLICT (taxon_key) DO UPDATE SET
                geom = EXCLUDED.geom, status = EXCLUDED.status, source = EXCLUDED.source,
                note = EXCLUDED.note, updated_ts = now(), confirmed_ts = EXCLUDED.confirmed_ts
            """)
        with self.engine.begin() as c:
            c.execute(
                sql,
                {
                    "k": nr.taxon_key,
                    "g": json.dumps(nr.geometry),
                    "status": nr.status,
                    "source": nr.source,
                    "note": nr.note,
                    "confirmed_ts": nr.confirmed_ts,
                },
            )
        saved = self.get_native_range(nr.taxon_key)
        assert saved is not None
        return saved

    # --------------------------------------------------- model lineage
    def add_model_version(self, mv: ModelVersionRecord) -> None:
        sql = text("""
            INSERT INTO model_versions (taxon_key, model_version, model_type, trained_ts,
                trigger, model_path, artifacts, metrics, severity, mlflow_run_id)
            VALUES (:taxon_key, :model_version, :model_type, :trained_ts, :trigger, :model_path,
                CAST(:artifacts AS jsonb), CAST(:metrics AS jsonb), CAST(:severity AS jsonb),
                :mlflow_run_id)
            """)
        with self.engine.begin() as c:
            c.execute(
                sql,
                {
                    "taxon_key": mv.taxon_key,
                    "model_version": mv.model_version,
                    "model_type": mv.model_type,
                    "trained_ts": mv.trained_ts,
                    "trigger": mv.trigger,
                    "model_path": mv.model_path,
                    "artifacts": _dumps(mv.artifacts),
                    "metrics": _dumps(mv.metrics),
                    "severity": _dumps(mv.severity),
                    "mlflow_run_id": mv.mlflow_run_id,
                },
            )

    def list_model_versions(self, taxon_key: int) -> list[ModelVersionRecord]:
        with self.engine.connect() as c:
            rows = c.execute(
                text(
                    "SELECT * FROM model_versions WHERE taxon_key = :k "
                    "ORDER BY model_version DESC"
                ),
                {"k": taxon_key},
            )
            return [ModelVersionRecord(**dict(r._mapping)) for r in rows]

    def get_model_version(self, taxon_key: int, version: int) -> ModelVersionRecord | None:
        with self.engine.connect() as c:
            row = c.execute(
                text("SELECT * FROM model_versions WHERE taxon_key = :k AND model_version = :v"),
                {"k": taxon_key, "v": version},
            ).first()
        return ModelVersionRecord(**dict(row._mapping)) if row else None

    def update_model_artifacts(
        self, taxon_key: int, version: int, artifacts: dict[str, Any]
    ) -> None:
        with self.engine.begin() as c:
            c.execute(
                text(
                    "UPDATE model_versions SET artifacts = CAST(:a AS jsonb) "
                    "WHERE taxon_key = :k AND model_version = :v"
                ),
                {"a": _dumps(artifacts), "k": taxon_key, "v": version},
            )

    def update_model_severity(self, taxon_key: int, version: int, severity: dict[str, Any]) -> None:
        with self.engine.begin() as c:
            c.execute(
                text(
                    "UPDATE model_versions SET severity = CAST(:s AS jsonb) "
                    "WHERE taxon_key = :k AND model_version = :v"
                ),
                {"s": _dumps(severity), "k": taxon_key, "v": version},
            )

    # --------------------------------------------------------- jobs
    def _row_to_job(self, row: Any) -> Job:
        d = dict(row._mapping)
        d["kind"] = JobKind(d["kind"])
        d["status"] = JobStatus(d["status"])
        return Job(**{k: d[k] for k in _JOB_COLS})

    def create_job(self, job: Job) -> Job:
        with self.engine.begin() as c:
            c.execute(
                text("""
                    INSERT INTO jobs (job_id, taxon_key, kind, status, stage, progress, message,
                                      result, created_ts, updated_ts)
                    VALUES (:job_id, :taxon_key, :kind, :status, :stage, :progress, :message,
                            CAST(:result AS jsonb), :created_ts, :updated_ts)
                    """),
                {
                    "job_id": job.job_id,
                    "taxon_key": job.taxon_key,
                    "kind": str(job.kind),
                    "status": str(job.status),
                    "stage": job.stage,
                    "progress": job.progress,
                    "message": job.message,
                    "result": _dumps(job.result),
                    "created_ts": job.created_ts,
                    "updated_ts": job.updated_ts,
                },
            )
        return job

    def update_job(self, job_id: uuid.UUID, **fields: Any) -> Job:
        sets, params = [], {"id": job_id}
        for k, v in fields.items():
            if k not in _JOB_COLS:
                raise ValueError(f"Unknown job field {k}")
            if k == "result":
                sets.append("result = CAST(:result AS jsonb)")
                params[k] = _dumps(v)
            else:
                sets.append(f"{k} = :{k}")
                params[k] = str(v) if isinstance(v, (JobStatus, JobKind)) else v
        sets.append("updated_ts = now()")
        with self.engine.begin() as c:
            c.execute(text(f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = :id"), params)
        job = self.get_job(job_id)
        if job is None:
            raise NotFoundError(job_id)
        return job

    def get_job(self, job_id: uuid.UUID) -> Job | None:
        with self.engine.connect() as c:
            row = c.execute(text("SELECT * FROM jobs WHERE job_id = :id"), {"id": job_id}).first()
        return self._row_to_job(row) if row else None

    def list_jobs(self, taxon_key: int | None = None, limit: int = 20) -> list[Job]:
        where = "WHERE taxon_key = :k" if taxon_key is not None else ""
        with self.engine.connect() as c:
            rows = c.execute(
                text(f"SELECT * FROM jobs {where} ORDER BY created_ts DESC LIMIT :lim"),
                {"k": taxon_key, "lim": limit},
            )
            return [self._row_to_job(r) for r in rows]

    def fail_stale_jobs(self, older_than: timedelta) -> list[int | None]:
        with self.engine.begin() as c:
            res = c.execute(
                text("""
                    UPDATE jobs SET status = 'failed', message = :msg, updated_ts = now()
                    WHERE status IN ('queued', 'running') AND updated_ts < now() - :age
                    RETURNING taxon_key
                    """),
                {"msg": STALE_JOB_MESSAGE, "age": older_than},
            )
            return [r[0] for r in res]

    @contextmanager
    def species_lock(
        self, taxon_key: int, on_wait: Callable[[], None] | None = None
    ) -> Iterator[None]:
        """Session-level advisory lock on (namespace, taxon_key), held on a dedicated
        connection for the whole run. Postgres drops it when that connection closes, so a
        killed worker cannot leave a species locked."""
        conn = self.engine.connect()
        try:
            waited = False
            while not conn.execute(
                text("SELECT pg_try_advisory_lock(:ns, :k)"),
                {"ns": _SPECIES_LOCK_NAMESPACE, "k": taxon_key},
            ).scalar():
                conn.rollback()
                if not waited and on_wait is not None:
                    on_wait()
                waited = True
                time.sleep(_SPECIES_LOCK_POLL_S)
            conn.commit()
            try:
                yield
            finally:
                conn.execute(
                    text("SELECT pg_advisory_unlock(:ns, :k)"),
                    {"ns": _SPECIES_LOCK_NAMESPACE, "k": taxon_key},
                )
                conn.commit()
        finally:
            conn.close()
