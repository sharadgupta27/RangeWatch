"""Domain types shared by connectors, persistence, orchestration and the API layer."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal

BIOCLIM_BANDS: tuple[str, ...] = tuple(f"bio{i}" for i in range(1, 20))
BIOCLIM_DESCRIPTIONS: dict[str, str] = {
    "bio1": "Annual mean temperature",
    "bio2": "Mean diurnal range",
    "bio3": "Isothermality",
    "bio4": "Temperature seasonality",
    "bio5": "Max temperature of warmest month",
    "bio6": "Min temperature of coldest month",
    "bio7": "Temperature annual range",
    "bio8": "Mean temperature of wettest quarter",
    "bio9": "Mean temperature of driest quarter",
    "bio10": "Mean temperature of warmest quarter",
    "bio11": "Mean temperature of coldest quarter",
    "bio12": "Annual precipitation",
    "bio13": "Precipitation of wettest month",
    "bio14": "Precipitation of driest month",
    "bio15": "Precipitation seasonality",
    "bio16": "Precipitation of wettest quarter",
    "bio17": "Precipitation of driest quarter",
    "bio18": "Precipitation of warmest quarter",
    "bio19": "Precipitation of coldest quarter",
}

ZONE_LABELS: dict[int, str] = {
    0: "not suitable",
    1: "native range (suitable)",
    2: "established outside native range",
    3: "candidate invasion / expansion zone",
}

RangeLabel = Literal["native", "introduced", "unknown"]
OccurrenceSource = Literal["GBIF", "iNaturalist"]
ModelType = Literal["A", "B"]


def utcnow() -> datetime:
    return datetime.now(UTC)


class SpeciesStatus(StrEnum):
    """Lifecycle state surfaced to the frontend as status badges."""

    REGISTERED = "registered"
    INGESTING = "ingesting"
    AWAITING_NATIVE_RANGE_REVIEW = "awaiting_native_range_review"
    TRAINING = "training"
    UP_TO_DATE = "up_to_date"
    MINOR_UPDATE_PENDING = "minor_update_pending"
    RETRAIN_PENDING = "retrain_pending"
    FAILED = "failed"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobKind(StrEnum):
    PIPELINE = "pipeline"
    BULLETIN = "bulletin"
    SCENARIOS = "scenarios"
    VALIDATION = "validation"


@dataclass(frozen=True, slots=True)
class OccurrenceRecord:
    taxon_key: int
    source: OccurrenceSource
    record_id: int
    longitude: float
    latitude: float
    event_date: date | None = None
    coordinate_uncertainty_m: float | None = None
    basis_of_record: str | None = None
    country_code: str | None = None
    range_label: RangeLabel = "unknown"
    inat_crossref_id: int | None = None


@dataclass(slots=True)
class SpeciesRecord:
    taxon_key: int
    scientific_name: str
    canonical_name: str | None = None
    common_name: str | None = None
    inat_taxon_id: int | None = None
    thumbnail_url: str | None = None
    status: SpeciesStatus = SpeciesStatus.REGISTERED
    last_gbif_fetch_ts: datetime | None = None
    last_inat_fetch_ts: datetime | None = None
    last_gbif_download_key: str | None = None
    last_gbif_download_doi: str | None = None
    n_occurrences_native: int = 0
    n_occurrences_total: int = 0
    n_pending_records: int = 0
    bioclim_version_used: str | None = None
    model_version: int = 0
    model_path: str | None = None
    model_trained_ts: datetime | None = None
    model_metrics: dict[str, Any] | None = None
    retrain_needed: bool = False
    severity_config: dict[str, Any] | None = None
    last_error: str | None = None
    created_ts: datetime = field(default_factory=utcnow)
    updated_ts: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class NativeRange:
    taxon_key: int
    geometry: dict[str, Any]  # GeoJSON (Multi)Polygon, EPSG:4326
    status: Literal["draft", "confirmed"]
    source: str
    note: str | None = None
    updated_ts: datetime = field(default_factory=utcnow)
    confirmed_ts: datetime | None = None


@dataclass(slots=True)
class ModelVersionRecord:
    taxon_key: int
    model_version: int
    model_type: ModelType
    trigger: str
    model_path: str
    artifacts: dict[str, Any]
    metrics: dict[str, Any]
    severity: dict[str, Any] | None = None
    mlflow_run_id: str | None = None
    trained_ts: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class Job:
    kind: JobKind
    taxon_key: int | None
    job_id: uuid.UUID = field(default_factory=uuid.uuid4)
    status: JobStatus = JobStatus.QUEUED
    stage: str | None = None
    progress: float = 0.0
    message: str | None = None
    result: dict[str, Any] | None = None
    created_ts: datetime = field(default_factory=utcnow)
    updated_ts: datetime = field(default_factory=utcnow)
