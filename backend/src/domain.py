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
# Binary change maps (U6): after + 2 × before, each 1 when suitable.
CHANGE_LABELS: dict[int, str] = {0: "stable unsuitable", 1: "gain", 2: "loss", 3: "stable suitable"}

RangeLabel = Literal["native", "introduced", "unknown"]
OccurrenceSource = Literal["GBIF", "iNaturalist"]
# Training rows from an uploaded dataset carry source "upload:<dataset_id>" (never stored in
# `occurrences`; see the reference datasets below).
UPLOAD_SOURCE_PREFIX = "upload:"
ModelType = Literal["A", "B"]
OccurrenceStatus = Literal["present", "absent"]
ReferenceKind = Literal["gbif_absences", "upload"]
# "evaluation": independent test data, never trained on (GBIF absences are always this).
# "training": an upload whose presences join the training set (after the same data-quality
# tests); its absences, which a presence-background model cannot use, still evaluate it.
ReferenceRole = Literal["evaluation", "training"]
# Expert annotations of single records (U11), append-only; the latest per record and kind wins:
#   range_label: override the polygon-derived label (value native | introduced | unknown)
#   exclude:     "true" = suspicious, left out of training ("false" withdraws it)
#   cultivated:  "true" = planted / cultivated, left out of training
#   verified:    "true" = keep despite a data-quality exclusion
AnnotationKind = Literal["range_label", "exclude", "cultivated", "verified"]
# Early-detection alerts (U10): new records outside the suitability envelope or far from
# every known record. B10 classifies each record by the target group's prior effort there.
AlertKind = Literal["outside_envelope", "outside_known_range"]
AlertChannel = Literal["webhook", "email"]
Role = Literal["viewer", "analyst", "curator"]


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
    CROSSCHECK = "crosscheck"
    HIRES = "hires"
    EVALUATION = "evaluation"
    BENCHMARK = "benchmark"
    BATCH = "batch"
    TARGET_GROUPS = "target_groups"
    SHARED_BIAS = "shared_bias"


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
    # GBIF interpretation issue flags (e.g. COUNTRY_COORDINATE_MISMATCH). Empty for records
    # without any (and for iNaturalist, which has none); see features/quality.py.
    issues: tuple[str, ...] = ()
    # Free-text and identification details used by the data-quality tests: GBIF locality,
    # habitat, occurrenceRemarks, recordedBy, recordNumber, institutionCode; iNaturalist
    # place_guess, description and identification counts. None = ingested before details
    # were kept (filled once by the next fetch, like `issues`); {} = none given.
    details: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ReferenceRecord:
    """One record of a reference dataset: an explicit GBIF absence or a row of an uploaded
    survey. Stored apart from `occurrences` (append-only); only presences of a dataset with
    role "training" are used for training."""

    taxon_key: int
    dataset_id: str
    record_id: int
    longitude: float
    latitude: float
    occurrence_status: OccurrenceStatus = "present"
    event_date: date | None = None
    coordinate_uncertainty_m: float | None = None
    basis_of_record: str | None = None
    country_code: str | None = None
    sampling_protocol: str | None = None
    source_dataset_key: str | None = None


@dataclass(slots=True)
class ReferenceDataset:
    taxon_key: int
    dataset_id: str
    kind: ReferenceKind
    name: str
    role: ReferenceRole = "evaluation"
    citation: str | None = None
    license: str | None = None
    note: str | None = None
    file_name: str | None = None
    file_sha256: str | None = None
    # How the upload was parsed (columns used, rows skipped and why, CRS).
    parse_report: dict[str, Any] | None = None
    # Independence check against the GBIF/iNaturalist occurrence store.
    overlap: dict[str, Any] | None = None
    n_present: int = 0
    n_absent: int = 0
    # Incremental-fetch watermark (GBIF absences only).
    last_fetch_ts: datetime | None = None
    created_ts: datetime = field(default_factory=utcnow)
    updated_ts: datetime = field(default_factory=utcnow)


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
    # Regional model (U2): {"bbox": [west, south, east, north], "resolution": "30s" | None}.
    # None = the global model at the bioclim version's own resolution.
    study_region: dict[str, Any] | None = None
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


@dataclass(slots=True)
class RecordAnnotation:
    """One expert annotation of one occurrence record (append-only; never edits the record)."""

    taxon_key: int
    source: OccurrenceSource
    record_id: int
    kind: AnnotationKind
    value: str
    note: str | None = None
    author: str = "anonymous"
    annotation_id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_ts: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class Alert:
    taxon_key: int
    kind: AlertKind
    n_records: int
    # [{source, record_id, longitude, latitude, event_date, suitability?, distance_km?,
    #   effort_class: expansion | discovery | unknown}]
    records: list[dict[str, Any]]
    summary: str
    model_version: int | None = None
    # B10 counts: expansion (previously well sampled) / discovery (previously unsampled)
    effort_classes: dict[str, int] = field(default_factory=dict)
    # Delivery outcome per subscription (updated after sending)
    deliveries: list[dict[str, Any]] = field(default_factory=list)
    alert_id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_ts: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class SpeciesListEntry:
    """One species of a list: as given (name or key) and its GBIF backbone resolution."""

    query: str
    taxon_key: int | None = None
    scientific_name: str | None = None
    match_type: str | None = None  # EXACT / FUZZY / HIGHERRANK / KEY / NONE
    confidence: int | None = None


@dataclass(slots=True)
class SpeciesList:
    """A watch list (e.g. the EU list of invasive alien species of concern) for batch runs
    and registry filtering (U9). Edited in place; it holds no occurrence data."""

    name: str
    entries: list[SpeciesListEntry] = field(default_factory=list)
    description: str | None = None
    source: str | None = None  # citation / URL of the official list
    created_by: str = "anonymous"
    list_id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_ts: datetime = field(default_factory=utcnow)
    updated_ts: datetime = field(default_factory=utcnow)

    @property
    def taxon_keys(self) -> list[int]:
        return [e.taxon_key for e in self.entries if e.taxon_key is not None]


@dataclass(slots=True)
class AlertSubscription:
    channel: AlertChannel
    target: str  # webhook URL or e-mail address
    taxon_key: int | None = None  # None = every species
    kinds: list[str] = field(default_factory=lambda: ["outside_envelope", "outside_known_range"])
    created_by: str = "anonymous"
    subscription_id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_ts: datetime = field(default_factory=utcnow)
