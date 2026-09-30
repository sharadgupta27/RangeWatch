"""Pydantic API contract. The React client's types are generated from this OpenAPI schema
(openapi-typescript) — keep every response model explicit and accurate."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.domain import JobKind, JobStatus, SpeciesStatus

RangeLabelT = Literal["native", "introduced", "unknown"]
SourceT = Literal["GBIF", "iNaturalist"]
ModelTypeT = Literal["A", "B"]
ConfidenceT = Literal["low", "moderate", "high"]


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------ species
class TaxonSearchResult(ApiModel):
    taxon_key: int
    scientific_name: str
    canonical_name: str | None = None
    rank: str | None = None
    breadcrumb: list[str] = Field(default_factory=list, description="Kingdom → genus")
    common_name: str | None = None
    thumbnail_url: str | None = None
    inat_taxon_id: int | None = None
    in_registry: bool = False
    status: SpeciesStatus | None = None


class SpeciesSummary(ApiModel):
    taxon_key: int
    scientific_name: str
    common_name: str | None = None
    thumbnail_url: str | None = None
    status: SpeciesStatus
    n_occurrences_total: int
    n_occurrences_native: int
    n_pending_records: int
    model_version: int
    model_trained_ts: datetime | None = None
    last_gbif_fetch_ts: datetime | None = None
    retrain_needed: bool
    severity_score: float | None = Field(None, description="0–100, current model")
    model_type: ModelTypeT | None = None


class SeverityWeights(ApiModel):
    suitability: float = Field(0.35, ge=0)
    climate_analogy: float = Field(0.25, ge=0)
    spread_rate: float = Field(0.20, ge=0)
    ecological_impact_prior: float = Field(0.20, ge=0)


class SeverityConfigModel(ApiModel):
    weights: SeverityWeights = Field(default_factory=SeverityWeights)
    impact_prior: float = Field(0.5, ge=0, le=1)
    impact_prior_source: str = "default (unknown)"
    suitability_saturation: float = Field(0.10, gt=0, le=1)
    spread_half_saturation: float = Field(5.0, gt=0)
    spread_window_years: int = Field(10, ge=2, le=50)


class SeverityConfidence(ApiModel):
    level: ConfidenceT
    mess_ok_fraction: float
    model_type: ModelTypeT
    note: str


class SeverityComponents(ApiModel):
    suitability: float
    climate_analogy: float
    spread_rate: float
    ecological_impact_prior: float


class SeverityResult(ApiModel):
    score: float
    score_0_100: float
    category: Literal["low", "moderate", "high", "very high"]
    components: SeverityComponents
    weights: SeverityComponents
    contributions: SeverityComponents
    descriptions: dict[str, str]
    confidence: SeverityConfidence
    formula: str
    config: SeverityConfigModel
    inputs: dict[str, Any]


class ModelSummary(ApiModel):
    model_version: int
    model_type: ModelTypeT
    confidence_label: Literal["standard", "lower"]
    trained_ts: datetime
    trigger: str
    auc_mean: float | None = None
    cbi_mean: float | None = None
    tss_mean: float | None = None
    transferability_caveat: str
    severity: SeverityResult | None = None


class OccurrenceSources(ApiModel):
    """How the merged occurrence set was assembled (records are de-duplicated across sources)."""

    gbif: int = Field(description="Records ingested from GBIF")
    gbif_from_inaturalist: int = Field(
        description="Subset of the GBIF records that are iNaturalist research-grade "
        "observations republished to GBIF (linked by inat_crossref_id)"
    )
    inaturalist_direct: int = Field(
        description="Research-grade observations fetched from iNaturalist and not yet in GBIF"
    )
    last_inat_fetch_ts: datetime | None = None


class SpeciesDetail(SpeciesSummary):
    canonical_name: str | None = None
    inat_taxon_id: int | None = None
    last_gbif_download_key: str | None = None
    last_gbif_download_doi: str | None = None
    bioclim_version_used: str | None = None
    last_error: str | None = None
    native_range_status: Literal["missing", "draft", "confirmed"]
    sources: OccurrenceSources
    current_model: ModelSummary | None = None
    severity_config: SeverityConfigModel


# ----------------------------------------------------------- native range
class GeoJSONPolygon(ApiModel):
    type: Literal["Polygon"]
    coordinates: list[list[list[float]]]


class GeoJSONMultiPolygon(ApiModel):
    type: Literal["MultiPolygon"]
    coordinates: list[list[list[list[float]]]]


class NativeRangeOut(ApiModel):
    taxon_key: int
    geometry: GeoJSONMultiPolygon
    status: Literal["draft", "confirmed"]
    source: str
    note: str | None = None
    updated_ts: datetime
    confirmed_ts: datetime | None = None


class NativeRangeUpdate(ApiModel):
    geometry: GeoJSONPolygon | GeoJSONMultiPolygon = Field(discriminator="type")
    confirm: bool = Field(False, description="Mark as user-reviewed; required before training")
    note: str | None = None
    train: bool = Field(False, description="Enqueue a pipeline run after confirming")


# ------------------------------------------------------------------- jobs
class JobOut(ApiModel):
    job_id: uuid.UUID
    taxon_key: int | None
    kind: JobKind
    status: JobStatus
    stage: str | None = None
    progress: float
    message: str | None = None
    result: dict[str, Any] | None = None
    created_ts: datetime
    updated_ts: datetime


class NativeRangeUpdateResult(ApiModel):
    native_range: NativeRangeOut
    job: JobOut | None = None


class RunRequest(ApiModel):
    force_retrain: bool = False
    full_resync: bool = Field(
        False, description="Re-fetch all records (not only the delta) and append missing ones"
    )


class BulletinRequest(ApiModel):
    model_version: int | None = None


# ------------------------------------------------------------ occurrences
class OccurrenceRow(ApiModel):
    source: SourceT
    record_id: int
    longitude: float
    latitude: float
    event_date: date | None = None
    coordinate_uncertainty_m: float | None = None
    basis_of_record: str | None = None
    country_code: str | None = None
    range_label: RangeLabelT
    ingested_ts: datetime


class OccurrencePage(ApiModel):
    items: list[OccurrenceRow]
    total: int
    offset: int
    limit: int


class TimelineYear(ApiModel):
    year: int
    native: int
    introduced: int
    unknown: int
    cumulative: int


class Timeline(ApiModel):
    years: list[TimelineYear]
    min_year: int | None
    max_year: int | None


# ----------------------------------------------------------- model lineage
class TrainingSnapshotInfo(ApiModel):
    """A per-version, content-addressed data file (see training_snapshot.py)."""

    path: str = Field(description="Artifact path, relative to the artifact root")
    sha256: str
    n_rows: int
    bytes: int


class ReproducibilityInfo(ApiModel):
    bioclim_version_used: str
    bioclim_source: str | None = None
    bioclim_sha256: str | None = None
    bioclim_dvc_md5: str | None = Field(
        None, description="DVC hash of the bioclim version directory (data/bioclim/<v>.dvc)"
    )
    gbif_download_key: str | None = None
    gbif_download_doi: str | None = None
    gbif_citation: str
    inat_citation: str | None = None
    package_versions: dict[str, str]
    random_seeds: dict[str, int]
    training_config: dict[str, Any]
    native_range_source: str
    native_range_confirmed_ts: str | None = None
    training_data: TrainingSnapshotInfo | None = Field(
        None,
        description="Exact MaxEnt input (presences + background with predictors); "
        "absent for versions trained before snapshots were introduced",
    )
    occurrence_manifest: TrainingSnapshotInfo | None = Field(
        None, description="All valid records considered, with ingested and effective labels"
    )


class FoldMetricsOut(ApiModel):
    fold: int
    n_train_presence: int
    n_test_presence: int
    n_test_background: int
    auc: float | None
    tss: float | None
    cbi: float | None


class ModelVersionOut(ApiModel):
    model_version: int
    model_type: ModelTypeT
    confidence_label: Literal["standard", "lower"]
    trained_ts: datetime
    trigger: str
    auc_mean: float | None = None
    auc_std: float | None = None
    cbi_mean: float | None = None
    cbi_std: float | None = None
    tss_mean: float | None = None
    tss_std: float | None = None
    n_presence: int
    n_background: int
    n_cv_folds: int
    feature_classes: str
    beta_multiplier: float
    severity_score: float | None = None
    mlflow_run_id: str | None = None
    reproducibility: ReproducibilityInfo


class ProjectionSummaryOut(ApiModel):
    land_area_km2: float
    nonnative_land_area_km2: float
    native_suitable_area_km2: float
    established_outside_area_km2: float
    candidate_area_km2: float
    candidate_mean_suitability: float
    candidate_fraction_of_nonnative_land: float
    candidate_mess_ok_fraction: float
    candidate_envelope_fraction: float
    extrapolation_area_km2: float
    top_candidate_regions: list[dict[str, Any]]
    top_limiting_variables: list[dict[str, Any]]


class ModelVersionDetail(ModelVersionOut):
    threshold: float
    threshold_rule: str
    cv_method: str
    predictors: list[str]
    variable_importance: dict[str, float]
    folds: list[FoldMetricsOut]
    tuning: list[dict[str, Any]]
    projection: ProjectionSummaryOut
    transferability_caveat: str
    severity: SeverityResult | None = None
    n_records_by_effective_label: dict[str, int]


# ------------------------------------------------------------------ layers
class LegendEntry(ApiModel):
    value: float
    color: str
    label: str


class RasterLayer(ApiModel):
    id: str
    label: str
    tile_url: str = Field(description="XYZ template with {z}/{x}/{y}")
    kind: Literal["continuous", "categorical"]
    legend: list[LegendEntry]
    description: str


class VectorLayer(ApiModel):
    id: str
    label: str
    tile_url: str = Field(description="MVT XYZ template; extra query params may be appended")
    source_layer: str


class ScenarioLayer(ApiModel):
    id: str = Field(description="'scenario:<name>'")
    name: str
    label: str
    suitability: RasterLayer
    extrapolation: RasterLayer | None = Field(
        None, description="This scenario's own MESS < 0 mask (null for legacy projections)"
    )
    extrapolated_land_fraction: float | None = None


class LayerCaveats(ApiModel):
    model_type: ModelTypeT
    confidence_label: Literal["standard", "lower"]
    transferability_caveat: str
    mess_required: bool = True


class LayerSet(ApiModel):
    taxon_key: int
    model_version: int | None
    occurrences: VectorLayer
    native_range: VectorLayer
    suitability: RasterLayer | None = None
    mess: RasterLayer | None = None
    zones: RasterLayer | None = None
    extrapolation: RasterLayer | None = Field(
        None, description="MESS < 0 mask; always offered as an overlay with any projection"
    )
    scenarios: list[ScenarioLayer] = Field(default_factory=list)
    scenarios_available: list[str] = Field(
        default_factory=list,
        description="Scenarios registered for the model's bioclim version (projected or not)",
    )
    caveats: LayerCaveats | None = None


class BulletinStatus(ApiModel):
    taxon_key: int
    model_version: int | None
    pdf_available: bool
    html_available: bool
    pdf_url: str | None = None
    html_url: str | None = None


class BioclimInfo(ApiModel):
    version: str
    source: str
    resolution: str
    selected_predictors: list[str]
    vif: dict[str, float]
    descriptions: dict[str, str]
    scenarios: list[str]
    citation: str | None = None


class Health(ApiModel):
    status: Literal["ok"]
    bioclim_version: str
    bioclim_available: bool


# -------------------------------------------------------------- validation suite
class ValidationRegion(ApiModel):
    name: str
    bbox: list[float]
    expect: Literal["suitable", "unsuitable"]
    ref: str
    fraction_suitable: float | None = None
    mean_suitability: float | None = None
    passed: bool


class TransferabilityResult(ApiModel):
    auc: float | None = None
    cbi: float | None = None
    n_native_train: int | None = None
    n_introduced_test: int | None = None
    literature_mean_auc: float | None = None
    method: str | None = None
    skipped: str | None = None


class ValidationSpecies(ApiModel):
    taxon_key: int
    scientific_name: str
    native_range_reference: str
    status: Literal["evaluated", "no_model", "error"]
    passed: bool
    error: str | None = None
    model_version: int | None = None
    model_type: ModelTypeT | None = None
    bioclim_version: str | None = None
    auc_mean: float | None = None
    cbi_mean: float | None = None
    tss_mean: float | None = None
    n_occurrences: int | None = None
    regions: list[ValidationRegion] = Field(default_factory=list)
    invaded_captured: int | None = None
    invaded_total: int | None = None
    controls_correct: int | None = None
    controls_total: int | None = None
    transferability: TransferabilityResult | None = None


class ValidationSummary(ApiModel):
    n_species: int
    n_passed: int
    mean_transfer_auc: float | None = None
    literature_mean_transfer_auc: float


class ValidationReport(ApiModel):
    generated_ts: datetime
    criteria: dict[str, Any]
    species: list[ValidationSpecies]
    summary: ValidationSummary


class ValidationRunRequest(ApiModel):
    taxon_keys: list[int] | None = Field(None, description="Subset of reference species")
    train: bool = Field(True, description="Ingest/train reference species that lack a model")
