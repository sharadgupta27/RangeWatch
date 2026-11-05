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
QualityTierT = Literal["A", "B", "C", "excluded"]
AlgorithmT = Literal["maxent", "ensemble", "integrated", "ppm"]
AnnotationKindT = Literal["range_label", "exclude", "cultivated", "verified"]
AlertKindT = Literal["outside_envelope", "outside_known_range"]
RoleT = Literal["viewer", "analyst", "curator"]


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
    confidence_low_extrapolated: float = Field(
        0.25,
        ge=0,
        le=1,
        description="Low confidence above this share of the candidate zone flagged by ≥ 3 of "
        "the 5 extrapolation diagnostics (MESS share for models without them)",
    )
    confidence_moderate_extrapolated: float = Field(
        0.10, ge=0, le=1, description="Moderate confidence above this extrapolated share"
    )
    confidence_min_analog: float = Field(
        0.60,
        ge=0,
        le=1,
        description="Moderate confidence below this share of the candidate zone flagged by "
        "no diagnostic",
    )
    confidence_max_mess_extrapolated: float = Field(
        0.25,
        ge=0,
        le=1,
        description="MESS cap: at most moderate confidence above this share of the candidate "
        "zone with MESS < 0",
    )
    confidence_min_cbi: float = Field(
        0.20,
        ge=-1,
        le=1,
        description="CBI floor: at most moderate confidence below this spatial-CV CBI",
    )


class SeverityConfidence(ApiModel):
    level: ConfidenceT
    basis: Literal["consensus", "mess"] = Field(
        "mess", description="consensus of the five diagnostics, or MESS for older models"
    )
    reasons: list[str] = Field(
        default_factory=list, description="Every rule that lowered confidence (empty = high)"
    )
    cbi_mean: float | None = None
    mess_ok_fraction: float
    consensus_ok_fraction: float | None = None
    consensus_majority_fraction: float | None = None
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
    study_region: StudyRegionOut | None = Field(
        None, description="Regional model area (U2); null = global model"
    )


# ----------------------------------------------------------- native range
class StudyRegionOut(ApiModel):
    bbox: list[float] = Field(description="[west, south, east, north] in degrees")
    resolution: str | None = Field(
        None, description="Registered high-resolution stack (e.g. 30s); null = base grid"
    )


class StudyRegionUpdate(ApiModel):
    bbox: list[float] | None = Field(
        None,
        min_length=4,
        max_length=4,
        description="[west, south, east, north]; null removes the region (global model)",
    )
    resolution: str | None = None


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
    issues: list[str] | None = Field(
        None, description="GBIF issue flags (null: ingested before issues were recorded)"
    )
    quality_flags: list[str] | None = Field(
        None,
        description="Data-quality flags from the current model version's assessment (null: "
        "not assessed, e.g. outside the climate grid or ingested after training)",
    )
    quality_tier: QualityTierT | None = None
    annotation: EffectiveAnnotationOut | None = Field(
        None, description="Effective expert annotation (latest per kind), if any"
    )
    ingested_ts: datetime


class EffectiveAnnotationOut(ApiModel):
    range_label: RangeLabelT | None = None
    exclude: bool | None = None
    cultivated: bool | None = None
    verified: bool | None = None
    n_annotations: int
    last_author: str | None = None
    last_ts: str | None = None


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


class TargetGroupInfo(ApiModel):
    rank: str
    taxon_key: int = Field(description="GBIF key of the target group (e.g. the species' order)")
    name: str
    n_records: int = Field(description="GBIF records of the target group in the region")
    n_pixels: int
    map_zoom: int
    pixel_deg: float
    n_tiles: int
    query: str = Field(description="GBIF map API density-tile query used as sampling effort")
    fetched_ts: str
    weighting: str | None = Field(
        None,
        description="unique_cells (each cell once), record_count (bias file) or "
        "sqrt_record_count; absent for versions trained before the option existed",
    )
    n_background_cells: int | None = Field(
        None, description="Distinct cells in the background sample"
    )
    matched: dict[str, Any] | None = Field(
        None,
        description="B1: effort restricted to the presences' basis of record and year window",
    )


class BackgroundInfo(ApiModel):
    method: str = Field(description="Method actually used, e.g. target_group_order, buffer_500km")
    requested_method: str
    buffer_km: float
    target_group: TargetGroupInfo | None = None
    fallback_reason: str | None = Field(
        None, description="Why target-group background was not used, if it was requested"
    )
    bias_covariate_override: str | None = Field(
        None,
        description="Bias covariate that replaced the target-group background (bias modelled "
        "explicitly; background uniform in the buffer)",
    )


class DataQualityTestOut(ApiModel):
    id: str
    label: str
    severity: Literal["error", "warning"]
    description: str
    reference: str | None = None
    excluded: bool = Field(description="Records it flags are excluded from training")
    status: Literal["run", "skipped"]
    reason: str | None = Field(None, description="Why the test was skipped")
    n_flagged: int


class DataQualityOut(ApiModel):
    """Record-level data-quality assessment of one training run (features/quality.py)."""

    version: str
    config: dict[str, Any]
    reference: dict[str, Any] | None = Field(
        None, description="Reference gazetteer version and per-table SHA-256 (null: none)"
    )
    n_records: int = Field(description="Records with climate data that were assessed")
    n_excluded: int
    n_flagged_kept: int = Field(description="Records with warning flags only (kept)")
    n_outside_climate_grid: int | None = None
    tiers: dict[str, int]
    tier_rules: dict[str, str]
    tests: list[DataQualityTestOut]
    issue_counts: dict[str, int] = Field(default_factory=dict)
    n_issues_unknown: int = 0
    n_details_unknown: int = Field(
        0, description="Records ingested before locality / collector / identification details"
    )
    n_undated: int = 0
    grid_cell_m: float | None = None
    environmental_outlier_cutoff: dict[str, Any] | None = None
    snapshot: TrainingSnapshotInfo | None = Field(
        None, description="quality_flags.parquet: every assessed record with its flags"
    )


class TrainingUploadInfo(ApiModel):
    """An upload with role "training" whose presences joined this version's training set."""

    dataset_id: str
    name: str
    file_name: str | None = None
    file_sha256: str | None = None
    citation: str | None = None
    n_presences: int
    n_overlap_excluded: int = Field(description="Duplicates of a GBIF/iNaturalist record, left out")
    n_added: int = Field(description="Presences appended before quality tests and thinning")
    used_as: Literal["presence_only", "survey"] = Field(
        "presence_only",
        description="survey: presence–absence data of an integrated model (presences not "
        "added to the presence-only set)",
    )
    n_survey_presences: int | None = None
    n_survey_absences: int | None = None
    n_absences_not_used: int = Field(
        description="Absences: a presence-background model cannot use them (they evaluate it)"
    )


class PresenceWeightsInfo(ApiModel):
    """Observation weights given to presences (features/sampling_bias.py)."""

    method: Literal["none", "quality_tier", "density", "quality_tier_density"]
    tier_weights: dict[str, float] | None = None
    n_by_tier: dict[str, int] | None = None
    density_neighbors: int | None = None
    n_presences: int | None = None
    rescaled_to_mean: float | None = None
    min: float | None = None
    median: float | None = None
    max: float | None = None


class StaticBiasLayerInfo(ApiModel):
    """A registered static observer-bias raster used as a covariate (travel time, roads, …)."""

    name: str
    status: Literal["used", "unavailable"]
    reason: str | None = None
    covariate: str | None = None
    direction: Literal["increasing", "decreasing"] | None = Field(
        None, description="Whether larger values mean more (increasing) or less effort"
    )
    transform: str | None = None
    condition: str | None = None
    conditioning_value: float | None = None
    n_missing_filled: int | None = None
    presence_mean: float | None = None
    background_mean: float | None = None
    citation: str | None = None
    sha256: str | None = None


class BiasCovariatesInfo(ApiModel):
    """Bias covariate fitted with the model and held constant for prediction."""

    method: Literal["none", "target_group_effort"]
    status: Literal["used", "unavailable"] | None = None
    reason: str | None = None
    covariate: str | None = None
    transform: str | None = None
    condition: Literal["max", "p95", "estimated", "offset"] | None = None
    conditioning_value: float | None = None
    note: str | None = None
    reference: str | None = None
    target_group: dict[str, Any] | None = None
    n_records: int | None = None
    pixel_deg: float | None = None
    map_zoom: int | None = None
    query: str | None = None
    fetched_ts: str | None = None
    presence_mean: float | None = None
    background_mean: float | None = None
    background_note: str | None = None
    importance: dict[str, float] | None = Field(
        None, description="Permutation importance (%) of the covariate itself"
    )
    static_layers: list[StaticBiasLayerInfo] | None = None


class AlgorithmInfo(ApiModel):
    requested: AlgorithmT
    used: AlgorithmT
    fallback_reason: str | None = None
    point_process: dict[str, Any] | None = Field(
        None, description="Point process: bias mode used, shared-bias estimate, fallback"
    )


class ThinningInfo(ApiModel):
    """How presences were thinned (features/thinning.py; B3, B4)."""

    method: Literal["distance", "autocorrelation", "environmental"]
    n_before: int
    n_after: int
    seed: int
    distance_km: float | None = None
    reps: int | None = None
    selected_seed: int | None = None
    n_kept_min: int | None = None
    n_kept_max: int | None = None
    autocorrelation: dict[str, Any] | None = Field(
        None, description="Variogram ranges per predictor and the clamped distance (B3)"
    )
    fallback_reason: str | None = None
    reference: str | None = None
    n_axes: int | None = None
    n_bins: int | None = None
    variance_explained: list[float] | None = None
    n_occupied_cells: int | None = None


class PseudoAbsenceInfo(ApiModel):
    """Pseudo-absences of the presence–absence ensemble members (Barbet-Massin et al. 2012)."""

    strategy: Literal["background", "disk", "environmental_profile", "completeness"]
    method: str = Field(description="What was actually drawn (after any fallback)")
    n: int
    n_requested: int
    buffer_km: float
    seed: int
    min_distance_km: float | None = None
    fallback_reason: str | None = None
    reference: str | None = None
    envelope: str | None = None
    caveat: str | None = None
    zone: str | None = None
    min_records: float | None = None
    target_group: dict[str, Any] | None = None
    query: str | None = None
    n_effort_pixels: int | None = None
    n_well_sampled: int | None = None
    n_without_presence: int | None = None
    n_sampled: int | None = None


class AnnotationsInfo(ApiModel):
    """Expert annotations applied to this version's training set (U11)."""

    n_records_annotated: int = 0
    n_annotations: int = 0
    n_excluded_by_annotation: int = 0
    n_excluded_suspicious: int = 0
    n_excluded_cultivated: int = 0
    n_kept_verified: int = 0
    n_label_overrides: int = 0
    rules: str | None = None
    snapshot: TrainingSnapshotInfo | None = None


class StudyRegionInfo(ApiModel):
    bbox: list[float]
    resolution: str
    grid_bbox: list[float]
    n_cells: int
    cell_deg: float
    n_records_outside: int = 0


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
    background: BackgroundInfo | None = Field(
        None, description="Background sampling record; absent for versions trained before it"
    )
    training_data: TrainingSnapshotInfo | None = Field(
        None,
        description="Exact MaxEnt input (presences + background with predictors); "
        "absent for versions trained before snapshots were introduced",
    )
    occurrence_manifest: TrainingSnapshotInfo | None = Field(
        None, description="All valid records considered, with ingested and effective labels"
    )
    data_quality: DataQualityOut | None = Field(
        None, description="Data-quality tests and exclusions; absent for older versions"
    )
    training_uploads: list[TrainingUploadInfo] = Field(
        default_factory=list, description="Uploads whose presences joined the training set"
    )
    presence_weights: PresenceWeightsInfo | None = Field(
        None, description="Presence observation weights; absent for older versions"
    )
    algorithm: AlgorithmInfo | None = Field(
        None, description="Requested and used algorithm (null: MaxEnt, before the option)"
    )
    pseudo_absences: PseudoAbsenceInfo | None = None
    annotations: AnnotationsInfo | None = Field(
        None, description="null for versions trained before record annotations existed"
    )
    study_region: StudyRegionInfo | None = Field(None, description="null = global model")
    bias_covariates: BiasCovariatesInfo | None = Field(
        None, description="Bias covariate + conditioning; absent for older versions"
    )
    thinning: ThinningInfo | None = Field(
        None, description="Thinning method and result; absent for older versions (distance)"
    )


class SamplingEffortOut(ApiModel):
    """Global GBIF sampling effort of the target group, summarised per range zone."""

    status: Literal["available", "unavailable", "disabled"]
    reason: str | None = None
    target_group: dict[str, Any] | None = None
    n_records: int | None = None
    max_records_per_pixel: int | None = None
    pixel_deg: float | None = None
    map_zoom: int | None = None
    n_tiles: int | None = None
    query: str | None = None
    fetched_ts: str | None = None
    land_unsampled_fraction: float | None = Field(
        None, description="Share of land area without any record of the target group"
    )
    native_suitable_unsampled_fraction: float | None = None
    established_outside_unsampled_fraction: float | None = None
    candidate_unsampled_fraction: float | None = Field(
        None,
        description="Share of the candidate invasion zone without target-group records: "
        "there, the absence of presences says nothing about suitability",
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
    algorithm: AlgorithmT = "maxent"
    threshold_rule: str = "10th percentile training presence"
    reproducibility: ReproducibilityInfo


ExtrapolationDiagnosticId = Literal["mess", "exdet", "mop", "shape", "aoa"]


class ExtrapolationDiagnosticOut(ApiModel):
    id: ExtrapolationDiagnosticId
    label: str
    name: str
    reference: str
    flag_rule: str = Field(description="When this diagnostic flags a cell as extrapolation")
    threshold: float | None
    threshold_rule: str = Field(description="How the threshold was derived")
    land_area_km2: float
    land_fraction: float
    candidate_flagged_fraction: float = Field(
        description="Share of the candidate invasion zone flagged by this diagnostic"
    )


class ExtrapolationSummaryOut(ApiModel):
    reference_set: str
    n_reference: int
    n_cv_folds: int = Field(description="Spatial CV folds used for the MOP/Shape/AOA thresholds")
    mop_percentage: float = Field(description="% of closest reference points averaged by MOP")
    mop_k: int
    aoa_weights: dict[str, float] = Field(
        description="Predictor weights of the AOA distance (permutation importance, max = 1)"
    )
    diagnostics: list[ExtrapolationDiagnosticOut]
    consensus_land_area_km2: list[float] = Field(
        description="Land area flagged by exactly 0, 1, …, 5 of the diagnostics"
    )
    candidate_consensus_ok_fraction: float = Field(
        description="Share of the candidate zone flagged by none of the diagnostics"
    )
    candidate_consensus_majority_fraction: float = Field(
        description="Share of the candidate zone flagged by at least 3 of the 5 diagnostics"
    )
    exdet_univariate_area_km2: float
    exdet_combinatorial_area_km2: float
    top_combinatorial_variables: list[dict[str, Any]] = Field(
        description="exDet most influential covariates where NT2 > 1 (novel combinations)"
    )


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
    limiting_factor: LimitingFactorOut | None = Field(
        None, description="Limiting factor per area (Elith et al. 2010); null for older versions"
    )
    ensemble_agreement: EnsembleAgreementOut | None = None
    extrapolation: ExtrapolationSummaryOut | None = Field(
        None,
        description="MESS, exDet, MOP, Shape and AOA verdicts (null for models trained before "
        "the advanced diagnostics existed)",
    )


class AreaShareOut(ApiModel):
    variable: str
    area_km2: float
    fraction: float


class LimitingFactorOut(ApiModel):
    rule: str
    land: list[AreaShareOut]
    candidate: list[AreaShareOut]


class EnsembleAgreementOut(ApiModel):
    rule: str
    land_mean: float | None = None
    candidate_mean: float | None = None
    low_agreement_land_fraction: float | None = None
    low_agreement_below: float


class PredictorAgreementOut(ApiModel):
    predictor: str
    pearson_r: float | None
    mean_diff: float | None = Field(description="Mean of (alternative − primary)")
    mean_abs_diff: float | None
    scale_ratio: float | None = Field(
        description="Median |alternative| / median |primary|; far from 1 suggests a unit mismatch"
    )
    units_suspect: bool


class CrossCheckMetricsOut(ApiModel):
    auc_mean: float | None = None
    cbi_mean: float | None = None
    tss_mean: float | None = None
    threshold: float | None = None


class ClimateCrossCheckOut(ApiModel):
    """The model version refitted on an independent climate source (same training points,
    feature classes, regularisation and CV seed) and compared with the original."""

    alt_bioclim_version: str
    alt_source: str
    primary_bioclim_version: str
    created_ts: str
    n_presence: int
    n_background: int
    n_dropped: int = Field(description="Training points without data in the alternative stack")
    feature_classes: str
    beta_multiplier: float
    predictors: list[PredictorAgreementOut]
    primary: CrossCheckMetricsOut
    alternative: CrossCheckMetricsOut
    suitability_rank_correlation: float | None = Field(
        description="Spearman ρ of the two models' suitability at the training points"
    )
    classification_agreement: float | None
    classification_kappa: float | None
    verdict: Literal["consistent", "moderate", "divergent"]
    interpretation: str


class ThresholdOut(ApiModel):
    label: str
    value: float
    sensitivity: float = Field(description="Share of training presences ≥ threshold")
    specificity_background: float | None = Field(
        None, description="Share of background (pseudo-absence) points below the threshold"
    )


class ResponseCurveOut(ApiModel):
    predictor: str
    x: list[float]
    y: list[float]
    presence_range: list[float] = Field(description="5th–95th percentile at presences")


class ResponseCurvesOut(ApiModel):
    method: str
    curves: list[ResponseCurveOut]


class EnsembleMemberOut(ApiModel):
    algorithm: Literal["maxent", "glm", "gam", "rf", "brt"]
    label: str
    included: bool
    weight: float
    threshold: float
    auc_mean: float | None = None
    auc_std: float | None = None
    cbi_mean: float | None = None
    cbi_std: float | None = None
    tss_mean: float | None = None
    tss_std: float | None = None
    n_cv_folds: int
    params: dict[str, Any]
    note: str | None = None


class EnsembleOut(ApiModel):
    members: list[EnsembleMemberOut]
    weighting: str
    cv_test_set: str
    reference: str


class IntegratedOut(ApiModel):
    method: str
    reference: str
    features: str
    l2_penalty: float
    converged: bool
    iterations: int
    neg_log_likelihood: float
    n_presence_only: int
    n_quadrature: int
    n_pa_present: int
    n_pa_absent: int
    intercept_presence_absence: float
    intercept_presence_only: float
    coefficients: dict[str, float]
    bias_coefficients: dict[str, float] = Field(
        description="Estimated observer-bias coefficients (presence-only data only)"
    )
    prediction: str
    cv_survey_auc: list[float | None]
    cv_survey_auc_mean: float | None = None
    presence_weights_used: bool


class OffsetTestOut(ApiModel):
    column: str
    assumed_coefficient: float
    free_coefficient: float
    lr_statistic: float
    p_value: float
    interpretation: str
    penalty_note: str | None = None


class PointProcessOut(ApiModel):
    """Point-process model with an effort offset (B8) or a shared bias (B9)."""

    method: str
    reference: str
    features: str
    bias_mode: Literal["offset", "estimated", "shared"]
    l2_penalty: float
    converged: bool
    iterations: int
    neg_log_likelihood: float
    n_presences: int
    n_quadrature: int
    coefficients: dict[str, float]
    free_bias_coefficients: dict[str, float]
    fixed_bias_coefficients: dict[str, float]
    offset: str
    offset_test: OffsetTestOut | None = None
    prediction: str
    presence_weights_used: bool


class InferredAbsencesOut(ApiModel):
    status: Literal["evaluated", "insufficient", "unavailable", "disabled"]
    reason: str | None = None
    note: str | None = None
    zone: str | None = None
    min_records: float | None = None
    min_distance_km: float | None = None
    target_group: dict[str, Any] | None = None
    n_effort_pixels: int | None = None
    n_well_sampled: int | None = None
    n_without_presence: int | None = None
    n_sampled: int | None = None
    n: int | None = None
    threshold: float | None = None
    specificity: float | None = Field(None, description="Share of inferred absences < threshold")
    mean_suitability: float | None = None
    auc_training_presence_vs_inferred: float | None = None


class ChangeSummaryOut(ApiModel):
    status: Literal["available", "unavailable"] = "available"
    reason: str | None = None
    from_version: int | None = None
    to_version: int | None = None
    codes: dict[str, str] | None = None
    gain_km2: float | None = None
    loss_km2: float | None = None
    stable_suitable_km2: float | None = None
    stable_unsuitable_km2: float | None = None
    net_change_fraction: float | None = None
    before_threshold: float | None = None
    after_threshold: float | None = None


class PriorityCellOut(ApiModel):
    rank: int
    longitude: float
    latitude: float
    priority: float
    suitability: float
    effort: float
    uncertainty: float


class SurveyPriorityOut(ApiModel):
    status: Literal["available", "unavailable"]
    reason: str | None = None
    formula: str | None = None
    uncertainty_source: str | None = None
    suitable_area_km2: float | None = None
    top_cells: list[PriorityCellOut] = Field(default_factory=list)


class ModelVersionDetail(ModelVersionOut):
    threshold: float
    thresholds: dict[str, ThresholdOut] = Field(
        default_factory=dict, description="Every threshold rule (empty for older versions)"
    )
    cv_method: str
    predictors: list[str]
    variable_importance: dict[str, float]
    folds: list[FoldMetricsOut]
    tuning: list[dict[str, Any]]
    projection: ProjectionSummaryOut
    transferability_caveat: str
    severity: SeverityResult | None = None
    n_records_by_effective_label: dict[str, int]
    crosschecks: list[ClimateCrossCheckOut] = Field(default_factory=list)
    response_curves: ResponseCurvesOut | None = None
    ensemble: EnsembleOut | None = None
    integrated: IntegratedOut | None = None
    point_process: PointProcessOut | None = None
    inferred_absences: InferredAbsencesOut | None = None
    change_previous: ChangeSummaryOut | None = Field(
        None, description="Binary change against the previous version (U6)"
    )
    survey_priority: SurveyPriorityOut | None = None
    sampling_effort: SamplingEffortOut | None = Field(
        None, description="Absent for versions trained before the effort layer existed"
    )


# -------------------------------------------------------- reference datasets
class IndependentMetricsOut(ApiModel):
    """Scores on records the model never saw (modeling/evaluation.independent_evaluation)."""

    threshold: float
    n_presence: int
    n_absence: int
    n_outside_grid: int
    sensitivity: float | None = Field(None, description="Share of presences ≥ threshold")
    mean_suitability_presence: float | None = None
    cbi: float | None = Field(None, description="Presences vs the model's own background")
    specificity: float | None = Field(
        None, description="Share of absences < threshold (a lower bound: absence ≠ empty cell)"
    )
    mean_suitability_absence: float | None = None
    auc_presence_absence: float | None = None
    tss: float | None = None
    sensitivity_analog: float | None = Field(
        None, description="Sensitivity at presences inside the training climate (MESS ≥ 0)"
    )
    sensitivity_extrapolated: float | None = Field(
        None, description="Sensitivity at presences in extrapolated cells (MESS < 0)"
    )
    n_presence_extrapolated: int | None = None


class ReferenceEvaluationOut(ApiModel):
    model_version: int
    model_type: ModelTypeT | None = None
    trained_ts: datetime | None = None
    evaluated_ts: str
    records_scored: Literal["all", "absences"] = Field(
        "all",
        description="absences: a training dataset, whose presences are not independent",
    )
    n_records: int
    n_excluded_overlap: int = Field(
        description="Records left out because they duplicate a GBIF/iNaturalist record"
    )
    metrics: IndependentMetricsOut
    metrics_all_records: IndependentMetricsOut | None = Field(
        None, description="Including the overlapping records (only when there are any)"
    )


class OverlapOut(ApiModel):
    n_checked: int
    n_overlap: int
    fraction: float
    distance_m: float
    max_days: int
    n_occurrences_compared: int
    checked_ts: str
    rule: str


class ReferenceDatasetOut(ApiModel):
    taxon_key: int
    dataset_id: str
    kind: Literal["gbif_absences", "upload"]
    role: Literal["evaluation", "training"] = Field(
        description="evaluation: independent test data, never trained on. training: its "
        "presences join the training set; only its absences evaluate the model"
    )
    name: str
    citation: str | None = None
    license: str | None = None
    note: str | None = None
    file_name: str | None = None
    file_sha256: str | None = None
    parse_report: dict[str, Any] | None = None
    overlap: OverlapOut | None = Field(
        None, description="Independence check against the occurrence store (uploads only)"
    )
    n_present: int
    n_absent: int
    last_fetch_ts: datetime | None = None
    created_ts: datetime
    updated_ts: datetime
    evaluations: list[ReferenceEvaluationOut] = Field(
        default_factory=list, description="One entry per evaluated model version, newest first"
    )


class DatasetUploadResult(ApiModel):
    dataset: ReferenceDatasetOut
    job: JobOut | None = Field(None, description="Independence check + evaluation job")


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


class GcmEnsembleOut(ApiModel):
    """U7: one SSP × period combined across GCMs (the scenario entry `ensemble_<ssp>_<period>`)."""

    ssp: str
    period: str
    gcms: list[str]
    n_gcms: int
    threshold: float = Field(description="Model threshold each GCM's map is binarised at")
    combination: str
    suitable_all_gcms_land_fraction: float
    suitable_majority_land_fraction: float
    suitable_any_gcm_land_fraction: float
    gcm_disagreement_land_fraction: float = Field(
        description="Share of land suitable under some but not all GCMs"
    )
    extrapolated_any_gcm_land_fraction: float
    extrapolated_all_gcms_land_fraction: float
    land_fraction_by_votes: dict[str, float] = Field(
        description="Share of land by number of GCMs calling it suitable"
    )
    agreement: RasterLayer | None = Field(
        None, description="Number of GCMs calling each cell suitable (map layer)"
    )


class ScenarioLayer(ApiModel):
    id: str = Field(description="'scenario:<name>'")
    name: str
    label: str
    suitability: RasterLayer
    extrapolation: RasterLayer | None = Field(
        None, description="This scenario's own MESS < 0 mask (null for legacy projections)"
    )
    extrapolated_land_fraction: float | None = None
    consensus: RasterLayer | None = Field(
        None, description="This scenario's own diagnostic-consensus overlay (null for legacy)"
    )
    diagnostics_land_fraction: dict[str, float] | None = Field(
        None, description="Share of land flagged by each extrapolation diagnostic"
    )
    change: RasterLayer | None = Field(
        None, description="Present → scenario gain / loss / stable at the model threshold"
    )
    change_summary: ChangeSummaryOut | None = None
    gcm_ensemble: GcmEnsembleOut | None = Field(
        None,
        description="Set for multi-GCM ensembles (U7): suitability is the GCM mean, the "
        "extrapolation mask flags MESS < 0 under any GCM",
    )


class HiresLayer(ApiModel):
    """Regional high-resolution projection of the same model (no retraining)."""

    resolution: str
    bbox: list[float] = Field(description="[west, south, east, north] of the projected region")
    suitability: RasterLayer
    extrapolation: RasterLayer = Field(description="This region's own MESS < 0 mask")
    extrapolated_land_fraction: float | None = None
    consensus: RasterLayer | None = Field(
        None, description="This region's own diagnostic-consensus overlay (null for legacy)"
    )
    diagnostics_land_fraction: dict[str, float] | None = None
    created_ts: str


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
    diagnostics: list[RasterLayer] = Field(
        default_factory=list,
        description="Advanced extrapolation diagnostics as map views: exdet, mop, shape, aoa "
        "and consensus (empty for models trained before they existed)",
    )
    consensus_overlay: RasterLayer | None = Field(
        None,
        description="Haze overlay graded by how many diagnostics flag extrapolation; an "
        "alternative to the MESS < 0 overlay",
    )
    scenarios: list[ScenarioLayer] = Field(default_factory=list)
    scenarios_available: list[str] = Field(
        default_factory=list,
        description="Scenarios registered for the model's bioclim version (projected or not)",
    )
    hires: HiresLayer | None = None
    hires_available: str | None = Field(
        None, description="Resolution of the model's high-resolution stack, if one is registered"
    )
    hires_max_cells: int | None = None
    effort: RasterLayer | None = Field(
        None, description="Target-group sampling effort (log records per pixel)"
    )
    reference: VectorLayer | None = Field(
        None, description="Evaluation-only reference records (uploads, GBIF absences)"
    )
    products: list[RasterLayer] = Field(
        default_factory=list,
        description="Further map products: binary maps per threshold rule, limiting factor, "
        "ensemble agreement, change since the previous version, survey priority",
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
    hires_resolutions: list[str] = Field(default_factory=list)
    crosscheck_version: str | None = Field(
        None, description="Independent climate stack used for the climate-data cross-check"
    )
    crosscheck_source: str | None = None


class HiresRequest(ApiModel):
    bbox: list[float] = Field(
        min_length=4, max_length=4, description="[west, south, east, north] in degrees"
    )


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


# -------------------------------------------------------------- annotations (U11)
class AnnotationIn(ApiModel):
    source: SourceT
    record_id: int
    kind: AnnotationKindT
    value: str = Field(
        description="range_label: native | introduced | unknown; others: true | false"
    )
    note: str | None = Field(None, max_length=2000)


class AnnotationOut(ApiModel):
    annotation_id: uuid.UUID
    taxon_key: int
    source: SourceT
    record_id: int
    kind: AnnotationKindT
    value: str
    note: str | None = None
    author: str
    created_ts: datetime


class AnnotationResult(ApiModel):
    annotation: AnnotationOut
    retrain_flagged: bool = Field(description="The current model no longer reflects the data")


# -------------------------------------------------------------- alerts (U10 / B10)
class AlertRecordOut(ApiModel):
    source: SourceT
    record_id: int
    longitude: float
    latitude: float
    event_date: str | None = None
    suitability: float | None = None
    distance_km: float | None = None
    effort_class: Literal["expansion", "discovery", "unknown"]


class AlertOut(ApiModel):
    alert_id: uuid.UUID
    taxon_key: int
    kind: AlertKindT
    n_records: int
    records: list[AlertRecordOut]
    summary: str
    model_version: int | None = None
    effort_classes: dict[str, int] = Field(
        description="B10: expansion (cell sampled before) / discovery (unsampled) / unknown"
    )
    deliveries: list[dict[str, Any]]
    created_ts: datetime


class AlertSubscriptionIn(ApiModel):
    channel: Literal["webhook", "email"]
    target: str = Field(min_length=3, max_length=500, description="https URL or e-mail address")
    taxon_key: int | None = Field(None, description="null = every species")
    kinds: list[AlertKindT] = Field(
        default_factory=lambda: ["outside_envelope", "outside_known_range"], min_length=1
    )


class AlertSubscriptionOut(ApiModel):
    subscription_id: uuid.UUID
    channel: Literal["webhook", "email"]
    target: str
    taxon_key: int | None = None
    kinds: list[AlertKindT]
    created_by: str
    created_ts: datetime


# -------------------------------------------------------------- users (U14)
class MeOut(ApiModel):
    user: str
    role: RoleT
    auth_mode: Literal["none", "header"]
    permissions: dict[str, bool] = Field(
        description="What this user may do: run (analyst), curate (confirm native ranges, "
        "annotate records)"
    )


# -------------------------------------------------------------- exports (U8)
class ExportItem(ApiModel):
    id: str
    label: str
    url: str
    media_type: str
    description: str


class ExportIndex(ApiModel):
    taxon_key: int
    model_version: int
    items: list[ExportItem]


class OdmapSection(ApiModel):
    section: str
    entries: dict[str, Any]


class OdmapReport(ApiModel):
    """ODMAP protocol (Zurell et al. 2020) generated from the version's stored metadata."""

    protocol: str
    taxon_key: int
    model_version: int
    generated_ts: datetime
    sections: list[OdmapSection]


class StacAsset(ApiModel):
    href: str
    type: str
    title: str
    roles: list[str]


class StacItem(ApiModel):
    """STAC 1.0 Item describing one model version's outputs."""

    type: Literal["Feature"] = "Feature"
    stac_version: str
    stac_extensions: list[str] = Field(default_factory=list)
    id: str
    bbox: list[float]
    geometry: dict[str, Any]
    properties: dict[str, Any]
    links: list[dict[str, Any]]
    assets: dict[str, StacAsset]


# -------------------------------------------------------------- benchmark (§4.6)
class BenchmarkRequest(ApiModel):
    n_presences: int = Field(300, ge=50, le=5000)
    bias: Literal["synthetic", "gbif_effort"] = Field(
        "synthetic",
        description="synthetic: effort decays from a few 'cities'; gbif_effort: the real GBIF "
        "effort of `target_group_key` (map API)",
    )
    target_group_key: int | None = Field(None, description="GBIF key, e.g. an order")
    bias_strength: float = Field(3.0, ge=0, le=10, description="Exponent on the effort surface")
    variants: list[
        Literal["no_correction", "target_group", "bias_covariate", "density_weights"]
    ] = Field(
        default_factory=lambda: [
            "no_correction",
            "target_group",
            "bias_covariate",
            "density_weights",
        ],
        min_length=1,
    )
    region: list[float] | None = Field(
        None, min_length=4, max_length=4, description="[w, s, e, n]; null = global land"
    )
    seed: int = 42


class BenchmarkVariant(ApiModel):
    variant: str
    description: str
    spearman_vs_truth: float | None = None
    schoeners_d: float | None = None
    auc_vs_true_occupancy: float | None = None
    cv_cbi_mean: float | None = Field(None, description="What spatial CV on the biased data says")
    cv_auc_mean: float | None = None
    error: str | None = None


class BenchmarkReport(ApiModel):
    benchmark_id: str
    created_ts: datetime
    request: BenchmarkRequest
    virtual_species: dict[str, Any]
    bias: dict[str, Any]
    n_presences: int
    n_evaluation_cells: int
    variants: list[BenchmarkVariant]
    best_variant: str | None = None
    interpretation: str


# -------------------------------------------------------- model cards (U13)
class ModelCardFact(ApiModel):
    label: str
    value: str


class ModelCardSection(ApiModel):
    id: str
    title: str
    paragraphs: list[str]
    facts: list[ModelCardFact] = Field(default_factory=list)


class ModelCardOut(ApiModel):
    """Plain-language summary of one model version; confidence and caveats come first."""

    taxon_key: int
    scientific_name: str
    model_version: int
    generated_ts: str
    confidence_label: Literal["standard", "lower"]
    model_type: ModelTypeT
    headline: str
    transferability_caveat: str
    caveats: list[str] = Field(description="Shown before anything else, in this order")
    sections: list[ModelCardSection]
    citation: str
    source: str


# -------------------------------------------------------- species lists (U9)
class SpeciesListEntryOut(ApiModel):
    query: str = Field(description="As given: a scientific name or a GBIF taxon key")
    taxon_key: int | None = Field(None, description="null: no GBIF backbone match")
    scientific_name: str | None = None
    match_type: str | None = Field(None, description="EXACT, FUZZY, HIGHERRANK, KEY or NONE")
    confidence: int | None = None
    registered: bool = False
    status: str | None = Field(None, description="Registry status of a registered species")
    model_version: int | None = None


class SpeciesListIn(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    source: str | None = Field(None, max_length=500, description="Citation / URL of the list")
    names: list[str] = Field(default_factory=list, description="Scientific names to resolve")
    taxon_keys: list[int] = Field(default_factory=list)


class SpeciesListOut(ApiModel):
    list_id: str
    name: str
    description: str | None = None
    source: str | None = None
    created_by: str
    created_ts: datetime
    updated_ts: datetime
    entries: list[SpeciesListEntryOut]
    n_species: int
    n_unresolved: int
    n_registered: int
    n_trained: int


class BatchRunRequest(ApiModel):
    force_retrain: bool = False
    only_registered: bool = Field(
        False, description="Skip species not yet in the registry (no first-time downloads)"
    )


# -------------------------------------------------------- bias studies (B1, B9)
class TargetGroupOptionOut(ApiModel):
    option: str
    rank: str
    match: bool
    error: str | None = None
    background_method: str | None = None
    fallback_reason: str | None = None
    target_group_name: str | None = None
    target_group_taxon_key: int | None = None
    target_group_records: int | None = None
    matched: dict[str, Any] | None = None
    n_presence: int | None = None
    cv_auc_mean: float | None = None
    cv_cbi_mean: float | None = None
    independent: list[dict[str, Any]] = Field(default_factory=list)


class TargetGroupComparisonOut(ApiModel):
    comparison_id: str
    taxon_key: int
    created_ts: str
    bioclim_version: str
    configured_rank: str
    options: list[TargetGroupOptionOut]
    best_by_cv_cbi: str | None = None
    criterion: str
    note: str
    reference: str


class SharedBiasSpeciesOut(ApiModel):
    taxon_key: int
    scientific_name: str | None = None
    n_presences: int
    n_quadrature: int


class SharedBiasOut(ApiModel):
    estimate_id: str
    created_ts: str
    target_group: dict[str, Any]
    coefficients: dict[str, float] = Field(description="Per raw unit of each bias column")
    standardised_coefficients: dict[str, float]
    converged: bool
    iterations: int
    neg_log_likelihood: float
    l2_penalty: float
    species: list[SharedBiasSpeciesOut]
    skipped: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[str]
    static_layers: list[dict[str, Any]] = Field(default_factory=list)
    effort_query: str
    effort_pixel_deg: float
    quadrature: str
    method: str
    reference: str
    use: str
