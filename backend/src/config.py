"""Runtime configuration. All secrets come from environment variables / a git-ignored .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SDM_", env_file=".env", extra="ignore")

    # --- persistence ---
    database_url: str = "postgresql+psycopg2://sdm:sdm@localhost:5432/sdm"
    redis_url: str = "redis://localhost:6379/0"
    data_root: Path = Path(__file__).resolve().parents[2] / "data"
    artifact_root: Path = Path(__file__).resolve().parents[2] / "artifacts"
    mlflow_tracking_uri: str | None = None

    # --- global static layer ---
    bioclim_version: str = "bioclim_v1"
    # Independent climate stack (e.g. CHELSA built with `bioclim_store build-chelsa`) that
    # models can be refitted on for the climate-data cross-check; None disables it.
    crosscheck_bioclim_version: str | None = None
    # Upper bound on the cells of one regional high-resolution projection (~4000 × 4000).
    hires_max_cells: int = 16_000_000

    # --- external APIs (credentials only via env) ---
    gbif_user: str | None = None
    gbif_pwd: str | None = None
    gbif_email: str | None = None
    gbif_download_poll_seconds: int = 30
    gbif_download_timeout_seconds: int = 3 * 60 * 60
    gbif_search_max_records: int = 100_000
    inat_enabled: bool = True
    inat_max_records: int = 20_000
    coordinate_uncertainty_max_m: float = 10_000.0

    # --- tile servers (URLs as reachable from the browser) ---
    titiler_public_url: str = "http://localhost:8001"
    tileserv_public_url: str = "http://localhost:7800"
    # How the artifact root is mounted inside the titiler container.
    titiler_artifact_root: str = "/data/artifacts"

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    # Public base URL of this API (download links in the alert feed, STAC items, ODMAP).
    api_public_url: str = "http://localhost:8000"

    # --- retrain trigger (documented, reviewed decision; see CLAUDE.md) ---
    retrain_delta_fraction: float = 0.05
    envelope_min_outside_points: int = 1
    # B10 effort-aware trigger — PROPOSED, pending review (docs/adr/0001-effort-aware-
    # retrain-trigger.md). "envelope" (default) is the reviewed contract; "effort_aware"
    # counts only records in cells the target group had already sampled (real expansions),
    # not discoveries in previously unsampled cells. Both classes are always reported.
    retrain_trigger_mode: Literal["envelope", "effort_aware"] = "envelope"
    # A cell counts as previously sampled with ≥ this many target-group GBIF records.
    effort_sampled_min_records: int = 10

    # --- modeling defaults ---
    random_seed: int = 42
    # "target_group" (GBIF effort of the species' `target_group_rank`; falls back to buffer
    # when too sparse) or "buffer". Changing either changes model results: log + review.
    background_method: str = "target_group"
    target_group_rank: str = "order"
    # How target-group cells are drawn: "unique_cells" (each cell with effort once),
    # "record_count" (bias file: ∝ the group's records per cell) or "sqrt_record_count".
    target_group_weighting: str = "unique_cells"
    # B1: restrict target-group effort to the presences' basis of record / 5–95 % year window.
    target_group_match_source: bool = False
    target_group_match_years: bool = False
    # Thinning: distance (fixed SDM_THIN_KM, default) | autocorrelation (B3: variogram range,
    # clamped to [thin_km, thin_max_km]) | environmental (B4: one record per environmental
    # bin). SDM_THIN_REPS > 1 = maximal-retention thinning (spThin). Logged per version.
    thin_method: str = "distance"
    thin_km: float = 10.0
    thin_reps: int = 1
    thin_max_km: float = 50.0
    env_thin_bins: int = 20
    # Global target-group sampling-effort raster per model version (map layer + per-zone
    # "unsampled" shares). Costs ~32 GBIF map-tile requests per training run.
    effort_layer_enabled: bool = True
    # Sampling-bias handling (features/sampling_bias.py); off by default, logged per version.
    # Presence observation weights: none | quality_tier | density | quality_tier_density.
    presence_weighting: str = "none"
    # Bias covariate: none | target_group_effort (observer effort as a predictor, held at
    # `bias_condition` — max | p95 — for prediction; the background is then uniform).
    bias_covariate: str = "none"
    bias_condition: str = "max"
    # Static bias rasters registered with `bioclim_store add-bias-layer` (travel_time, …),
    # added as covariates and held at their most-effort end for prediction.
    bias_static_layers: list[str] = Field(default_factory=list)
    # Threshold for the binary zones: p10 | max_tss | equal_sens_spec | min_training_presence
    # | fixed (with SDM_THRESHOLD_VALUE). Every rule is computed and logged per version.
    threshold_rule: str = "p10"
    threshold_value: float | None = None
    # Algorithm: maxent | ensemble (MaxEnt + GLM/GAM/RF/BRT with pseudo-absences) |
    # integrated (presence-only + survey presence–absence data; experimental, see
    # docs/adr/0002-integrated-sdm.md). Logged per version.
    algorithm: str = "maxent"
    # ppm (B8 / B9): offset (log effort, coefficient 1) | estimated | shared (the target
    # group's joint estimate, `POST /species/{key}/shared-bias`; falls back to offset).
    ppm_bias: str = "offset"
    shared_bias_min_species: int = 3
    ensemble_members: list[str] = Field(
        default_factory=lambda: ["maxent", "glm", "gam", "rf", "brt"]
    )
    # Pseudo-absences: background | disk | environmental_profile | completeness
    pseudo_absence_strategy: str = "disk"
    n_pseudo_absences: int = 5_000
    pa_min_distance_km: float = 50.0
    completeness_min_records: float = 50.0
    # Limiting-factor raster (one extra prediction per predictor during projection).
    limiting_factor_enabled: bool = True
    # Per-version check against inferred absences (well-surveyed cells of the native range
    # without the species; §3.2). One regional GBIF effort request per training run.
    inferred_absence_check: bool = True
    # Survey-priority raster ("where to look next", needs the effort layer).
    priority_layer_enabled: bool = True

    # --- batch runs over species lists (U9): most species enqueued by one request ---
    batch_max_species: int = 200
    # Most names resolved against the GBIF backbone in one list request.
    list_max_names: int = 200

    # --- data quality (features/quality.py); tests and thresholds are logged per version ---
    quality_reference_version: str = "quality_refs_v1"
    # Tests whose flags exclude a record from training; None = every "error" test.
    quality_exclude: list[str] | None = None
    quality_baseline_start_year: int = 1970

    # --- reference (evaluation-only) datasets ---
    upload_max_bytes: int = 25 * 1024 * 1024
    upload_max_rows: int = 200_000
    # An uploaded record duplicates a GBIF/iNat record if one lies within this distance
    # (and, when both are dated, within `overlap_max_days`).
    overlap_distance_m: float = 100.0
    overlap_max_days: int = 1
    gbif_absences_enabled: bool = True

    # --- early-detection alerts (U10) ---
    alerts_enabled: bool = True
    # A new record farther than this from every known record (and outside the native range)
    # raises an "outside known range" alert.
    alert_known_range_km: float = 100.0
    # Webhook targets must be https and resolve to public addresses unless allowed here.
    alert_webhook_allow_http: bool = False
    alert_webhook_allowed_hosts: list[str] | None = None
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str = "rangewatch@localhost"
    smtp_starttls: bool = True

    # --- users and roles (U14) ---
    # "none": single-user deployment, every request acts as a curator (local development).
    # "header": an identity-aware proxy (oauth2-proxy, Pomerium, …) in front of the API
    # authenticates users and sets the headers below; the API must only be reachable through
    # it, and the proxy must strip these headers from client requests.
    auth_mode: Literal["none", "header"] = "none"
    auth_user_header: str = "X-Auth-Request-User"
    auth_groups_header: str = "X-Auth-Request-Groups"
    # Identity-provider group → role (viewer < analyst < curator).
    auth_role_groups: dict[str, str] = Field(
        default_factory=lambda: {
            "rangewatch-curators": "curator",
            "rangewatch-analysts": "analyst",
        }
    )
    auth_default_role: Literal["viewer", "analyst", "curator"] = "viewer"

    # --- storage growth: species without a user request for this long get their feature
    # table archived (zstd-recompressed); 0 disables. ---
    archive_inactive_after_days: int = 180

    @property
    def bioclim_root(self) -> Path:
        return self.data_root / "bioclim"

    @property
    def quality_reference_root(self) -> Path:
        return self.data_root / "reference"

    @property
    def native_range_root(self) -> Path:
        return self.data_root / "native_range_polygons"

    @property
    def gbif_download_enabled(self) -> bool:
        return bool(self.gbif_user and self.gbif_pwd and self.gbif_email)


@lru_cache
def get_settings() -> Settings:
    return Settings()
