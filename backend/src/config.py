"""Runtime configuration. All secrets come from environment variables / a git-ignored .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

    # --- retrain trigger (documented, reviewed decision; see CLAUDE.md) ---
    retrain_delta_fraction: float = 0.05
    envelope_min_outside_points: int = 1

    # --- modeling defaults ---
    random_seed: int = 42
    # "target_group" (GBIF effort of the species' `target_group_rank`; falls back to buffer
    # when too sparse) or "buffer". Changing either changes model results: log + review.
    background_method: str = "target_group"
    target_group_rank: str = "order"
    # How target-group cells are drawn: "unique_cells" (each cell with effort once),
    # "record_count" (bias file: ∝ the group's records per cell) or "sqrt_record_count".
    target_group_weighting: str = "unique_cells"
    # Global target-group sampling-effort raster per model version (map layer + per-zone
    # "unsampled" shares). Costs ~32 GBIF map-tile requests per training run.
    effort_layer_enabled: bool = True

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
