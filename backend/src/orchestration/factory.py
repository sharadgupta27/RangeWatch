"""Wires real dependencies (PostGIS, GBIF, iNaturalist, MLflow, bulletin) into the pipeline."""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache

from src.config import Settings, get_settings
from src.orchestration.pipeline import ProgressFn, SpeciesPipeline, StepRunner, direct_runner
from src.persistence.artifact_store import ArtifactStore
from src.persistence.postgres_repository import PostgresRepository
from src.persistence.repository import Repository

log = logging.getLogger(__name__)


@lru_cache
def get_repository() -> Repository:
    return PostgresRepository.from_url(get_settings().database_url)


def get_artifact_store(settings: Settings | None = None) -> ArtifactStore:
    s = settings or get_settings()
    return ArtifactStore(s.artifact_root, s.titiler_artifact_root)


def build_pipeline(
    progress: ProgressFn | None = None, step: StepRunner = direct_runner
) -> SpeciesPipeline:
    from src.bulletin.generator import BulletinGenerator
    from src.connectors.gbif_client import GbifClient
    from src.connectors.inaturalist_client import InatClient
    from src.modeling.maxent_trainer import TrainingConfig
    from src.persistence.model_registry import MlflowModelRegistry

    s = get_settings()
    repo = get_repository()
    artifacts = get_artifact_store(s)
    pipeline = SpeciesPipeline(
        repo=repo,
        gbif=GbifClient(
            user=s.gbif_user,
            pwd=s.gbif_pwd,
            email=s.gbif_email,
            max_uncertainty_m=s.coordinate_uncertainty_max_m,
            poll_seconds=s.gbif_download_poll_seconds,
            timeout_seconds=s.gbif_download_timeout_seconds,
            search_max_records=s.gbif_search_max_records,
        ),
        inat=(
            InatClient(s.coordinate_uncertainty_max_m, s.inat_max_records)
            if s.inat_enabled
            else None
        ),
        artifacts=artifacts,
        registry=MlflowModelRegistry(s.mlflow_tracking_uri),
        settings=s,
        bulletin=BulletinGenerator(repo, artifacts, s),
        # SDM_RANDOM_SEED drives thinning, background sampling and CV folds (logged per version).
        training_config=TrainingConfig(
            seed=s.random_seed,
            background_method=s.background_method,
            target_group_rank=s.target_group_rank,
            target_group_weighting=s.target_group_weighting,
            target_group_match_source=s.target_group_match_source,
            target_group_match_years=s.target_group_match_years,
            thin_method=s.thin_method,
            thin_km=s.thin_km,
            thin_reps=s.thin_reps,
            thin_max_km=s.thin_max_km,
            env_thin_bins=s.env_thin_bins,
            bias_static_layers=tuple(s.bias_static_layers),
            ppm_bias=s.ppm_bias,
            presence_weighting=s.presence_weighting,
            bias_covariate=s.bias_covariate,
            bias_condition=s.bias_condition,
            threshold_rule=s.threshold_rule,
            threshold_value=s.threshold_value,
            algorithm=s.algorithm,
            ensemble_members=tuple(s.ensemble_members),
            pseudo_absence_strategy=s.pseudo_absence_strategy,
            n_pseudo_absences=s.n_pseudo_absences,
            pa_min_distance_km=s.pa_min_distance_km,
            completeness_min_records=s.completeness_min_records,
            limiting_factor=s.limiting_factor_enabled,
        ),
        step=step,
    )
    if progress is not None:
        pipeline.progress = progress
    return pipeline


def job_progress(job_id: str | None) -> ProgressFn | None:
    """Progress callback that writes stage/progress into the `jobs` table."""
    if not job_id:
        return None
    jid = uuid.UUID(job_id)
    repo = get_repository()

    def report(stage: str, fraction: float, message: str | None) -> None:
        try:
            repo.update_job(
                jid, stage=stage, progress=float(min(max(fraction, 0), 1)), message=message
            )
        except Exception:  # progress reporting must never break the job
            log.exception("Failed to report progress for job %s", job_id)

    return report
