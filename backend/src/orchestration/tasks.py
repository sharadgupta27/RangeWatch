"""Celery tasks wrapping the Prefect flows and keeping the `jobs` table up to date."""

from __future__ import annotations

import logging
import uuid
from datetime import timedelta
from typing import Any

from src.config import get_settings
from src.domain import JobStatus, SpeciesStatus
from src.orchestration.celery_app import celery_app
from src.orchestration.factory import get_repository
from src.persistence.repository import STALE_JOB_MESSAGE

log = logging.getLogger(__name__)


def _run_job(job_id: str, fn: Any, /, **kwargs: Any) -> dict[str, Any]:
    """Run `fn(**kwargs)` while tracking the job row. Positional-only so `kwargs` may itself
    carry a `job_id` for the flow's progress reporting."""
    jid = uuid.UUID(job_id)
    repo = get_repository()
    repo.update_job(jid, status=JobStatus.RUNNING, progress=0.01, message="Started")
    try:
        result = fn(**kwargs)
    except Exception as exc:
        log.exception("Job %s failed", job_id)
        repo.update_job(jid, status=JobStatus.FAILED, message=f"{type(exc).__name__}: {exc}")
        raise
    repo.update_job(
        jid, status=JobStatus.SUCCEEDED, progress=1.0, result=result, message="Completed"
    )
    return result


@celery_app.task(name="sdm.run_species_pipeline")
def run_species_pipeline(
    job_id: str, taxon_key: int, force_retrain: bool = False, full_resync: bool = False
) -> dict:
    from src.orchestration.flows import species_pipeline_flow

    return _run_job(
        job_id,
        species_pipeline_flow,
        taxon_key=taxon_key,
        force_retrain=force_retrain,
        full_resync=full_resync,
        job_id=job_id,
    )


@celery_app.task(name="sdm.generate_bulletin")
def generate_bulletin(job_id: str, taxon_key: int, model_version: int | None = None) -> dict:
    from src.orchestration.flows import bulletin_flow

    return _run_job(job_id, bulletin_flow, taxon_key=taxon_key, model_version=model_version)


@celery_app.task(name="sdm.project_scenarios")
def project_scenarios(job_id: str, taxon_key: int) -> dict:
    from src.orchestration.flows import scenarios_flow

    return _run_job(job_id, scenarios_flow, taxon_key=taxon_key, job_id=job_id)


@celery_app.task(name="sdm.climate_crosscheck")
def climate_crosscheck(job_id: str, taxon_key: int) -> dict:
    from src.orchestration.flows import crosscheck_flow

    return _run_job(job_id, crosscheck_flow, taxon_key=taxon_key, job_id=job_id)


@celery_app.task(name="sdm.project_hires")
def project_hires(job_id: str, taxon_key: int, bbox: list[float]) -> dict:
    from src.orchestration.flows import hires_flow

    return _run_job(job_id, hires_flow, taxon_key=taxon_key, bbox=bbox, job_id=job_id)


@celery_app.task(name="sdm.run_validation_suite")
def run_validation_suite(
    job_id: str, taxon_keys: list[int] | None = None, train: bool = True
) -> dict:
    from src.orchestration.flows import validation_flow

    return _run_job(job_id, validation_flow, taxon_keys=taxon_keys, train=train, job_id=job_id)


@celery_app.task(name="sdm.refresh_all_species")
def refresh_all_species() -> list:
    from src.orchestration.flows import refresh_all_species_flow

    return refresh_all_species_flow()


@celery_app.task(name="sdm.reap_stale_jobs")
def reap_stale_jobs(max_silence_minutes: int | None = None) -> int:
    """Fail jobs whose worker died without reporting (keeps the UI from polling forever).

    Pipelines report progress per asset step; the longest silent step is waiting on a GBIF
    download, so the default window is that timeout plus an hour of slack.
    """
    if max_silence_minutes is None:
        max_silence_minutes = get_settings().gbif_download_timeout_seconds // 60 + 60
    repo = get_repository()
    reaped = repo.fail_stale_jobs(timedelta(minutes=max_silence_minutes))
    for key in {k for k in reaped if k is not None}:
        # A lost job leaves the species mid-stage; surface that instead of "ingesting" forever.
        sp = repo.get_species(key)
        if sp is not None and sp.status in (SpeciesStatus.INGESTING, SpeciesStatus.TRAINING):
            repo.update_species(key, status=SpeciesStatus.FAILED, last_error=STALE_JOB_MESSAGE)
    if reaped:
        log.warning("Marked %d stale job(s) as failed", len(reaped))
    return len(reaped)


@celery_app.task(name="sdm.archive_inactive_species")
def archive_inactive_species() -> list[int]:
    """Archive feature tables of species not requested for SDM_ARCHIVE_INACTIVE_AFTER_DAYS."""
    from src.orchestration.factory import get_artifact_store
    from src.orchestration.maintenance import archive_inactive_features

    days = get_settings().archive_inactive_after_days
    if days <= 0:
        return []
    return archive_inactive_features(get_repository(), get_artifact_store(), timedelta(days=days))
