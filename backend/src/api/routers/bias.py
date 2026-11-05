"""Sampling-bias studies of a species (B1 target-group comparison, B9 shared bias)."""

from __future__ import annotations

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import Analyst
from src.api.converters import job_out
from src.api.deps import (
    JobQueue,
    TaxonSearch,
    artifacts_dep,
    gbif_search_dep,
    queue_dep,
    repository_dep,
    settings_dep,
)
from src.api.schemas import JobOut, SharedBiasOut, TargetGroupComparisonOut
from src.config import Settings
from src.domain import Job, JobKind
from src.persistence.artifact_store import ArtifactStore
from src.persistence.repository import Repository

log = logging.getLogger(__name__)
router = APIRouter(prefix="/species/{taxon_key}", tags=["bias"])
RepoDep = Annotated[Repository, Depends(repository_dep)]
QueueDep = Annotated[JobQueue, Depends(queue_dep)]
ArtifactsDep = Annotated[ArtifactStore, Depends(artifacts_dep)]
SettingsDep = Annotated[Settings, Depends(settings_dep)]
GbifDep = Annotated[TaxonSearch, Depends(gbif_search_dep)]

# Same artifact paths as src/orchestration/bias_studies.py (not imported: it pulls in the
# modelling stack, which the API process must not load).
TARGET_GROUP_COMPARISON = "target_group_comparisons/{taxon_key}.json"
SHARED_BIAS = "shared_bias/{rank}_{group_key}.json"


def _species(repo: Repository, taxon_key: int) -> None:
    if repo.get_species(taxon_key) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Species {taxon_key} not in registry")


def _read(artifacts: ArtifactStore, rel: str) -> dict | None:
    path = artifacts.local_path(rel)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@router.post(
    "/target-group-comparison",
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="runTargetGroupComparison",
    dependencies=[Analyst],
)
def run_comparison(taxon_key: int, repo: RepoDep, queue: QueueDep) -> JobOut:
    """B1: refit with order / class / phylum target groups (and matched source + years)."""
    _species(repo, taxon_key)
    nr = repo.get_native_range(taxon_key)
    if nr is None or nr.status != "confirmed":
        raise HTTPException(status.HTTP_409_CONFLICT, "Confirm the native range first")
    job = repo.create_job(Job(kind=JobKind.TARGET_GROUPS, taxon_key=taxon_key, message="Queued"))
    queue.enqueue_target_groups(str(job.job_id), taxon_key)
    return job_out(job)


@router.get(
    "/target-group-comparison",
    response_model=TargetGroupComparisonOut,
    operation_id="getTargetGroupComparison",
)
def get_comparison(
    taxon_key: int, repo: RepoDep, artifacts: ArtifactsDep
) -> TargetGroupComparisonOut:
    _species(repo, taxon_key)
    report = _read(artifacts, TARGET_GROUP_COMPARISON.format(taxon_key=taxon_key))
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No target-group comparison yet")
    return TargetGroupComparisonOut.model_validate(report)


@router.post(
    "/shared-bias",
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="runSharedBias",
    dependencies=[Analyst],
)
def run_shared_bias(taxon_key: int, repo: RepoDep, queue: QueueDep) -> JobOut:
    """B9: estimate the observer bias jointly from every registered species of this
    species' target group (used by SDM_ALGORITHM=ppm with SDM_PPM_BIAS=shared)."""
    _species(repo, taxon_key)
    job = repo.create_job(Job(kind=JobKind.SHARED_BIAS, taxon_key=taxon_key, message="Queued"))
    queue.enqueue_shared_bias(str(job.job_id), taxon_key)
    return job_out(job)


@router.get("/shared-bias", response_model=SharedBiasOut, operation_id="getSharedBias")
def get_shared_bias(
    taxon_key: int,
    repo: RepoDep,
    artifacts: ArtifactsDep,
    gbif: GbifDep,
    settings: SettingsDep,
) -> SharedBiasOut:
    """The shared bias estimate of the species' target group, if one was made."""
    _species(repo, taxon_key)
    rank = settings.target_group_rank.lower()
    try:
        group = gbif.get_taxon(taxon_key).higher_taxon(rank)
    except Exception as exc:
        log.warning("GBIF taxon lookup failed for %s: %s", taxon_key, exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "GBIF taxonomy unavailable") from exc
    if group is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"The species has no {rank}")
    report = _read(artifacts, SHARED_BIAS.format(rank=rank, group_key=int(group[0])))
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No shared bias estimate yet")
    return SharedBiasOut.model_validate(report)
