"""Validation-suite report (workplan phase 11) and on-demand runs."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import Analyst
from src.api.converters import job_out
from src.api.deps import JobQueue, artifacts_dep, queue_dep, repository_dep
from src.api.schemas import (
    BenchmarkReport,
    BenchmarkRequest,
    JobOut,
    ValidationReport,
    ValidationRunRequest,
)
from src.domain import Job, JobKind
from src.persistence.artifact_store import ArtifactStore
from src.persistence.repository import Repository

router = APIRouter(prefix="/validation", tags=["validation"])


@router.get("", response_model=ValidationReport, operation_id="getValidationReport")
def latest_report(
    artifacts: Annotated[ArtifactStore, Depends(artifacts_dep)],
) -> ValidationReport:
    path = artifacts.local_path("validation/latest.json")
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The validation suite has not run yet")
    return ValidationReport.model_validate(json.loads(path.read_text(encoding="utf-8")))


@router.post(
    "/runs",
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="runValidationSuite",
    dependencies=[Analyst],
)
def run(
    repo: Annotated[Repository, Depends(repository_dep)],
    queue: Annotated[JobQueue, Depends(queue_dep)],
    body: ValidationRunRequest | None = None,
) -> JobOut:
    body = body or ValidationRunRequest()
    job = repo.create_job(Job(kind=JobKind.VALIDATION, taxon_key=None, message="Queued"))
    queue.enqueue_validation(str(job.job_id), body.taxon_keys, body.train)
    return job_out(job)


# ------------------------------------------------------- virtual-species benchmark (§4.6)
@router.get("/benchmark", response_model=BenchmarkReport, operation_id="getVirtualSpeciesBenchmark")
def latest_benchmark(
    artifacts: Annotated[ArtifactStore, Depends(artifacts_dep)],
) -> BenchmarkReport:
    """Latest virtual-species bias benchmark: which bias correction recovers a known niche."""
    path = artifacts.local_path("benchmarks/latest.json")
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The benchmark has not run yet")
    return BenchmarkReport.model_validate(json.loads(path.read_text(encoding="utf-8")))


@router.post(
    "/benchmark",
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="runVirtualSpeciesBenchmark",
    dependencies=[Analyst],
)
def run_benchmark(
    body: BenchmarkRequest,
    repo: Annotated[Repository, Depends(repository_dep)],
    queue: Annotated[JobQueue, Depends(queue_dep)],
) -> JobOut:
    if body.bias == "gbif_effort" and body.target_group_key is None:
        raise HTTPException(422, "bias 'gbif_effort' needs target_group_key")
    job = repo.create_job(Job(kind=JobKind.BENCHMARK, taxon_key=None, message="Queued"))
    queue.enqueue_benchmark(str(job.job_id), body.model_dump())
    return job_out(job)
