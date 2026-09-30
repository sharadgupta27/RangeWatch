"""Validation-suite report (workplan phase 11) and on-demand runs."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.converters import job_out
from src.api.deps import JobQueue, artifacts_dep, queue_dep, repository_dep
from src.api.schemas import JobOut, ValidationReport, ValidationRunRequest
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
