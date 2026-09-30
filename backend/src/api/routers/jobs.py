"""Async job status (polled by the frontend via TanStack Query refetchInterval)."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.converters import job_out
from src.api.deps import repository_dep
from src.api.schemas import JobOut
from src.persistence.repository import Repository

router = APIRouter(prefix="/jobs", tags=["jobs"])
RepoDep = Annotated[Repository, Depends(repository_dep)]


@router.get("/{job_id}", response_model=JobOut, operation_id="getJob")
def get_job(job_id: uuid.UUID, repo: RepoDep) -> JobOut:
    job = repo.get_job(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found")
    return job_out(job)


@router.get("", response_model=list[JobOut], operation_id="listJobs")
def list_jobs(
    repo: RepoDep,
    taxon_key: int | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[JobOut]:
    return [job_out(j) for j in repo.list_jobs(taxon_key, limit)]
