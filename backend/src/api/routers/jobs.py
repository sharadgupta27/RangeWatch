"""Async job status: a live Server-Sent Events stream, with plain endpoints for polling."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from src.api.converters import job_out
from src.api.deps import repository_dep
from src.api.schemas import JobOut
from src.domain import JobStatus
from src.persistence.repository import Repository

router = APIRouter(prefix="/jobs", tags=["jobs"])
RepoDep = Annotated[Repository, Depends(repository_dep)]

# The stream re-reads job rows server-side (workers write progress to the database) and
# pushes a snapshot only when it changed. Connections are recycled so proxies never see an
# idle or endless response; EventSource reconnects on its own after `retry` ms.
SSE_ACTIVE_POLL_S = 1.0
SSE_IDLE_POLL_S = 3.0
SSE_KEEPALIVE_S = 15.0
SSE_MAX_SECONDS = 300.0
SSE_RETRY_MS = 2_000
_ACTIVE = {JobStatus.QUEUED, JobStatus.RUNNING}


@router.get(
    "/events",
    operation_id="streamJobs",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "text/event-stream of `jobs` events; each `data:` line is a JSON "
            "array of JobOut (newest first, same as GET /jobs), sent on connect and on change",
            "content": {
                "text/event-stream": {
                    "schema": {"type": "array", "items": {"$ref": "#/components/schemas/JobOut"}}
                }
            },
        }
    },
)
async def stream_jobs(
    request: Request,
    repo: RepoDep,
    taxon_key: int | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> StreamingResponse:
    """Live job list for a species (or all jobs): replaces client polling while connected."""

    async def events() -> AsyncIterator[str]:
        yield f"retry: {SSE_RETRY_MS}\n\n"
        started = last_sent = time.monotonic()
        previous: str | None = None
        while True:
            jobs = await run_in_threadpool(repo.list_jobs, taxon_key, limit)
            payload = json.dumps(jsonable_encoder([job_out(j) for j in jobs]))
            now = time.monotonic()
            if payload != previous:
                yield f"event: jobs\ndata: {payload}\n\n"
                previous, last_sent = payload, now
            elif now - last_sent >= SSE_KEEPALIVE_S:
                yield ": keepalive\n\n"
                last_sent = now
            if now - started >= SSE_MAX_SECONDS or await request.is_disconnected():
                return
            active = any(j.status in _ACTIVE for j in jobs)
            await asyncio.sleep(SSE_ACTIVE_POLL_S if active else SSE_IDLE_POLL_S)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # X-Accel-Buffering: nginx must pass events through instead of buffering them.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
