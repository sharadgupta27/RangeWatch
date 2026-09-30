"""Bulletin generation (async job) and download."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from src.api.converters import job_out
from src.api.deps import JobQueue, artifacts_dep, queue_dep, repository_dep
from src.api.schemas import BulletinRequest, BulletinStatus, JobOut
from src.domain import Job, JobKind
from src.persistence.artifact_store import ArtifactStore
from src.persistence.repository import Repository

router = APIRouter(prefix="/species/{taxon_key}/bulletin", tags=["bulletin"])
RepoDep = Annotated[Repository, Depends(repository_dep)]
ArtifactsDep = Annotated[ArtifactStore, Depends(artifacts_dep)]


def _resolve_version(repo: Repository, taxon_key: int, model_version: int | None) -> int:
    sp = repo.get_species(taxon_key)
    if sp is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Species {taxon_key} not in registry")
    version = model_version or sp.model_version
    if not version:
        raise HTTPException(status.HTTP_409_CONFLICT, "No trained model yet")
    return version


@router.post(
    "", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED, operation_id="generateBulletin"
)
def generate(
    taxon_key: int,
    repo: RepoDep,
    queue: Annotated[JobQueue, Depends(queue_dep)],
    body: BulletinRequest | None = None,
) -> JobOut:
    version = _resolve_version(repo, taxon_key, body.model_version if body else None)
    job = repo.create_job(Job(kind=JobKind.BULLETIN, taxon_key=taxon_key, message="Queued"))
    queue.enqueue_bulletin(str(job.job_id), taxon_key, version)
    return job_out(job)


@router.get("", response_model=BulletinStatus, operation_id="getBulletinStatus")
def bulletin_status(
    taxon_key: int, repo: RepoDep, artifacts: ArtifactsDep, model_version: int | None = None
) -> BulletinStatus:
    sp = repo.get_species(taxon_key)
    if sp is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Species {taxon_key} not in registry")
    version = model_version or sp.model_version or None
    if not version:
        return BulletinStatus(
            taxon_key=taxon_key, model_version=None, pdf_available=False, html_available=False
        )
    vdir = artifacts.local_path(str(artifacts.version_dir(taxon_key, version)))
    pdf, html = (vdir / "bulletin.pdf").exists(), (vdir / "bulletin.html").exists()
    base = f"/species/{taxon_key}/bulletin/file"
    return BulletinStatus(
        taxon_key=taxon_key,
        model_version=version,
        pdf_available=pdf,
        html_available=html,
        pdf_url=f"{base}?format=pdf&model_version={version}" if pdf else None,
        html_url=f"{base}?format=html&model_version={version}" if html else None,
    )


@router.get(
    "/file",
    response_class=FileResponse,
    operation_id="downloadBulletin",
    responses={200: {"content": {"application/pdf": {}, "text/html": {}}}},
)
def download(
    taxon_key: int,
    repo: RepoDep,
    artifacts: ArtifactsDep,
    format: Literal["pdf", "html"] = "pdf",
    model_version: int | None = None,
) -> FileResponse:
    version = _resolve_version(repo, taxon_key, model_version)
    path = (
        artifacts.local_path(str(artifacts.version_dir(taxon_key, version))) / f"bulletin.{format}"
    )
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Bulletin not generated yet")
    sp = repo.get_species(taxon_key)
    name = (sp.canonical_name or sp.scientific_name if sp else str(taxon_key)).replace(" ", "_")
    return FileResponse(
        path,
        media_type="application/pdf" if format == "pdf" else "text/html",
        filename=f"bulletin_{name}_v{version}.{format}",
    )
