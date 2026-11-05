"""Reference datasets: records kept apart from the GBIF/iNaturalist occurrence store. Users
upload surveys (e.g. agency field records) either as independent evaluation data (default) or
as extra training data; explicit GBIF absences are added by the pipeline (evaluation only).
Parsing happens in the request (bounded size); the independence check and the scoring of every
model version run as a background job. Adding or removing a training upload flags the species
for retraining (a user edit, like the native-range polygon; the retrain threshold is
unchanged)."""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from src.api.auth import Analyst
from src.api.converters import job_out, reference_dataset_out
from src.api.deps import JobQueue, queue_dep, repository_dep, settings_dep
from src.api.schemas import DatasetUploadResult, JobOut, ReferenceDatasetOut
from src.config import Settings
from src.connectors.upload_client import ColumnMapping, UploadError, parse_upload
from src.domain import Job, JobKind, ReferenceDataset, SpeciesStatus
from src.persistence.repository import Repository

router = APIRouter(prefix="/species/{taxon_key}/datasets", tags=["datasets"])
RepoDep = Annotated[Repository, Depends(repository_dep)]
QueueDep = Annotated[JobQueue, Depends(queue_dep)]
OptionalForm = Annotated[str | None, Form()]


def _species_or_404(repo: Repository, taxon_key: int):
    sp = repo.get_species(taxon_key)
    if sp is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Species {taxon_key} not in registry")
    return sp


def _flag_retrain(repo: Repository, taxon_key: int) -> None:
    """Training data changed: the current model (if any) no longer reflects it."""
    sp = repo.get_species(taxon_key)
    if sp is not None and sp.model_version > 0:
        repo.update_species(taxon_key, retrain_needed=True, status=SpeciesStatus.RETRAIN_PENDING)


def _enqueue(repo: Repository, queue: JobQueue, taxon_key: int) -> Job:
    job = repo.create_job(Job(kind=JobKind.EVALUATION, taxon_key=taxon_key, message="Queued"))
    queue.enqueue_evaluation(str(job.job_id), taxon_key)
    return job


@router.get("", response_model=list[ReferenceDatasetOut], operation_id="listReferenceDatasets")
def list_datasets(taxon_key: int, repo: RepoDep) -> list[ReferenceDatasetOut]:
    """Reference datasets with each model version's independent evaluation on them."""
    _species_or_404(repo, taxon_key)
    versions = repo.list_model_versions(taxon_key)
    return [reference_dataset_out(ds, versions) for ds in repo.list_reference_datasets(taxon_key)]


@router.post(
    "",
    response_model=DatasetUploadResult,
    status_code=status.HTTP_201_CREATED,
    operation_id="uploadReferenceDataset",
    dependencies=[Analyst],
)
def upload_dataset(
    taxon_key: int,
    repo: RepoDep,
    queue: QueueDep,
    file: Annotated[
        UploadFile, File(description="CSV/TSV, GeoJSON, GeoPackage or zipped shapefile")
    ],
    name: Annotated[str, Form(min_length=1, max_length=200)],
    role: Annotated[
        Literal["evaluation", "training"],
        Form(
            description="evaluation (default): independent test data, never trained on. "
            "training: presences join the next model's training set"
        ),
    ] = "evaluation",
    citation: OptionalForm = None,
    license: OptionalForm = None,
    note: OptionalForm = None,
    longitude_column: OptionalForm = None,
    latitude_column: OptionalForm = None,
    species_column: OptionalForm = None,
    date_column: OptionalForm = None,
    uncertainty_column: OptionalForm = None,
    status_column: OptionalForm = None,
    record_id_column: OptionalForm = None,
    settings: Settings = Depends(settings_dep),
) -> DatasetUploadResult:
    """Upload a survey as an evaluation dataset (default) or as training data. Columns are
    detected unless named; rows of other species are dropped when the file has a species
    column. A training upload flags the species for retraining."""
    sp = _species_or_404(repo, taxon_key)
    content = file.file.read(settings.upload_max_bytes + 1)
    if len(content) > settings.upload_max_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds {settings.upload_max_bytes // (1024 * 1024)} MB",
        )
    dataset_id = f"upload-{uuid.uuid4().hex[:12]}"
    try:
        parsed = parse_upload(
            content,
            file.filename or "upload.csv",
            taxon_key,
            dataset_id,
            [n for n in (sp.canonical_name, sp.scientific_name) if n],
            ColumnMapping(
                longitude=longitude_column or None,
                latitude=latitude_column or None,
                species=species_column or None,
                date=date_column or None,
                uncertainty=uncertainty_column or None,
                status=status_column or None,
                record_id=record_id_column or None,
            ),
            max_rows=settings.upload_max_rows,
        )
    except UploadError as exc:
        raise HTTPException(422, str(exc)) from exc
    ds = repo.save_reference_dataset(
        ReferenceDataset(
            taxon_key=taxon_key,
            dataset_id=dataset_id,
            kind="upload",
            role=role,
            name=name.strip(),
            citation=citation,
            license=license,
            note=note,
            file_name=file.filename,
            file_sha256=parsed.file_sha256,
            parse_report=parsed.report,
            n_present=parsed.n_present,
            n_absent=parsed.n_absent,
        )
    )
    repo.append_reference_records(parsed.records)
    if role == "training":
        _flag_retrain(repo, taxon_key)
    job = _enqueue(repo, queue, taxon_key)
    return DatasetUploadResult(dataset=reference_dataset_out(ds, []), job=job_out(job))


@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="deleteReferenceDataset",
    dependencies=[Analyst],
)
def delete_dataset(taxon_key: int, dataset_id: str, repo: RepoDep) -> None:
    """Remove an uploaded dataset and its records. Scores already stored with model versions
    stay in their lineage; removing a training upload flags the species for retraining. GBIF
    absences are managed by the pipeline and cannot be removed."""
    _species_or_404(repo, taxon_key)
    ds = repo.get_reference_dataset(taxon_key, dataset_id)
    if ds is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    if ds.kind != "upload":
        raise HTTPException(status.HTTP_409_CONFLICT, "Only uploaded datasets can be removed")
    repo.delete_reference_dataset(taxon_key, dataset_id)
    if ds.role == "training":
        _flag_retrain(repo, taxon_key)


@router.post(
    "/evaluate",
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="evaluateReferenceDatasets",
    dependencies=[Analyst],
)
def evaluate(taxon_key: int, repo: RepoDep, queue: QueueDep) -> JobOut:
    """Re-run the independence check and score every model version on every dataset."""
    _species_or_404(repo, taxon_key)
    return job_out(_enqueue(repo, queue, taxon_key))
