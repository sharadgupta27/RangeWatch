"""Expert record annotations (U11): append-only; the latest per record and kind applies to
the next training run (left out / kept / relabelled) and is snapshotted with that version."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.auth import Curator, CurrentUser
from src.api.deps import repository_dep
from src.api.schemas import AnnotationIn, AnnotationOut, AnnotationResult
from src.domain import RecordAnnotation, SpeciesStatus
from src.features.annotations import validate_annotation
from src.persistence.repository import Repository

router = APIRouter(prefix="/species/{taxon_key}/annotations", tags=["annotations"])
RepoDep = Annotated[Repository, Depends(repository_dep)]


def _species_or_404(repo: Repository, taxon_key: int):
    sp = repo.get_species(taxon_key)
    if sp is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Species {taxon_key} not in registry")
    return sp


@router.get("", response_model=list[AnnotationOut], operation_id="listAnnotations")
def list_annotations(
    taxon_key: int,
    repo: RepoDep,
    source: Literal["GBIF", "iNaturalist"] | None = None,
    record_id: int | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> list[AnnotationOut]:
    """Annotation history, newest first (optionally for one record)."""
    _species_or_404(repo, taxon_key)
    anns = [
        a
        for a in repo.list_annotations(taxon_key)
        if (source is None or a.source == source)
        and (record_id is None or a.record_id == record_id)
    ]
    return [AnnotationOut.model_validate(a) for a in reversed(anns)][:limit]


@router.post(
    "",
    response_model=AnnotationResult,
    status_code=status.HTTP_201_CREATED,
    operation_id="createAnnotation",
    dependencies=[Curator],
)
def create_annotation(
    taxon_key: int, body: AnnotationIn, repo: RepoDep, user: CurrentUser
) -> AnnotationResult:
    """Annotate one GBIF / iNaturalist record (curator). A correction is a newer annotation;
    nothing is ever overwritten. Flags a retrain when the species has a model."""
    sp = _species_or_404(repo, taxon_key)
    try:
        value = validate_annotation(body.kind, body.value)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    occ = repo.load_occurrences(taxon_key)
    exists = (
        (occ["source"] == body.source) & (occ["record_id"].astype("int64") == body.record_id)
    ).any()
    if not exists:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No {body.source} record {body.record_id} for this species"
        )
    ann = repo.add_annotation(
        RecordAnnotation(
            taxon_key=taxon_key,
            source=body.source,
            record_id=body.record_id,
            kind=body.kind,
            value=value,
            note=body.note,
            author=user.name,
        )
    )
    flagged = sp.model_version > 0
    if flagged:
        repo.update_species(taxon_key, retrain_needed=True, status=SpeciesStatus.RETRAIN_PENDING)
    return AnnotationResult(annotation=AnnotationOut.model_validate(ann), retrain_flagged=flagged)
