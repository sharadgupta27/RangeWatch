"""Species lists / watch lists (U9): create from names or GBIF keys, filter the registry,
run the pipeline for every species of a list as one batch job."""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import Analyst, CurrentUser
from src.api.converters import job_out
from src.api.deps import (
    JobQueue,
    TaxonSearch,
    gbif_search_dep,
    queue_dep,
    repository_dep,
    settings_dep,
)
from src.api.schemas import (
    BatchRunRequest,
    JobOut,
    SpeciesListEntryOut,
    SpeciesListIn,
    SpeciesListOut,
)
from src.config import Settings
from src.domain import Job, JobKind, SpeciesList, SpeciesListEntry, utcnow
from src.persistence.repository import NotFoundError, Repository

log = logging.getLogger(__name__)
router = APIRouter(prefix="/lists", tags=["lists"])
RepoDep = Annotated[Repository, Depends(repository_dep)]
QueueDep = Annotated[JobQueue, Depends(queue_dep)]
GbifDep = Annotated[TaxonSearch, Depends(gbif_search_dep)]
SettingsDep = Annotated[Settings, Depends(settings_dep)]
UNPROCESSABLE = 422  # Starlette renamed the constant; the code is what the contract fixes


def _resolve(body: SpeciesListIn, gbif: TaxonSearch, repo: Repository) -> list[SpeciesListEntry]:
    """GBIF backbone match of each name; keys are taken as given (named from the registry
    or the backbone). Unmatched names stay in the list, unresolved, so nothing is dropped."""
    entries: list[SpeciesListEntry] = []
    seen: set[str] = set()
    for raw in body.names:
        name = " ".join(raw.split())
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        try:
            match = gbif.match_name(name)
        except Exception as exc:  # a lookup failure leaves the entry unresolved
            log.warning("GBIF name match failed for %r: %s", name, exc)
            match = None
        if match is None:
            entries.append(SpeciesListEntry(query=name, match_type="NONE"))
            continue
        taxon, match_type, confidence = match
        entries.append(
            SpeciesListEntry(
                query=name,
                taxon_key=taxon.taxon_key,
                scientific_name=taxon.scientific_name,
                match_type=match_type,
                confidence=confidence,
            )
        )
    for key in body.taxon_keys:
        if str(key) in seen:
            continue
        seen.add(str(key))
        sp = repo.get_species(key)
        name = sp.scientific_name if sp else None
        if name is None:
            try:
                name = gbif.get_taxon(key).scientific_name
            except Exception as exc:
                log.warning("GBIF taxon lookup failed for %s: %s", key, exc)
        entries.append(
            SpeciesListEntry(query=str(key), taxon_key=key, scientific_name=name, match_type="KEY")
        )
    return entries


def _out(sl: SpeciesList, repo: Repository) -> SpeciesListOut:
    entries = []
    for e in sl.entries:
        sp = repo.get_species(e.taxon_key) if e.taxon_key is not None else None
        entries.append(
            SpeciesListEntryOut(
                query=e.query,
                taxon_key=e.taxon_key,
                scientific_name=e.scientific_name,
                match_type=e.match_type,
                confidence=e.confidence,
                registered=sp is not None,
                status=str(sp.status) if sp else None,
                model_version=(sp.model_version or None) if sp else None,
            )
        )
    return SpeciesListOut(
        list_id=str(sl.list_id),
        name=sl.name,
        description=sl.description,
        source=sl.source,
        created_by=sl.created_by,
        created_ts=sl.created_ts,
        updated_ts=sl.updated_ts,
        entries=entries,
        n_species=len(set(sl.taxon_keys)),
        n_unresolved=sum(1 for e in sl.entries if e.taxon_key is None),
        n_registered=len({e.taxon_key for e in entries if e.registered}),
        n_trained=len({e.taxon_key for e in entries if e.model_version}),
    )


def _get(repo: Repository, list_id: uuid.UUID) -> SpeciesList:
    sl = repo.get_species_list(list_id)
    if sl is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Species list not found")
    return sl


def _check_size(body: SpeciesListIn, settings: Settings) -> None:
    if not body.names and not body.taxon_keys:
        raise HTTPException(UNPROCESSABLE, "Give scientific names or GBIF taxon keys")
    if len(body.names) + len(body.taxon_keys) > settings.list_max_names:
        raise HTTPException(
            UNPROCESSABLE,
            f"At most {settings.list_max_names} names / keys per request",
        )


@router.get("", response_model=list[SpeciesListOut], operation_id="listSpeciesLists")
def list_lists(repo: RepoDep) -> list[SpeciesListOut]:
    return [_out(sl, repo) for sl in repo.list_species_lists()]


@router.post(
    "",
    response_model=SpeciesListOut,
    status_code=status.HTTP_201_CREATED,
    operation_id="createSpeciesList",
    dependencies=[Analyst],
)
def create_list(
    body: SpeciesListIn, repo: RepoDep, gbif: GbifDep, settings: SettingsDep, user: CurrentUser
) -> SpeciesListOut:
    """Names are matched against the GBIF backbone (unmatched ones are kept, unresolved)."""
    _check_size(body, settings)
    sl = SpeciesList(
        name=body.name.strip(),
        description=body.description,
        source=body.source,
        entries=_resolve(body, gbif, repo),
        created_by=user.name,
    )
    return _out(repo.save_species_list(sl), repo)


@router.get("/{list_id}", response_model=SpeciesListOut, operation_id="getSpeciesList")
def get_list(list_id: uuid.UUID, repo: RepoDep) -> SpeciesListOut:
    return _out(_get(repo, list_id), repo)


@router.put(
    "/{list_id}",
    response_model=SpeciesListOut,
    operation_id="updateSpeciesList",
    dependencies=[Analyst],
)
def update_list(
    list_id: uuid.UUID, body: SpeciesListIn, repo: RepoDep, gbif: GbifDep, settings: SettingsDep
) -> SpeciesListOut:
    """Replace the list's name, description, source and species."""
    sl = _get(repo, list_id)
    _check_size(body, settings)
    sl.name = body.name.strip()
    sl.description = body.description
    sl.source = body.source
    sl.entries = _resolve(body, gbif, repo)
    sl.updated_ts = utcnow()
    return _out(repo.save_species_list(sl), repo)


@router.delete(
    "/{list_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    operation_id="deleteSpeciesList",
    dependencies=[Analyst],
)
def delete_list(list_id: uuid.UUID, repo: RepoDep) -> None:
    try:
        repo.delete_species_list(list_id)
    except NotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Species list not found") from exc


@router.post(
    "/{list_id}/runs",
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="runSpeciesList",
    dependencies=[Analyst],
)
def run_list(
    list_id: uuid.UUID,
    repo: RepoDep,
    queue: QueueDep,
    settings: SettingsDep,
    body: BatchRunRequest | None = None,
) -> JobOut:
    """One batch job running the species pipeline for every resolved species of the list
    (new species stop at the native-range review, as in a single run)."""
    sl = _get(repo, list_id)
    body = body or BatchRunRequest()
    keys = list(dict.fromkeys(sl.taxon_keys))
    if body.only_registered:
        keys = [k for k in keys if repo.get_species(k) is not None]
    if not keys:
        raise HTTPException(UNPROCESSABLE, "The list has no species to run")
    if len(keys) > settings.batch_max_species:
        raise HTTPException(
            UNPROCESSABLE,
            f"At most {settings.batch_max_species} species per batch",
        )
    job = repo.create_job(
        Job(kind=JobKind.BATCH, taxon_key=None, message=f"Queued: {len(keys)} species")
    )
    queue.enqueue_batch(str(job.job_id), keys, body.force_retrain)
    return job_out(job)
