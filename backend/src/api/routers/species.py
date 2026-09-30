"""Species endpoints: search, registry status, pipeline runs, native range, occurrences,
severity configuration."""

from __future__ import annotations

import logging
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.converters import job_out, model_summary, native_range_out, species_summary
from src.api.deps import (
    InatSearch,
    JobQueue,
    TaxonSearch,
    gbif_search_dep,
    inat_search_dep,
    queue_dep,
    repository_dep,
)
from src.api.schemas import (
    JobOut,
    NativeRangeOut,
    NativeRangeUpdate,
    NativeRangeUpdateResult,
    OccurrencePage,
    OccurrenceRow,
    OccurrenceSources,
    RunRequest,
    SeverityConfigModel,
    SeverityResult,
    SpeciesDetail,
    SpeciesSummary,
    TaxonSearchResult,
    Timeline,
    TimelineYear,
)
from src.domain import Job, JobKind
from src.modeling.severity_index import SeverityConfig
from src.orchestration.registry_ops import recompute_severity, update_native_range
from src.persistence.repository import Repository

log = logging.getLogger(__name__)
router = APIRouter(prefix="/species", tags=["species"])

RepoDep = Annotated[Repository, Depends(repository_dep)]
QueueDep = Annotated[JobQueue, Depends(queue_dep)]


def _get_species_or_404(repo: Repository, taxon_key: int):
    sp = repo.get_species(taxon_key)
    if sp is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Species {taxon_key} not in registry")
    return sp


@router.get("/search", response_model=list[TaxonSearchResult], operation_id="searchSpecies")
def search_species(
    repo: RepoDep,
    q: Annotated[str, Query(min_length=2, max_length=100)],
    limit: Annotated[int, Query(ge=1, le=25)] = 10,
    gbif: TaxonSearch = Depends(gbif_search_dep),
    inat: InatSearch | None = Depends(inat_search_dep),
) -> list[TaxonSearchResult]:
    """Resolve free text to canonical GBIF backbone taxon keys, enriched with iNat metadata."""
    suggestions = [
        s
        for s in gbif.suggest_species(q, limit=limit)
        if (s.rank or "").upper() in {"SPECIES", "SUBSPECIES", "VARIETY", ""}
    ]
    inat_by_name: dict[str, object] = {}
    if inat is not None:
        try:
            for t in inat.autocomplete(q, limit=limit):
                inat_by_name[t.name.lower()] = t
        except Exception as exc:  # enrichment only
            log.warning("iNaturalist autocomplete failed: %s", exc)
    out: list[TaxonSearchResult] = []
    for s in suggestions:
        t = inat_by_name.get((s.canonical_name or s.scientific_name).lower())
        reg = repo.get_species(s.taxon_key)
        out.append(
            TaxonSearchResult(
                taxon_key=s.taxon_key,
                scientific_name=s.scientific_name,
                canonical_name=s.canonical_name,
                rank=s.rank,
                breadcrumb=[
                    x for x in (s.kingdom, s.phylum, s.class_, s.order, s.family, s.genus) if x
                ],
                common_name=getattr(t, "common_name", None),
                thumbnail_url=getattr(t, "thumbnail_url", None),
                inat_taxon_id=getattr(t, "inat_taxon_id", None),
                in_registry=reg is not None,
                status=reg.status if reg else None,
            )
        )
    return out


@router.get("", response_model=list[SpeciesSummary], operation_id="listSpecies")
def list_species(repo: RepoDep) -> list[SpeciesSummary]:
    return [species_summary(repo, sp) for sp in repo.list_species()]


@router.get("/{taxon_key}", response_model=SpeciesDetail, operation_id="getSpecies")
def get_species(taxon_key: int, repo: RepoDep) -> SpeciesDetail:
    sp = _get_species_or_404(repo, taxon_key)
    nr = repo.get_native_range(taxon_key)
    summary = species_summary(repo, sp)
    return SpeciesDetail(
        **summary.model_dump(),
        canonical_name=sp.canonical_name,
        inat_taxon_id=sp.inat_taxon_id,
        last_gbif_download_key=sp.last_gbif_download_key,
        last_gbif_download_doi=sp.last_gbif_download_doi,
        bioclim_version_used=sp.bioclim_version_used,
        last_error=sp.last_error,
        native_range_status=nr.status if nr else "missing",
        sources=OccurrenceSources(
            **repo.source_counts(taxon_key), last_inat_fetch_ts=sp.last_inat_fetch_ts
        ),
        current_model=model_summary(repo, sp),
        severity_config=SeverityConfigModel.model_validate(
            SeverityConfig.from_dict(sp.severity_config).to_dict()
        ),
    )


@router.post(
    "/{taxon_key}/runs",
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="runSpeciesPipeline",
)
def run_pipeline(
    taxon_key: int, repo: RepoDep, queue: QueueDep, body: RunRequest | None = None
) -> JobOut:
    """Enqueue the incremental-fetch / conditional-retrain pipeline (full run if new)."""
    job = repo.create_job(Job(kind=JobKind.PIPELINE, taxon_key=taxon_key, message="Queued"))
    body = body or RunRequest()
    queue.enqueue_pipeline(str(job.job_id), taxon_key, body.force_retrain, body.full_resync)
    return job_out(job)


@router.post(
    "/{taxon_key}/scenarios",
    response_model=JobOut,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="projectScenarios",
)
def project_scenarios(taxon_key: int, repo: RepoDep, queue: QueueDep) -> JobOut:
    """Project the current model onto registered climate scenarios (no retraining)."""
    sp = _get_species_or_404(repo, taxon_key)
    if not sp.model_version:
        raise HTTPException(status.HTTP_409_CONFLICT, "No trained model yet")
    job = repo.create_job(Job(kind=JobKind.SCENARIOS, taxon_key=taxon_key, message="Queued"))
    queue.enqueue_scenarios(str(job.job_id), taxon_key)
    return job_out(job)


# ----------------------------------------------------------- native range
@router.get(
    "/{taxon_key}/native-range", response_model=NativeRangeOut, operation_id="getNativeRange"
)
def get_native_range(taxon_key: int, repo: RepoDep) -> NativeRangeOut:
    _get_species_or_404(repo, taxon_key)
    nr = repo.get_native_range(taxon_key)
    if nr is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No native-range polygon yet")
    return native_range_out(nr)


@router.put(
    "/{taxon_key}/native-range",
    response_model=NativeRangeUpdateResult,
    operation_id="updateNativeRange",
)
def put_native_range(
    taxon_key: int, body: NativeRangeUpdate, repo: RepoDep, queue: QueueDep
) -> NativeRangeUpdateResult:
    """Save a user-reviewed polygon. Confirming it unlocks (re)training."""
    _get_species_or_404(repo, taxon_key)
    # Reject before saving: a rejected request must not downgrade a confirmed range to draft.
    if body.train and not body.confirm:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Training requires confirming the native range"
        )
    try:
        nr = update_native_range(
            repo, taxon_key, body.geometry.model_dump(), body.confirm, body.note
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    job = None
    if body.train:
        job = repo.create_job(Job(kind=JobKind.PIPELINE, taxon_key=taxon_key, message="Queued"))
        queue.enqueue_pipeline(str(job.job_id), taxon_key, False)
    return NativeRangeUpdateResult(
        native_range=native_range_out(nr), job=job_out(job) if job else None
    )


# ------------------------------------------------------------ occurrences
@router.get(
    "/{taxon_key}/occurrences", response_model=OccurrencePage, operation_id="listOccurrences"
)
def list_occurrences(
    taxon_key: int,
    repo: RepoDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    source: Literal["GBIF", "iNaturalist"] | None = None,
    range_label: Literal["native", "introduced", "unknown"] | None = None,
) -> OccurrencePage:
    """Paginated occurrence list for table views (map layers use pg_tileserv vector tiles)."""
    _get_species_or_404(repo, taxon_key)
    df, total = repo.page_occurrences(taxon_key, offset, limit, source, range_label)
    df = df.astype(object).where(pd.notna(df), None)
    items = [OccurrenceRow.model_validate(r) for r in df.to_dict(orient="records")]
    return OccurrencePage(items=items, total=total, offset=offset, limit=limit)


@router.get(
    "/{taxon_key}/occurrences/timeline",
    response_model=Timeline,
    operation_id="getOccurrenceTimeline",
)
def occurrence_timeline(taxon_key: int, repo: RepoDep) -> Timeline:
    _get_species_or_404(repo, taxon_key)
    df = repo.occurrence_timeline(taxon_key)
    if df.empty:
        return Timeline(years=[], min_year=None, max_year=None)
    piv = df.pivot_table(
        index="year", columns="range_label", values="n", aggfunc="sum", fill_value=0
    )
    piv = piv.reindex(range(int(piv.index.min()), int(piv.index.max()) + 1), fill_value=0)
    for c in ("native", "introduced", "unknown"):
        if c not in piv:
            piv[c] = 0
    cum = piv[["native", "introduced", "unknown"]].sum(axis=1).cumsum()
    years = [
        TimelineYear(
            year=int(y),
            native=int(r["native"]),
            introduced=int(r["introduced"]),
            unknown=int(r["unknown"]),
            cumulative=int(cum.loc[y]),
        )
        for y, r in piv.iterrows()
    ]
    return Timeline(years=years, min_year=years[0].year, max_year=years[-1].year)


# --------------------------------------------------------------- severity
@router.put(
    "/{taxon_key}/severity-config",
    response_model=SeverityResult | None,
    operation_id="updateSeverityConfig",
)
def put_severity_config(
    taxon_key: int, body: SeverityConfigModel, repo: RepoDep
) -> SeverityResult | None:
    """Update the severity weights/priors and re-weight the current model's components
    (no retraining; returns null if no model exists yet)."""
    _get_species_or_404(repo, taxon_key)
    try:
        cfg = SeverityConfig.from_dict(body.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    result = recompute_severity(repo, taxon_key, cfg)
    return SeverityResult.model_validate(result) if result else None
