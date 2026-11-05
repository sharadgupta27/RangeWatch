"""Data downloads and standards (U8): COG rasters, GeoPackage, ODMAP, STAC, citation bundle.

The index endpoint lists what a version offers (URLs relative to the API root, like the
bulletin's); the browser downloads files directly from these endpoints.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import FileResponse, PlainTextResponse

from src.api.deps import artifacts_dep, repository_dep, settings_dep
from src.api.schemas import ExportIndex, ExportItem, ModelCardOut, OdmapReport, StacItem
from src.bulletin import exports as ex
from src.bulletin.model_card import model_card, model_card_markdown
from src.config import Settings
from src.domain import ModelVersionRecord, SpeciesRecord
from src.persistence.artifact_store import ArtifactStore
from src.persistence.repository import Repository

router = APIRouter(prefix="/species/{taxon_key}/models/{model_version}/exports", tags=["exports"])
RepoDep = Annotated[Repository, Depends(repository_dep)]
ArtifactsDep = Annotated[ArtifactStore, Depends(artifacts_dep)]
GPKG = "application/geopackage+sqlite3"


def _load(
    repo: Repository, taxon_key: int, model_version: int
) -> tuple[SpeciesRecord, ModelVersionRecord]:
    sp = repo.get_species(taxon_key)
    mv = repo.get_model_version(taxon_key, model_version) if sp else None
    if sp is None or mv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model version not found")
    return sp, mv


def _name(sp: SpeciesRecord, mv: ModelVersionRecord, suffix: str) -> str:
    base = (sp.canonical_name or sp.scientific_name).replace(" ", "_")
    return f"{base}_v{mv.model_version}_{suffix}"


@router.get("", response_model=ExportIndex, operation_id="listExports")
def index(
    taxon_key: int, model_version: int, repo: RepoDep, artifacts: ArtifactsDep
) -> ExportIndex:
    sp, mv = _load(repo, taxon_key, model_version)
    base = f"/species/{taxon_key}/models/{model_version}/exports"
    items: list[ExportItem] = []
    for key in ex.raster_paths(mv, artifacts):
        root = key.split(":")[0]
        label, desc, _ = ex.RASTERS.get(root, (key, key, []))
        if key.startswith("scenario:"):
            _, name, part = key.split(":")
            label, desc = f"Scenario {name.replace('_', ' ')} — {part}", f"{part} under {name}"
        items.append(
            ExportItem(
                id=key,
                label=label,
                url=f"{base}/raster/{key}",
                media_type=ex.COG_TYPE,
                description=f"Cloud-optimised GeoTIFF. {desc}",
            )
        )
    for rule, t in (mv.metrics.get("thresholds") or {}).items():
        items.append(
            ExportItem(
                id=f"binary:{rule}",
                label=f"Binary map — {t['label']}",
                url=f"{base}/raster/binary:{rule}",
                media_type=ex.COG_TYPE,
                description=f"Suitable (1) / unsuitable (0) at {t['value']:.4g}; nodata 255.",
            )
        )
    items += [
        ExportItem(
            id="model_card",
            label="Model card (plain language)",
            url=f"{base}/model-card?format=md",
            media_type="text/markdown",
            description="Confidence, caveats, quality, extrapolation and data in plain "
            "language, from the stored metrics; JSON with ?format=json.",
        ),
        ExportItem(
            id="occurrences",
            label="Records (GeoPackage)",
            url=f"{base}/occurrences.gpkg",
            media_type=GPKG,
            description="Layers: occurrences (effective labels + quality flags) and "
            "training_data (presences, background, pseudo-absences, survey records).",
        ),
        ExportItem(
            id="odmap",
            label="ODMAP protocol",
            url=f"{base}/odmap?format=md",
            media_type="text/markdown",
            description="Overview–Data–Model–Assessment–Prediction (Zurell et al. 2020), "
            "from the stored metadata; JSON with ?format=json.",
        ),
        ExportItem(
            id="stac",
            label="STAC item",
            url=f"{base}/stac",
            media_type="application/json",
            description="STAC 1.0 item listing every output with its download link.",
        ),
        ExportItem(
            id="citation",
            label="Citation bundle",
            url=f"{base}/citation",
            media_type="text/plain",
            description="GBIF download DOI, iNaturalist and upload citations, climate data, "
            "software versions and snapshot hashes.",
        ),
        ExportItem(
            id="bundle",
            label="Everything (zip)",
            url=f"{base}/bundle.zip",
            media_type="application/zip",
            description="All rasters, the GeoPackage, ODMAP, STAC and citation in one file.",
        ),
    ]
    return ExportIndex(taxon_key=taxon_key, model_version=model_version, items=items)


@router.get(
    "/raster/{layer}",
    response_class=FileResponse,
    operation_id="downloadExportRaster",
    responses={200: {"content": {"image/tiff": {}}}},
)
def raster(
    taxon_key: int, model_version: int, layer: str, repo: RepoDep, artifacts: ArtifactsDep
) -> FileResponse:
    sp, mv = _load(repo, taxon_key, model_version)
    if layer.startswith("binary:"):
        try:
            path = ex.binary_path(mv, artifacts, layer.split(":", 1)[1])
        except KeyError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown threshold rule") from exc
    else:
        path = ex.raster_paths(mv, artifacts).get(layer)
        if path is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No raster {layer!r} in this version")
    return FileResponse(
        path, media_type="image/tiff", filename=_name(sp, mv, layer.replace(":", "_") + ".tif")
    )


@router.get(
    "/occurrences.gpkg",
    response_class=FileResponse,
    operation_id="downloadExportGeoPackage",
    responses={200: {"content": {GPKG: {}}}},
)
def geopackage(
    taxon_key: int, model_version: int, repo: RepoDep, artifacts: ArtifactsDep
) -> FileResponse:
    sp, mv = _load(repo, taxon_key, model_version)
    vdir = artifacts.local_path(str(artifacts.version_dir(taxon_key, model_version)))
    path = vdir / "exports" / "occurrences.gpkg"
    if not path.exists():
        ex.occurrences_gpkg(mv, artifacts, path)
    return FileResponse(path, media_type=GPKG, filename=_name(sp, mv, "records.gpkg"))


@router.get(
    "/odmap",
    response_model=OdmapReport,
    operation_id="getOdmap",
    responses={200: {"content": {"text/markdown": {}}}},
)
def odmap(
    taxon_key: int,
    model_version: int,
    repo: RepoDep,
    format: Literal["json", "md"] = "json",
) -> OdmapReport | PlainTextResponse:
    sp, mv = _load(repo, taxon_key, model_version)
    report = ex.odmap(sp, mv)
    if format == "md":
        return PlainTextResponse(ex.odmap_markdown(report), media_type="text/markdown")
    return OdmapReport.model_validate(report)


@router.get(
    "/model-card",
    response_model=ModelCardOut,
    operation_id="getModelCard",
    responses={200: {"content": {"text/markdown": {}}}},
)
def get_model_card(
    taxon_key: int,
    model_version: int,
    repo: RepoDep,
    format: Literal["json", "md"] = "json",
) -> ModelCardOut | PlainTextResponse:
    """U13: plain-language summary, the confidence label and caveats first."""
    sp, mv = _load(repo, taxon_key, model_version)
    card = model_card(sp, mv)
    if format == "md":
        return PlainTextResponse(model_card_markdown(card), media_type="text/markdown")
    return ModelCardOut.model_validate(card)


@router.get("/stac", response_model=StacItem, operation_id="getStacItem")
def stac(
    taxon_key: int,
    model_version: int,
    repo: RepoDep,
    artifacts: ArtifactsDep,
    settings: Settings = Depends(settings_dep),
) -> StacItem:
    sp, mv = _load(repo, taxon_key, model_version)
    item = ex.stac_item(sp, mv, ex.raster_paths(mv, artifacts), settings.api_public_url)
    return StacItem.model_validate(item)


@router.get(
    "/citation",
    response_class=PlainTextResponse,
    operation_id="getCitationBundle",
    responses={200: {"content": {"text/plain": {}}}},
)
def citation(taxon_key: int, model_version: int, repo: RepoDep) -> PlainTextResponse:
    sp, mv = _load(repo, taxon_key, model_version)
    return PlainTextResponse(ex.citation_bundle(sp, mv))


@router.get(
    "/bundle.zip",
    response_class=Response,
    operation_id="downloadExportBundle",
    responses={200: {"content": {"application/zip": {}}}},
)
def bundle(
    taxon_key: int,
    model_version: int,
    repo: RepoDep,
    artifacts: ArtifactsDep,
    settings: Settings = Depends(settings_dep),
) -> Response:
    sp, mv = _load(repo, taxon_key, model_version)
    with tempfile.TemporaryDirectory() as tmp:
        data = ex.bundle_zip(
            sp, mv, artifacts, ex.raster_paths(mv, artifacts), settings.api_public_url, Path(tmp)
        )
    return Response(
        data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{_name(sp, mv, "bundle.zip")}"'},
    )
