"""Service metadata: health and the active global bioclim layer."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import CurrentUser
from src.api.converters import climate_version_out
from src.api.deps import settings_dep
from src.api.schemas import BioclimInfo, ClimateVersionOut, Health, MeOut
from src.config import Settings
from src.domain import PREDICTOR_DESCRIPTIONS
from src.features.raster_sampler import METADATA_FILENAME, BioclimMetadata, list_versions

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=Health, operation_id="health")
def health(settings: Settings = Depends(settings_dep)) -> Health:
    meta = settings.bioclim_root / settings.bioclim_version / METADATA_FILENAME
    return Health(
        status="ok", bioclim_version=settings.bioclim_version, bioclim_available=meta.exists()
    )


@router.get("/bioclim", response_model=BioclimInfo, operation_id="getBioclimInfo")
def bioclim(settings: Settings = Depends(settings_dep)) -> BioclimInfo:
    path = settings.bioclim_root / settings.bioclim_version / METADATA_FILENAME
    if not path.exists():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Bioclim layer {settings.bioclim_version} not built yet",
        )
    meta = BioclimMetadata.load(path)
    alt = settings.crosscheck_bioclim_version
    alt_path = settings.bioclim_root / alt / METADATA_FILENAME if alt else None
    alt_meta = BioclimMetadata.load(alt_path) if alt_path and alt_path.exists() else None
    return BioclimInfo(
        hires_resolutions=list(meta.hires),
        crosscheck_version=alt_meta.version if alt_meta else None,
        crosscheck_source=alt_meta.source if alt_meta else None,
        version=meta.version,
        source=meta.source,
        resolution=meta.resolution,
        selected_predictors=list(meta.selected_predictors),
        vif=meta.vif,
        descriptions=PREDICTOR_DESCRIPTIONS,
        scenarios=list(meta.scenarios),
        citation=meta.citation,
        static_layers=list(meta.static_layers),
        static_layer_citations=sorted(
            {c for e in meta.static_layers.values() if (c := e.get("citation"))}
        ),
    )


@router.get(
    "/bioclim/versions", response_model=list[ClimateVersionOut], operation_id="listClimateVersions"
)
def climate_versions(settings: Settings = Depends(settings_dep)) -> list[ClimateVersionOut]:
    """Installed climate data versions a species can be trained on (WorldClim, CHELSA, …),
    with their predictors grouped by process and whether future data exists for each."""
    return [climate_version_out(m, settings) for m in list_versions(settings.bioclim_root)]


@router.get("/me", response_model=MeOut, operation_id="getMe")
def me(user: CurrentUser, settings: Settings = Depends(settings_dep)) -> MeOut:
    """The calling user and role, so the UI can offer only permitted actions (the API
    enforces them regardless)."""
    return MeOut(
        user=user.name,
        role=user.role,
        auth_mode=settings.auth_mode,
        permissions={"view": True, "run": user.can("analyst"), "curate": user.can("curator")},
    )
