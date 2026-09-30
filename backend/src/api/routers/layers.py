"""Map-layer descriptors: tile URL templates for titiler (rasters) and pg_tileserv (vectors).

The browser loads tiles straight from the tile servers; this endpoint only tells it where
they are, how to style them and which caveats must accompany them. A MESS layer is always
returned together with any suitability projection.
"""

from __future__ import annotations

import json
from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.deps import artifacts_dep, repository_dep, settings_dep
from src.api.schemas import (
    LayerCaveats,
    LayerSet,
    LegendEntry,
    RasterLayer,
    ScenarioLayer,
    VectorLayer,
)
from src.config import Settings
from src.domain import ZONE_LABELS
from src.features.raster_sampler import METADATA_FILENAME, BioclimMetadata
from src.persistence.artifact_store import ArtifactStore
from src.persistence.repository import Repository

router = APIRouter(prefix="/species/{taxon_key}/layers", tags=["layers"])
RepoDep = Annotated[Repository, Depends(repository_dep)]

ZONE_RGBA = {0: (0, 0, 0, 0), 1: (42, 157, 143, 200), 2: (244, 162, 97, 220), 3: (214, 40, 40, 230)}
SUITABILITY_RAMP = [
    (0.0, "#440154"),
    (0.25, "#3b528b"),
    (0.5, "#21918c"),
    (0.75, "#5ec962"),
    (1.0, "#fde725"),
]
MESS_RAMP = [
    (-50.0, "#b2182b"),
    (-10.0, "#ef8a62"),
    (0.0, "#f7f7f7"),
    (25.0, "#67a9cf"),
    (50.0, "#2166ac"),
]


def _cog_tiles(settings: Settings, artifacts: ArtifactStore, uri: str, params: str) -> str:
    return (
        f"{settings.titiler_public_url}/cog/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png"
        f"?url={artifacts.titiler_url(uri)}&{params}"
    )


def _suitability_layer(
    settings: Settings, artifacts: ArtifactStore, uri: str, lid: str, label: str, description: str
) -> RasterLayer:
    return RasterLayer(
        id=lid,
        label=label,
        kind="continuous",
        tile_url=_cog_tiles(
            settings,
            artifacts,
            uri,
            "rescale=0,1&colormap_name=viridis&nodata=-9999&resampling=nearest",
        ),
        legend=[LegendEntry(value=v, color=c, label=f"{v:.2f}") for v, c in SUITABILITY_RAMP],
        description=description,
    )


@router.get("", response_model=LayerSet, operation_id="getLayers")
def get_layers(
    taxon_key: int,
    repo: RepoDep,
    model_version: int | None = None,
    settings: Settings = Depends(settings_dep),
    artifacts: ArtifactStore = Depends(artifacts_dep),
) -> LayerSet:
    sp = repo.get_species(taxon_key)
    if sp is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Species {taxon_key} not in registry")
    tileserv = settings.tileserv_public_url
    layers = LayerSet(
        taxon_key=taxon_key,
        model_version=None,
        occurrences=VectorLayer(
            id="occurrences",
            label="Occurrences",
            source_layer="occurrences",
            tile_url=f"{tileserv}/public.occurrence_tiles/{{z}}/{{x}}/{{y}}.pbf"
            f"?taxon_key={taxon_key}",
        ),
        native_range=VectorLayer(
            id="native_range",
            label="Native range",
            source_layer="native_range",
            tile_url=f"{tileserv}/public.native_range_tiles/{{z}}/{{x}}/{{y}}.pbf"
            f"?taxon_key={taxon_key}",
        ),
    )
    version = model_version or sp.model_version
    if not version:
        return layers
    mv = repo.get_model_version(taxon_key, version)
    if mv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model version not found")

    a = mv.artifacts
    layers.model_version = version
    layers.suitability = _suitability_layer(
        settings,
        artifacts,
        a["suitability"],
        "suitability",
        "Suitability (current climate)",
        "MaxEnt cloglog suitability, 0–1. Interpret together with the MESS layer.",
    )
    layers.mess = RasterLayer(
        id="mess",
        label="MESS extrapolation",
        kind="continuous",
        tile_url=_cog_tiles(
            settings, artifacts, a["mess"], "rescale=-50,50&colormap_name=rdbu&nodata=-9999"
        ),
        legend=[LegendEntry(value=v, color=c, label=f"{v:+.0f}") for v, c in MESS_RAMP],
        description="Multivariate Environmental Similarity Surface. Negative values = climate "
        "outside the training range; suitability there is an extrapolation.",
    )
    cmap = quote(json.dumps({str(k): list(v) for k, v in ZONE_RGBA.items()}), safe="")
    layers.zones = RasterLayer(
        id="zones",
        label="Range zones",
        kind="categorical",
        tile_url=_cog_tiles(
            settings, artifacts, a["zones"], f"colormap={cmap}&nodata=255&resampling=nearest"
        ),
        legend=[
            LegendEntry(
                value=k, color="#{:02x}{:02x}{:02x}".format(*ZONE_RGBA[k][:3]), label=ZONE_LABELS[k]
            )
            for k in (1, 2, 3)
        ],
        description="Suitable cells classified by position relative to the confirmed native "
        "range and known occurrences (threshold: 10th-percentile training presence).",
    )
    # Light "uncertainty haze" that stays visible over the suitability ramp on either basemap.
    extra_cmap = quote(json.dumps({"0": [0, 0, 0, 0], "1": [226, 232, 240, 165]}), safe="")
    layers.extrapolation = RasterLayer(
        id="extrapolation",
        label="Extrapolation (MESS < 0)",
        kind="categorical",
        tile_url=_cog_tiles(
            settings,
            artifacts,
            a["extrapolation"],
            f"colormap={extra_cmap}&nodata=255&resampling=nearest",
        ),
        legend=[LegendEntry(value=1, color="#e2e8f0", label="MESS < 0 — extrapolation")],
        description="Cells where at least one predictor lies outside the training range. "
        "Suitability shown underneath is an extrapolation: treat as low confidence.",
    )

    def extrapolation_layer(uri: str, lid: str, label: str) -> RasterLayer:
        return RasterLayer(
            id=lid,
            label=label,
            kind="categorical",
            tile_url=_cog_tiles(
                settings, artifacts, uri, f"colormap={extra_cmap}&nodata=255&resampling=nearest"
            ),
            legend=[LegendEntry(value=1, color="#e2e8f0", label="MESS < 0 — extrapolation")],
            description="Cells where at least one predictor lies outside the training range "
            "under this climate. Suitability underneath is an extrapolation.",
        )

    scenario_layers: list[ScenarioLayer] = []
    for name, entry in a.get("scenarios", {}).items():
        # Legacy entries were a bare suitability path without a paired extrapolation mask.
        suit_uri = entry["suitability"] if isinstance(entry, dict) else entry
        extra_uri = entry.get("extrapolation") if isinstance(entry, dict) else None
        label = name.replace("_", " ")
        scenario_layers.append(
            ScenarioLayer(
                id=f"scenario:{name}",
                name=name,
                label=label,
                suitability=_suitability_layer(
                    settings,
                    artifacts,
                    suit_uri,
                    f"scenario:{name}",
                    f"Suitability — {label}",
                    "Same model projected onto a CMIP6 future climate (no retraining). "
                    "Extrapolation risk is typically higher than for current climate.",
                ),
                extrapolation=(
                    extrapolation_layer(
                        extra_uri, f"extrapolation:{name}", f"Extrapolation (MESS < 0) — {label}"
                    )
                    if extra_uri
                    else None
                ),
                extrapolated_land_fraction=(
                    entry.get("extrapolated_land_fraction") if isinstance(entry, dict) else None
                ),
            )
        )
    layers.scenarios = scenario_layers
    bioclim_version = mv.metrics.get("reproducibility", {}).get("bioclim_version_used")
    meta_path = settings.bioclim_root / str(bioclim_version) / METADATA_FILENAME
    if bioclim_version and meta_path.exists():
        layers.scenarios_available = list(BioclimMetadata.load(meta_path).scenarios)
    layers.caveats = LayerCaveats(
        model_type=mv.model_type,
        confidence_label=mv.metrics["confidence_label"],
        transferability_caveat=mv.metrics["transferability_caveat"],
    )
    return layers
