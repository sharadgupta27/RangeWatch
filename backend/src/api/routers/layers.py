"""Map-layer descriptors: tile URL templates for titiler (rasters) and pg_tileserv (vectors).

The browser loads tiles straight from the tile servers; this endpoint only tells it where
they are, how to style them and which caveats must accompany them. A MESS layer is always
returned together with any suitability projection; models trained with the advanced
diagnostics also get exDet, MOP, Shape, AOA and consensus views plus a consensus overlay.
"""

from __future__ import annotations

import json
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.deps import artifacts_dep, repository_dep, settings_dep
from src.api.schemas import (
    HiresLayer,
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
_BIG = 1e30  # open interval bound for titiler interval colormaps (JSON has no infinity)
EXDET_BINS = [
    (-_BIG, -1.0, "#67001f", "NT1 < −1 · strong univariate novelty"),
    (-1.0, 0.0, "#d6604d", "NT1 < 0 · predictor outside training range"),
    (0.0, 0.5, "#2166ac", "NT2 < 0.5 · analogous"),
    (0.5, 1.0, "#92c5de", "NT2 0.5–1 · analogous"),
    (1.0, 2.0, "#fdb863", "NT2 1–2 · novel combination"),
    (2.0, _BIG, "#e66101", "NT2 > 2 · strongly novel combination"),
]
CONSENSUS_COLORS = ["#4393c3", "#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]
# Haze in the colour of the MESS overlay, denser the more diagnostics agree on extrapolation.
CONSENSUS_HAZE_ALPHA = [0, 50, 85, 120, 155, 190]
HAZE_RGB = (226, 232, 240)


def _rgba(hex_color: str, alpha: int = 255) -> list[int]:
    return [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)] + [alpha]


def _threshold_bins(t: float) -> list[tuple[float, float, str, str]]:
    """Classes relative to a diagnostic's threshold: blue within it, red beyond it."""
    return [
        (-_BIG, 0.5 * t, "#2166ac", f"< {0.5 * t:.3g}"),
        (0.5 * t, t, "#92c5de", f"{0.5 * t:.3g}–{t:.3g} · within threshold"),
        (t, 1.5 * t, "#f4a582", f"{t:.3g}–{1.5 * t:.3g} · beyond threshold"),
        (1.5 * t, 2 * t, "#d6604d", f"{1.5 * t:.3g}–{2 * t:.3g}"),
        (2 * t, _BIG, "#67001f", f"> {2 * t:.3g} · over 2× threshold"),
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


def _interval_layer(
    settings: Settings,
    artifacts: ArtifactStore,
    uri: str,
    lid: str,
    label: str,
    bins: list[tuple[float, float, str, str]],
    description: str,
) -> RasterLayer:
    """A continuous diagnostic rendered in classes through a titiler interval colormap."""
    cmap = quote(json.dumps([[[lo, hi], _rgba(c)] for lo, hi, c, _ in bins]), safe="")
    return RasterLayer(
        id=lid,
        label=label,
        kind="categorical",
        tile_url=_cog_tiles(
            settings, artifacts, uri, f"colormap={cmap}&nodata=-9999&resampling=nearest"
        ),
        legend=[LegendEntry(value=i, color=c, label=t) for i, (_, _, c, t) in enumerate(bins)],
        description=description,
    )


def _consensus_overlay(
    settings: Settings, artifacts: ArtifactStore, uri: str, lid: str, label: str
) -> RasterLayer:
    cmap = {str(n): [*HAZE_RGB, a] for n, a in enumerate(CONSENSUS_HAZE_ALPHA)}
    return RasterLayer(
        id=lid,
        label=label,
        kind="categorical",
        tile_url=_cog_tiles(
            settings,
            artifacts,
            uri,
            f"colormap={quote(json.dumps(cmap), safe='')}&nodata=255&resampling=nearest",
        ),
        legend=[
            LegendEntry(
                value=n,
                color="#{:02x}{:02x}{:02x}{:02x}".format(*HAZE_RGB, CONSENSUS_HAZE_ALPHA[n]),
                label=f"{n} of 5 diagnostics flag extrapolation",
            )
            for n in range(1, 6)
        ],
        description="Haze grows with the number of extrapolation diagnostics (MESS, exDet, "
        "MOP, Shape, AOA) that flag the cell: the denser it is, the less the suitability "
        "underneath can be trusted.",
    )


def _diagnostic_layers(
    settings: Settings, artifacts: ArtifactStore, a: dict[str, Any], ext: dict[str, Any]
) -> list[RasterLayer]:
    """exDet / MOP / Shape / AOA / consensus map views of one model version."""
    thr = {d["id"]: d["threshold"] for d in ext.get("diagnostics", [])}
    out: list[RasterLayer] = []
    if "exdet" in a:
        out.append(
            _interval_layer(
                settings,
                artifacts,
                a["exdet"],
                "exdet",
                "exDet novelty",
                EXDET_BINS,
                "Extrapolation Detection (Mesgaran et al. 2014). NT1 < 0: a predictor lies "
                "outside its training range (univariate novelty — the cells MESS flags). "
                "NT2 > 1: every predictor is in range but their combination was never seen "
                "in training (novel correlation structure), which MESS cannot detect.",
            )
        )
    specs = {
        "mop": (
            "MOP distance",
            "Mobility-Oriented Parity (Owens et al. 2013; Cobos et al. 2024): mean "
            f"standardised distance to the closest {ext.get('mop_percentage', 1):g}% of "
            "training conditions.",
        ),
        "shape": (
            "Shape extrapolation",
            "Shape (Velazco et al. 2024): Mahalanobis distance to the nearest training point "
            "relative to the training data's mean distance to its centroid (× 100).",
        ),
        "aoa": (
            "AOA dissimilarity index",
            "Area of Applicability (Meyer & Pebesma 2021): distance to the nearest training "
            "point in the importance-weighted predictor space, relative to the mean distance "
            "between training points. Beyond the threshold the cell is outside the AOA, where "
            "the model's cross-validated performance does not carry over.",
        ),
    }
    for key, (label, text) in specs.items():
        t = thr.get(key)
        if key not in a or not t or t <= 0:
            continue
        out.append(
            _interval_layer(
                settings,
                artifacts,
                a[key],
                key,
                label,
                _threshold_bins(t),
                f"{text} Threshold {t:.3g}: outlier-trimmed maximum of the training points' "
                "values against the other spatial CV folds.",
            )
        )
    if "consensus" in a:
        cmap = quote(
            json.dumps({str(n): _rgba(c, 220) for n, c in enumerate(CONSENSUS_COLORS)}), safe=""
        )
        out.append(
            RasterLayer(
                id="consensus",
                label="Extrapolation consensus",
                kind="categorical",
                tile_url=_cog_tiles(
                    settings,
                    artifacts,
                    a["consensus"],
                    f"colormap={cmap}&nodata=255" "&resampling=nearest",
                ),
                legend=[
                    LegendEntry(
                        value=n,
                        color=c,
                        label="no diagnostic flags extrapolation" if n == 0 else f"{n} of 5",
                    )
                    for n, c in enumerate(CONSENSUS_COLORS)
                ],
                description="How many of the five diagnostics (MESS, exDet, MOP, Shape, AOA) "
                "flag the cell as extrapolation. They test different things — range, "
                "combinations, distance to analogous conditions, distance in the model's "
                "weighted space — so agreement is stronger evidence than any one of them.",
            )
        )
    return out


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

    ext = mv.metrics.get("projection", {}).get("extrapolation") or {}
    layers.diagnostics = _diagnostic_layers(settings, artifacts, a, ext)
    if "consensus" in a:
        layers.consensus_overlay = _consensus_overlay(
            settings, artifacts, a["consensus"], "consensus-overlay", "Extrapolation consensus"
        )

    def consensus_overlay(entry: Any, lid: str, label: str) -> RasterLayer | None:
        uri = entry.get("consensus") if isinstance(entry, dict) else None
        return _consensus_overlay(settings, artifacts, uri, lid, label) if uri else None

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
                consensus=consensus_overlay(
                    entry, f"consensus:{name}", f"Extrapolation consensus — {label}"
                ),
                diagnostics_land_fraction=(
                    entry.get("diagnostics_land_fraction") if isinstance(entry, dict) else None
                ),
            )
        )
    layers.scenarios = scenario_layers
    hires = a.get("hires")
    if hires:
        res_label = hires["resolution"].replace("s", "″").replace("m", "′")
        layers.hires = HiresLayer(
            resolution=hires["resolution"],
            bbox=hires["bbox"],
            suitability=_suitability_layer(
                settings,
                artifacts,
                hires["suitability"],
                "hires",
                f"Suitability — {res_label} detail",
                "Same model projected at high resolution inside the selected region (no "
                "retraining). Interpret together with this region's extrapolation mask.",
            ),
            extrapolation=extrapolation_layer(
                hires["extrapolation"], "extrapolation:hires", f"Extrapolation — {res_label}"
            ),
            extrapolated_land_fraction=hires.get("extrapolated_land_fraction"),
            consensus=consensus_overlay(
                hires, "consensus:hires", f"Extrapolation consensus — {res_label}"
            ),
            diagnostics_land_fraction=hires.get("diagnostics_land_fraction"),
            created_ts=hires["created_ts"],
        )
    bioclim_version = mv.metrics.get("reproducibility", {}).get("bioclim_version_used")
    meta_path = settings.bioclim_root / str(bioclim_version) / METADATA_FILENAME
    if bioclim_version and meta_path.exists():
        meta = BioclimMetadata.load(meta_path)
        layers.scenarios_available = list(meta.scenarios)
        layers.hires_available = meta.finest_hires()
        if layers.hires_available:
            layers.hires_max_cells = settings.hires_max_cells
    layers.caveats = LayerCaveats(
        model_type=mv.model_type,
        confidence_label=mv.metrics["confidence_label"],
        transferability_caveat=mv.metrics["transferability_caveat"],
    )
    return layers
