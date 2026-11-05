"""Reusable, citable outputs of a model version (U8): rasters as COG, the training records as
GeoPackage, an ODMAP protocol (Zurell et al. 2020), a STAC 1.0 item and a citation bundle.

Everything is generated from what the version already stores (artifacts + metrics +
reproducibility block): nothing is re-derived or summarised away.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from src.bulletin.model_card import model_card, model_card_markdown
from src.domain import CHANGE_LABELS, ZONE_LABELS, ModelVersionRecord, SpeciesRecord
from src.features.raster_sampler import write_binary
from src.persistence.artifact_store import ArtifactStore

# Raster artifact key → (label, description, STAC roles)
RASTERS: dict[str, tuple[str, str, list[str]]] = {
    "suitability": ("Suitability", "Relative suitability 0–1 (float32, nodata −9999)", ["data"]),
    "mess": ("MESS", "Multivariate environmental similarity (negative = extrapolation)", ["data"]),
    "extrapolation": ("Extrapolation mask", "1 where MESS < 0 (uint8, nodata 255)", ["data"]),
    "zones": (
        "Range zones",
        "; ".join(f"{k} {v}" for k, v in ZONE_LABELS.items()) + " (uint8, nodata 255)",
        ["data"],
    ),
    "consensus": ("Extrapolation consensus", "Diagnostics flagging extrapolation, 0–5", ["data"]),
    "exdet": ("exDet", "NT1 where < 0, else NT2 (Mesgaran et al. 2014)", ["data"]),
    "mop": ("MOP", "Mobility-oriented parity distance", ["data"]),
    "shape": ("Shape", "Shape extrapolation metric (Velazco et al. 2024)", ["data"]),
    "aoa": ("AOA DI", "Area-of-applicability dissimilarity index", ["data"]),
    "limiting": (
        "Limiting factor",
        "Index into the version's predictor list (254 = none), Elith et al. 2010",
        ["data"],
    ),
    "agreement": ("Ensemble agreement", "Weighted share of members agreeing, 0–1", ["data"]),
    "change_previous": (
        "Change since previous version",
        "; ".join(f"{k} {v}" for k, v in CHANGE_LABELS.items()),
        ["data"],
    ),
    "priority": ("Survey priority", "Where surveys add most, 0–1", ["data"]),
    "effort": ("Sampling effort", "log10(target-group GBIF records + 1)", ["data"]),
}
COG_TYPE = "image/tiff; application=geotiff; profile=cloud-optimized"


def raster_paths(mv: ModelVersionRecord, artifacts: ArtifactStore) -> dict[str, Path]:
    """Every raster of a version, keyed by export id (scenario rasters as scenario:<name>…)."""
    out: dict[str, Path] = {}
    a = mv.artifacts
    for key in RASTERS:
        if isinstance(a.get(key), str):
            out[key] = artifacts.local_path(a[key])
    for name, entry in (a.get("scenarios") or {}).items():
        if isinstance(entry, dict):
            for part in ("suitability", "extrapolation", "consensus", "change"):
                if entry.get(part):
                    out[f"scenario:{name}:{part}"] = artifacts.local_path(entry[part])
            if (entry.get("gcm_ensemble") or {}).get("agreement"):
                out[f"scenario:{name}:gcm_agreement"] = artifacts.local_path(
                    entry["gcm_ensemble"]["agreement"]
                )
        elif isinstance(entry, str):
            out[f"scenario:{name}:suitability"] = artifacts.local_path(entry)
    return {k: p for k, p in out.items() if p.exists()}


def binary_path(mv: ModelVersionRecord, artifacts: ArtifactStore, rule: str) -> Path:
    """Binary map at one of the version's threshold rules (written once, then cached)."""
    thresholds = mv.metrics.get("thresholds") or {}
    if rule not in thresholds:
        raise KeyError(rule)
    vdir = artifacts.local_path(str(artifacts.version_dir(mv.taxon_key, mv.model_version)))
    path = vdir / f"binary_{rule}.tif"
    if not path.exists():
        write_binary(
            artifacts.local_path(mv.artifacts["suitability"]),
            float(thresholds[rule]["value"]),
            path,
        )
    return path


def occurrences_gpkg(mv: ModelVersionRecord, artifacts: ArtifactStore, out: Path) -> Path:
    """The version's occurrence manifest (effective labels) joined with its quality flags,
    plus the training set (presences, background, pseudo-absences, survey records)."""
    a = mv.artifacts
    manifest = pd.read_parquet(artifacts.local_path(a["occurrence_manifest"]))
    if a.get("quality_flags") and artifacts.local_path(a["quality_flags"]).exists():
        flags = pd.read_parquet(artifacts.local_path(a["quality_flags"]))
        keep = [c for c in flags.columns if c not in ("longitude", "latitude", "event_date")]
        flags = flags[keep].astype({"record_id": "Int64"}) if "record_id" in flags else flags
        if {"source", "record_id"} <= set(flags.columns):
            manifest = manifest.merge(flags, on=["source", "record_id"], how="left")
    for col in manifest.columns:
        if manifest[col].dtype == object:
            manifest[col] = manifest[col].map(
                lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v
            )
    occ = gpd.GeoDataFrame(
        manifest,
        geometry=gpd.points_from_xy(manifest["longitude"], manifest["latitude"]),
        crs="EPSG:4326",
    )
    training = pd.read_parquet(artifacts.local_path(a["training_data"]))
    train = gpd.GeoDataFrame(
        training,
        geometry=gpd.points_from_xy(training["longitude"], training["latitude"]),
        crs="EPSG:4326",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    occ.to_file(out, layer="occurrences", driver="GPKG", engine="pyogrio")
    train.to_file(out, layer="training_data", driver="GPKG", engine="pyogrio", append=True)
    return out


def citation_bundle(sp: SpeciesRecord, mv: ModelVersionRecord) -> str:
    repro = mv.metrics.get("reproducibility") or {}
    lines = [
        f"RangeWatch model {sp.scientific_name} (GBIF taxonKey {sp.taxon_key}), version "
        f"v{mv.model_version}, trained {mv.trained_ts:%Y-%m-%d}.",
        "",
        "Occurrence data",
        f"  {repro.get('gbif_citation', 'GBIF.org occurrence data')}",
    ]
    if repro.get("inat_citation"):
        lines.append(f"  {repro['inat_citation']}")
    for up in repro.get("training_uploads") or []:
        cite = up.get("citation") or f"uploaded dataset {up.get('name')}"
        lines.append(f"  {cite} (file SHA-256 {up.get('file_sha256') or 'n/a'})")
    lines += [
        "",
        "Climate data",
        f"  {repro.get('bioclim_source') or repro.get('bioclim_version_used')} "
        f"({repro.get('bioclim_version_used')}, SHA-256 {repro.get('bioclim_sha256')})",
        "",
        "Software",
        *(f"  {k} {v}" for k, v in (repro.get("package_versions") or {}).items()),
        "  elapid: Anderson C. B. (2023). elapid: Species distribution modeling tools for "
        "Python. Journal of Open Source Software 8(84), 4930.",
        "",
        "Training data snapshot",
        f"  training_data.parquet SHA-256 {(repro.get('training_data') or {}).get('sha256')}",
        "  occurrence_manifest.parquet SHA-256 "
        f"{(repro.get('occurrence_manifest') or {}).get('sha256')}",
        "",
        "Caveat",
        f"  {mv.metrics.get('transferability_caveat', '')}",
    ]
    return "\n".join(lines) + "\n"


def _first(d: dict[str, Any] | None, *keys: str) -> Any:
    for k in keys:
        if d and d.get(k) is not None:
            return d[k]
    return None


def _baseline_year(repro: dict[str, Any]) -> Any:
    cfg = (repro.get("data_quality") or {}).get("config") or {}
    return cfg.get("baseline_start_year", "any year")


def odmap(sp: SpeciesRecord, mv: ModelVersionRecord) -> dict[str, Any]:
    """ODMAP sections (Zurell et al. 2020) filled from the version's stored metadata."""
    m = mv.metrics
    repro = m.get("reproducibility") or {}
    cfg = repro.get("training_config") or {}
    proj = m.get("projection") or {}
    algo = m.get("algorithm", "maxent")
    sections = [
        (
            "Overview",
            {
                "authorship": "RangeWatch automated pipeline",
                "model_objective": "mapping and transfer: native range, established "
                "introductions and candidate expansion zones",
                "focal_taxon": f"{sp.scientific_name} (GBIF taxonKey {sp.taxon_key})",
                "location": "global" if not repro.get("study_region") else repro["study_region"],
                "spatial_extent_resolution": repro.get("study_region")
                or f"global, {repro.get('bioclim_version_used')}",
                "temporal_extent": "current climate baseline of the bioclim version; records "
                f"from {_baseline_year(repro)} on",
                "biodiversity_data_type": "presence-only"
                + (" + presence–absence surveys" if algo == "integrated" else "")
                + (" + pseudo-absences" if algo == "ensemble" else ""),
                "predictor_types": "climatic (WorldClim/CHELSA bioclimatic variables)",
                "algorithms": algo,
                "model_type": f"{mv.model_type} ({m.get('confidence_label')} confidence)",
                "software": repro.get("package_versions"),
                "workflow": "GBIF/iNaturalist ingestion → data-quality tests → expert "
                "annotations → thinning → background / pseudo-absences → spatial block CV "
                "tuning → projection with MESS/exDet/MOP/Shape/AOA",
                "trigger": mv.trigger,
            },
        ),
        (
            "Data",
            {
                "occurrence_source": repro.get("gbif_citation"),
                "inaturalist": repro.get("inat_citation"),
                "training_uploads": repro.get("training_uploads"),
                "data_cleaning": repro.get("data_quality"),
                "annotations": repro.get("annotations"),
                "n_presence": m.get("n_presence"),
                "n_records_by_effective_label": m.get("n_records_by_effective_label"),
                "background": repro.get("background"),
                "pseudo_absences": repro.get("pseudo_absences"),
                "absence_data": (
                    "none (presence-background)"
                    if algo == "maxent"
                    else (
                        "training uploads with absences (survey component)"
                        if algo == "integrated"
                        else "pseudo-absences"
                    )
                ),
                "n_background": m.get("n_background"),
                "data_partitioning": m.get("cv_method"),
                "predictors": m.get("predictors"),
                "predictor_source": repro.get("bioclim_source"),
                "bioclim_version": repro.get("bioclim_version_used"),
                "sampling_bias": {
                    "presence_weights": repro.get("presence_weights"),
                    "bias_covariates": repro.get("bias_covariates"),
                    "thinning_km": cfg.get("thin_km"),
                },
            },
        ),
        (
            "Model",
            {
                "variable_preselection": "VIF-based selection at stack build time",
                "feature_classes": m.get("feature_classes"),
                "regularisation": m.get("beta_multiplier"),
                "model_selection": "spatial-CV CBI (AUC tie-break) over the Wallace grid",
                "tuning": m.get("tuning"),
                "ensemble": m.get("ensemble"),
                "integrated": m.get("integrated"),
                "threshold_selection": {
                    "used": m.get("threshold_rule"),
                    "value": m.get("threshold"),
                    "all_rules": m.get("thresholds"),
                },
                "variable_importance": m.get("variable_importance"),
                "random_seeds": repro.get("random_seeds"),
                "training_config": cfg,
            },
        ),
        (
            "Assessment",
            {
                "performance_statistics": {
                    k: m.get(k)
                    for k in ("auc_mean", "auc_std", "cbi_mean", "cbi_std", "tss_mean", "tss_std")
                },
                "folds": m.get("folds"),
                "independent_evaluation": "see reference datasets of this version",
                "inferred_absences": m.get("inferred_absences"),
                "plausibility_check": "response curves and limiting-factor map",
                "response_curves_method": _first(m.get("response_curves"), "method"),
            },
        ),
        (
            "Prediction",
            {
                "prediction_unit": (
                    "relative suitability (cloglog)"
                    if algo != "integrated"
                    else "probability of detection in one survey unit"
                ),
                "extrapolation": proj.get("extrapolation"),
                "candidate_area_km2": proj.get("candidate_area_km2"),
                "uncertainty_quantification": "MESS, exDet, MOP, Shape, AOA (+ consensus)"
                + ("; ensemble agreement" if algo == "ensemble" else ""),
                "change_previous": m.get("change_previous"),
                "transferability_caveat": m.get("transferability_caveat"),
            },
        ),
    ]
    return {
        "protocol": "ODMAP v1.0 (Zurell et al. 2020, Ecography 43:1261–1277)",
        "taxon_key": sp.taxon_key,
        "model_version": mv.model_version,
        "generated_ts": datetime.now(UTC),
        "sections": [{"section": s, "entries": e} for s, e in sections],
    }


def odmap_markdown(report: dict[str, Any]) -> str:
    out = [f"# ODMAP protocol — taxon {report['taxon_key']}, v{report['model_version']}", ""]
    out.append(f"*{report['protocol']}; generated {report['generated_ts']:%Y-%m-%d %H:%M} UTC*")
    for sec in report["sections"]:
        out += ["", f"## {sec['section']}", ""]
        for k, v in sec["entries"].items():
            text = v if isinstance(v, str) else json.dumps(v, default=str)
            out.append(f"- **{k.replace('_', ' ')}**: {text}")
    return "\n".join(out) + "\n"


def stac_item(
    sp: SpeciesRecord,
    mv: ModelVersionRecord,
    rasters: dict[str, Path],
    base_url: str,
) -> dict[str, Any]:
    repro = mv.metrics.get("reproducibility") or {}
    region = repro.get("study_region")
    bbox = region["grid_bbox"] if region else [-180.0, -90.0, 180.0, 90.0]
    w, s, e, n = bbox
    prefix = f"{base_url}/species/{sp.taxon_key}/models/{mv.model_version}/exports"
    assets = {}
    for key in rasters:
        label, desc, roles = RASTERS.get(key.split(":")[0], (key, key, ["data"]))
        if key.startswith("scenario:"):
            _, name, part = key.split(":")
            label = f"{part} — {name}"
        assets[key] = {
            "href": f"{prefix}/raster/{key}",
            "type": COG_TYPE,
            "title": label,
            "roles": roles,
        }
    for name, media, title, roles in (
        ("occurrences.gpkg", "application/geopackage+sqlite3", "Training records", ["data"]),
        ("odmap", "application/json", "ODMAP protocol", ["metadata"]),
        ("citation", "text/plain", "Citation bundle", ["metadata"]),
    ):
        assets[name.split(".")[0]] = {
            "href": f"{prefix}/{name}",
            "type": media,
            "title": title,
            "roles": roles,
        }
    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": ["https://stac-extensions.github.io/projection/v1.1.0/schema.json"],
        "id": f"rangewatch-{sp.taxon_key}-v{mv.model_version}",
        "bbox": bbox,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
        },
        "properties": {
            "datetime": mv.trained_ts.isoformat(),
            "title": f"{sp.scientific_name} distribution model v{mv.model_version}",
            "proj:epsg": 4326,
            "rangewatch:taxon_key": sp.taxon_key,
            "rangewatch:scientific_name": sp.scientific_name,
            "rangewatch:model_type": mv.model_type,
            "rangewatch:algorithm": mv.metrics.get("algorithm", "maxent"),
            "rangewatch:confidence": mv.metrics.get("confidence_label"),
            "rangewatch:threshold": mv.metrics.get("threshold"),
            "rangewatch:threshold_rule": mv.metrics.get("threshold_rule"),
            "rangewatch:bioclim_version": repro.get("bioclim_version_used"),
            "rangewatch:gbif_download_doi": repro.get("gbif_download_doi"),
            "rangewatch:transferability_caveat": mv.metrics.get("transferability_caveat"),
        },
        "links": [
            {"rel": "self", "href": f"{prefix}/stac", "type": "application/json"},
            *(
                [{"rel": "cite-as", "href": f"https://doi.org/{repro['gbif_download_doi']}"}]
                if repro.get("gbif_download_doi")
                else []
            ),
        ],
        "assets": assets,
    }


def bundle_zip(
    sp: SpeciesRecord,
    mv: ModelVersionRecord,
    artifacts: ArtifactStore,
    rasters: dict[str, Path],
    base_url: str,
    workdir: Path,
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for key, path in rasters.items():
            z.write(path, f"rasters/{key.replace(':', '_')}.tif")
        gpkg = occurrences_gpkg(mv, artifacts, workdir / "occurrences.gpkg")
        z.write(gpkg, "occurrences.gpkg")
        report = odmap(sp, mv)
        z.writestr("odmap.json", json.dumps(report, indent=2, default=str))
        z.writestr("odmap.md", odmap_markdown(report))
        z.writestr("stac.json", json.dumps(stac_item(sp, mv, rasters, base_url), indent=2))
        z.writestr("CITATION.txt", citation_bundle(sp, mv))
        z.writestr("MODEL_CARD.md", model_card_markdown(model_card(sp, mv)))
    return buf.getvalue()
