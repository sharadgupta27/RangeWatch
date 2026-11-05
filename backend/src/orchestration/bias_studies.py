"""Sampling-bias studies run as jobs (improvements.md B1, B9).

* `compare_target_groups` (B1): refit a species' model with target-group backgrounds of
  different taxonomic ranks, with and without matching the presences' data sources and time
  window, and compare them by spatial-CV CBI / AUC and on independent reference data.
  Nothing is published; the report is stored per species.
* `estimate_shared_bias` (B9; Fithian et al. 2015): one observer-bias estimate for all
  registered species of a target group — a joint point process with species-specific climate
  responses and shared bias coefficients — reused as fixed offsets by `ppm_bias="shared"`.

Both reuse the pipeline's own steps (quality tests, annotations, thinning, background,
spatial CV), so a comparison differs from a normal training run only in the option compared.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.domain import utcnow
from src.features.background_sampler import sample_buffered_background
from src.features.raster_sampler import BioclimStack, sample_bias_layer
from src.features.sampling_bias import EFFORT_COVARIATE, effort_values
from src.modeling.evaluation import independent_evaluation
from src.modeling.point_process import SpeciesBiasData, fit_shared_bias
from src.orchestration.pipeline import SpeciesPipeline, shared_bias_path

log = logging.getLogger(__name__)

TARGET_GROUP_COMPARISON_DIR = "target_group_comparisons"
COMPARISON_RANKS = ("order", "class", "phylum")
SHARED_BIAS_QUADRATURE = 5_000


def _finite(v: Any) -> float | None:
    return float(v) if v is not None and np.isfinite(v) else None


def _write(pipeline: SpeciesPipeline, rel: str, report: dict[str, Any]) -> Path:
    path = pipeline.artifacts.local_path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def comparison_path(taxon_key: int) -> str:
    return f"{TARGET_GROUP_COMPARISON_DIR}/{int(taxon_key)}.json"


def load_report(pipeline_or_artifacts: Any, rel: str) -> dict[str, Any] | None:
    artifacts = getattr(pipeline_or_artifacts, "artifacts", pipeline_or_artifacts)
    path = artifacts.local_path(rel)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# B1: target-group comparison
# ---------------------------------------------------------------------------
def _independent_scores(
    pipeline: SpeciesPipeline, taxon_key: int, result: Any, info: dict[str, Any], stack: Any
) -> list[dict[str, Any]]:
    """The fitted model scored on reference datasets independent of its training data."""
    out = []
    data = info["training_data"]
    bg_pred = np.asarray(result.model.predict(data.background_x[result.predictors]), "float64")
    for ds in pipeline.repo.list_reference_datasets(taxon_key):
        recs = pipeline.repo.load_reference_records(taxon_key, ds.dataset_id)
        if ds.role == "training":
            recs = recs[recs["occurrence_status"] == "absent"]
        if not len(recs):
            continue
        x = stack.sample(recs["longitude"], recs["latitude"], result.predictors)
        ok = ~x.isna().any(axis=1).to_numpy()
        pred = np.full(len(recs), np.nan)
        if ok.any():
            pred[ok] = np.asarray(result.model.predict(x[ok]), "float64")
        status = recs["occurrence_status"].to_numpy()
        metrics = independent_evaluation(
            pred[ok & (status == "present")],
            pred[ok & (status == "absent")],
            bg_pred,
            result.threshold,
        )
        out.append({"dataset_id": ds.dataset_id, "name": ds.name, "metrics": metrics})
    return out


def compare_target_groups(
    pipeline: SpeciesPipeline,
    taxon_key: int,
    ranks: tuple[str, ...] = COMPARISON_RANKS,
    include_matched: bool = True,
) -> dict[str, Any]:
    species = pipeline.repo.get_species(taxon_key)
    nr = pipeline.repo.get_native_range(taxon_key)
    if species is None or nr is None or nr.status != "confirmed":
        raise RuntimeError("The comparison needs a registered species with a confirmed range")
    base_cfg = pipeline.training_config
    stack, region = pipeline.training_stack(species)
    occ = pipeline.extract_features(taxon_key, stack)
    options: list[dict[str, Any]] = [{"rank": r, "match": False} for r in ranks]
    if include_matched:
        options.append({"rank": base_cfg.target_group_rank.lower(), "match": True})
    rows: list[dict[str, Any]] = []
    try:
        for i, opt in enumerate(options):
            label = f"{opt['rank']}" + (" (matched source + years)" if opt["match"] else "")
            pipeline.progress("target_groups", i / len(options), f"Fitting with {label}")
            pipeline.training_config = replace(
                base_cfg,
                background_method="target_group",
                target_group_rank=opt["rank"],
                target_group_match_source=opt["match"],
                target_group_match_years=opt["match"],
                bias_covariate="none",
                bias_static_layers=(),
                algorithm="maxent",
                tune=False,
                limiting_factor=False,
            )
            row: dict[str, Any] = {"option": label, **opt}
            try:
                result, info = pipeline.fit_model(occ, nr, stack, region)
            except Exception as exc:  # one option failing must not lose the others
                log.warning("Target-group option %s failed: %s", label, exc)
                rows.append({**row, "error": f"{type(exc).__name__}: {exc}"})
                continue
            bg = info["background"]
            group = bg.get("target_group") or {}
            row.update(
                background_method=bg.get("method"),
                fallback_reason=bg.get("fallback_reason"),
                target_group_name=group.get("name"),
                target_group_taxon_key=group.get("taxon_key"),
                target_group_records=group.get("n_records"),
                matched=group.get("matched"),
                n_presence=result.n_presence,
                cv_auc_mean=_finite(result.cv_summary.get("auc_mean")),
                cv_cbi_mean=_finite(result.cv_summary.get("cbi_mean")),
                independent=_independent_scores(pipeline, taxon_key, result, info, stack),
            )
            rows.append(row)
    finally:
        pipeline.training_config = base_cfg

    def score(r: dict[str, Any]) -> float:
        v = r.get("cv_cbi_mean")
        return float(v) if v is not None and np.isfinite(v) else -np.inf

    valid = [
        r
        for r in rows
        if "error" not in r and r.get("background_method", "").startswith("target_group")
    ]
    best = max(valid, key=score)["option"] if valid else None
    report = {
        "comparison_id": uuid.uuid4().hex[:12],
        "taxon_key": taxon_key,
        "created_ts": utcnow().isoformat(),
        "bioclim_version": stack.version,
        "configured_rank": base_cfg.target_group_rank,
        "options": rows,
        "best_by_cv_cbi": best,
        "criterion": "highest mean spatial-CV CBI among options that used a target-group "
        "background; check the independent reference data before changing "
        "SDM_TARGET_GROUP_RANK (CV on biased data can reward a biased background)",
        "note": "fitted with the configured feature classes (no tuning), MaxEnt, no bias "
        "covariate; nothing was published",
        "reference": "Phillips et al. 2009; Botella et al. 2020",
    }
    _write(pipeline, comparison_path(taxon_key), report)
    pipeline.progress("done", 1.0, f"Target groups compared: best {best}")
    return report


# ---------------------------------------------------------------------------
# B9: shared bias of a target group
# ---------------------------------------------------------------------------
def _group_of(pipeline: SpeciesPipeline, taxon_key: int, rank: str) -> tuple[int, str] | None:
    return pipeline.gbif.get_taxon(taxon_key).higher_taxon(rank)


def estimate_shared_bias(pipeline: SpeciesPipeline, taxon_key: int) -> dict[str, Any]:
    cfg = pipeline.training_config
    settings = pipeline.settings
    rank = cfg.target_group_rank.lower()
    group = _group_of(pipeline, taxon_key, rank)
    if group is None:
        raise ValueError(f"taxon {taxon_key} has no {rank} in the GBIF backbone")
    group_key, group_name = group
    pipeline.progress("shared_bias", 0.05, f"Registered species of {rank} {group_name}")
    peers: list[tuple[int, str, pd.DataFrame, BioclimStack, list[str]]] = []
    skipped: list[dict[str, Any]] = []
    for sp in pipeline.repo.list_species():
        if not sp.model_version:
            continue
        mv = pipeline.repo.get_model_version(sp.taxon_key, sp.model_version)
        if mv is None:
            continue
        repro = mv.metrics.get("reproducibility") or {}
        known = ((repro.get("background") or {}).get("target_group") or {}).get("taxon_key")
        if (
            known is None
            or (repro.get("background") or {}).get("target_group", {}).get("rank", rank) != rank
        ):
            try:
                g = _group_of(pipeline, sp.taxon_key, rank)
            except Exception as exc:  # one lookup failing must not stop the estimate
                skipped.append({"taxon_key": sp.taxon_key, "reason": f"taxonomy: {exc}"})
                continue
            known = g[0] if g else None
        if known != group_key:
            continue
        snap = repro.get("training_data")
        path = pipeline.artifacts.local_path(snap["path"]) if snap else None
        if path is None or not path.exists():
            skipped.append({"taxon_key": sp.taxon_key, "reason": "no training snapshot"})
            continue
        predictors = list(mv.metrics.get("predictors") or [])
        pres = pd.read_parquet(path)
        pres = pres[pres["role"] == "presence"][["longitude", "latitude", *predictors]]
        if len(pres) < cfg.min_presences:
            skipped.append(
                {"taxon_key": sp.taxon_key, "reason": f"only {len(pres)} training presences"}
            )
            continue
        stack = BioclimStack.open_version(settings.bioclim_root, repro["bioclim_version_used"])
        peers.append(
            (sp.taxon_key, sp.scientific_name, pres.reset_index(drop=True), stack, predictors)
        )
    if len(peers) < settings.shared_bias_min_species:
        raise ValueError(
            f"only {len(peers)} registered species of {rank} {group_name} have a trained model "
            f"with ≥ {cfg.min_presences} presences (need {settings.shared_bias_min_species})"
        )
    # Uniform quadrature per species: a target-group background would already contain the
    # bias that is being estimated.
    frames = []
    for i, (key, _, pres, stack, predictors) in enumerate(peers):
        pipeline.progress("shared_bias", 0.1 + 0.4 * i / len(peers), f"Quadrature for {key}")
        quad = sample_buffered_background(
            pres["longitude"],
            pres["latitude"],
            stack,
            min(cfg.n_background, SHARED_BIAS_QUADRATURE),
            cfg.background_buffer_km,
            cfg.seed + 5,
            predictors,
        )
        q = pd.DataFrame({"longitude": quad.lon, "latitude": quad.lat}).join(
            quad.features[predictors].reset_index(drop=True)
        )
        frames.append((pres, q))
    lon = np.concatenate([np.r_[p["longitude"], q["longitude"]] for p, q in frames])
    lat = np.concatenate([np.r_[p["latitude"], q["latitude"]] for p, q in frames])
    pad = abs(peers[0][3].res[0])
    bbox = (
        max(-180.0, float(lon.min()) - pad),
        max(-90.0, float(lat.min()) - pad),
        min(180.0, float(lon.max()) + pad),
        min(90.0, float(lat.max()) + pad),
    )
    pipeline.progress("shared_bias", 0.55, f"GBIF effort of {rank} {group_name}")
    eff = pipeline.gbif.fetch_effort_density(group_key, bbox, max_pixel_deg=pad)
    if not eff.lon:
        raise ValueError(f"no GBIF records of {rank} {group_name} in the species' regions")
    columns = [EFFORT_COVARIATE]
    static_used: list[dict[str, Any]] = []
    for name in cfg.bias_static_layers:
        # A static layer joins only when every species' bioclim version has it registered.
        if not all(name in stack.metadata.bias_layers for _, _, _, stack, _ in peers):
            continue
        columns.append(f"bias_{name}")
        static_used.append({"name": name, "covariate": f"bias_{name}"})
    data: list[SpeciesBiasData] = []
    for (key, _, _, stack, predictors), (pres, q) in zip(peers, frames, strict=True):
        parts = []
        for frame in (pres, q):
            frame = frame.reset_index(drop=True).copy()
            frame[EFFORT_COVARIATE] = effort_values(
                eff.lon, eff.lat, eff.count, eff.pixel_deg, frame["longitude"], frame["latitude"]
            )
            for s in static_used:
                values, _ = sample_bias_layer(
                    settings.bioclim_root,
                    stack.version,
                    s["name"],
                    frame["longitude"],
                    frame["latitude"],
                )
                frame[s["covariate"]] = np.where(np.isnan(values), np.nanmedian(values), values)
            parts.append(frame)
        data.append(SpeciesBiasData(key, predictors, parts[0], parts[1]))
    pipeline.progress("shared_bias", 0.7, f"Joint fit over {len(data)} species")
    fit = fit_shared_bias(data, columns, l2=cfg.ppm_l2)
    names = {key: name for key, name, *_ in peers}
    report = {
        "estimate_id": uuid.uuid4().hex[:12],
        "created_ts": utcnow().isoformat(),
        "target_group": {"rank": rank, "taxon_key": group_key, "name": group_name},
        **fit,
        "species": [{**s, "scientific_name": names.get(s["taxon_key"])} for s in fit["species"]],
        "skipped": skipped,
        "columns": columns,
        "static_layers": static_used,
        "effort_query": eff.query,
        "effort_pixel_deg": eff.pixel_deg,
        "quadrature": f"{min(cfg.n_background, SHARED_BIAS_QUADRATURE)} uniform points in "
        f"each species' {cfg.background_buffer_km:g} km buffer (seed {cfg.seed + 5})",
        "use": "SDM_ALGORITHM=ppm with SDM_PPM_BIAS=shared fixes these coefficients as the "
        "offset of every species of this target group",
    }
    _write(pipeline, shared_bias_path(rank, group_key), report)
    pipeline.progress("done", 1.0, f"Shared bias of {rank} {group_name} from {len(data)} species")
    return report
