"""Workplan phase 11 — validation on well-documented invasive species.

Two complementary checks per reference species:

1. **Known-range sanity check** on the published (operational) model: land in documented
   invaded regions should be predicted suitable; control regions the species cannot occupy
   should not. This is a plausibility check — for Model B the invaded regions contributed
   training data, so it is *not* an independent test.
2. **Native-only transferability test** (independent): a Model A trained on native-range
   records only predicts the introduced records against background sampled around them. This
   is the cross-continental test behind the literature's mean AUC ≈ 0.7 caveat.

Native-range polygons for the suite are deliberately coarse literature outlines, supplied via
the curated-polygon route (`data/native_range_polygons/<taxon_key>.geojson`) and shown with
their source in the UI. They never replace a polygon a user has already edited or confirmed.

    python -m src.orchestration.validation_suite run [--species 2925303 ...] [--no-train]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np

from src.domain import NativeRange, utcnow
from src.features.background_sampler import (
    cell_dedupe,
    distance_thin,
    sample_buffered_background,
)
from src.features.raster_sampler import BioclimStack, raster_window_stats
from src.modeling.evaluation import auc, continuous_boyce_index
from src.modeling.maxent_trainer import TrainingData, train_maxent
from src.modeling.native_range import effective_range_labels, validate_geojson_polygon
from src.orchestration.pipeline import SpeciesPipeline

log = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]
LITERATURE_TRANSFER_AUC = 0.7
SUITABLE_MIN_FRACTION = 0.25  # invaded region "captured" if ≥ 25 % of its land is suitable
UNSUITABLE_MAX_FRACTION = 0.10  # control region "correct" if ≤ 10 % of its land is suitable
MIN_INTRODUCED_FOR_TRANSFER = 20


@dataclass(frozen=True)
class Region:
    name: str
    bbox: BBox
    expect: Literal["suitable", "unsuitable"]
    ref: str


@dataclass(frozen=True)
class ReferenceSpecies:
    scientific_name: str
    taxon_key: int
    native_outline: tuple[tuple[float, float], ...]
    native_ref: str
    regions: tuple[Region, ...]


SAHARA = Region("Central Sahara", (-5, 18, 25, 28), "unsuitable", "hyper-arid; no records")
SIBERIA = Region("Central Siberia", (80, 55, 110, 65), "unsuitable", "boreal/continental cold")
AMAZON = Region("Central Amazon", (-70, -10, -55, 0), "unsuitable", "humid tropics")
BOREAL_CANADA = Region("Boreal Canada", (-110, 55, -80, 65), "unsuitable", "boreal cold")

REFERENCE_SPECIES: tuple[ReferenceSpecies, ...] = (
    ReferenceSpecies(
        "Linepithema humile",
        1316908,
        ((-66, -38), (-47, -36), (-44, -24), (-50, -14), (-62, -15), (-66, -25)),
        "Paraná–Uruguay basins (Wild 2004; Tsutsui et al. 2001)",
        (
            Region(
                "Mediterranean Iberia & Italy",
                (-9, 36, 15, 44),
                "suitable",
                "Giraud et al. 2002 (European supercolony)",
            ),
            Region("Coastal California", (-124, 32, -116, 39), "suitable", "Suarez et al. 2001"),
            Region("Western Cape", (17.5, -35, 21, -32.5), "suitable", "Luruli 2007"),
            Region("SE Australia", (138, -39, 146, -34), "suitable", "Suhr et al. 2011"),
            SIBERIA,
            SAHARA,
        ),
    ),
    ReferenceSpecies(
        "Lantana camara",
        2925303,
        (
            (-106, 20),
            (-97, 26),
            (-80, 25),
            (-60, 18),
            (-50, 5),
            (-35, -8),
            (-40, -25),
            (-58, -28),
            (-70, -15),
            (-80, -5),
            (-92, 12),
        ),
        "Neotropics, Mexico to N Argentina (Day et al. 2003)",
        (
            Region("Eastern Australia", (148, -32, 154, -17), "suitable", "Taylor et al. 2012"),
            Region("Peninsular & N India", (73, 8, 88, 28), "suitable", "Mungi et al. 2020"),
            Region(
                "KwaZulu-Natal & Mpumalanga", (28, -32, 33, -24), "suitable", "Vardien et al. 2012"
            ),
            Region("East African highlands", (30, -4, 40, 3), "suitable", "Shackleton 2017"),
            BOREAL_CANADA,
            SIBERIA,
            SAHARA,
        ),
    ),
    ReferenceSpecies(
        "Ailanthus altissima",
        3190653,
        ((100, 22), (122, 22), (123, 30), (122, 40), (115, 42), (104, 38), (98, 30)),
        "Temperate China & Taiwan (Kowarik & Säumel 2007)",
        (
            Region(
                "Central & southern Europe", (0, 40, 20, 49), "suitable", "Kowarik & Säumel 2007"
            ),
            Region("Eastern USA", (-90, 33, -72, 42), "suitable", "Kasson et al. 2013"),
            Region("SE Australia", (144, -38, 152, -30), "suitable", "Kowarik & Säumel 2007"),
            AMAZON,
            SIBERIA,
            SAHARA,
        ),
    ),
    ReferenceSpecies(
        "Vespa velutina",
        1311477,
        (
            (72, 30),
            (90, 29),
            (105, 33),
            (122, 31),
            (122, 22),
            (110, 5),
            (115, -9),
            (95, -7),
            (95, 10),
            (88, 21),
            (72, 23),
        ),
        "N India to S China, Indochina & Indonesia (Archer 1994)",
        (
            Region("France", (-2, 43, 7, 49), "suitable", "Monceau et al. 2014"),
            Region("N Iberia", (-9, 41, 3, 44), "suitable", "Barbet-Massin et al. 2013"),
            Region("South Korea", (126, 34, 130, 38), "suitable", "Choi et al. 2012"),
            SAHARA,
            SIBERIA,
            Region("Australian arid interior", (125, -28, 140, -20), "unsuitable", "arid"),
        ),
    ),
    ReferenceSpecies(
        "Pueraria montana",
        2977636,
        (
            (100, 20),
            (122, 20),
            (132, 33),
            (142, 42),
            (140, 45),
            (128, 43),
            (118, 42),
            (104, 35),
            (98, 27),
        ),
        "China, Korea & Japan (Forseth & Innis 2004)",
        (
            Region("SE USA", (-95, 30, -76, 38), "suitable", "Forseth & Innis 2004"),
            SAHARA,
            SIBERIA,
        ),
    ),
    ReferenceSpecies(
        "Carpobrotus edulis",
        3084842,
        ((17, -35), (26, -35), (28, -33), (24, -30), (18, -28)),
        "Cape region, South Africa (Campoy et al. 2018)",
        (
            Region("Mediterranean coasts", (-9, 36, 16, 44), "suitable", "Campoy et al. 2018"),
            Region("Coastal California", (-124, 32, -117, 39), "suitable", "Campoy et al. 2018"),
            Region("SE Australia", (138, -39, 151, -33), "suitable", "Campoy et al. 2018"),
            SIBERIA,
            AMAZON,
            SAHARA,
        ),
    ),
)


def reference_polygon_geojson(sp: ReferenceSpecies) -> dict[str, Any]:
    ring = [list(p) for p in sp.native_outline] + [list(sp.native_outline[0])]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "status": "confirmed",
                    "source": f"validation-suite: {sp.native_ref}",
                    "note": "Coarse literature outline used by the validation suite. Review and "
                    "refine in the native-range editor before operational use.",
                },
            }
        ],
    }


def install_reference_polygons(pipeline: SpeciesPipeline, species: list[ReferenceSpecies]) -> None:
    """Write curated polygon files; apply them to already-registered species only when the
    stored range is still an unreviewed heuristic draft (user edits are never overridden)."""
    root = pipeline.settings.native_range_root
    root.mkdir(parents=True, exist_ok=True)
    for sp in species:
        fc = reference_polygon_geojson(sp)
        path = root / f"{sp.taxon_key}.geojson"
        if not path.exists():
            path.write_text(json.dumps(fc, indent=1), encoding="utf-8")
        if pipeline.repo.get_species(sp.taxon_key) is None:
            continue
        nr = pipeline.repo.get_native_range(sp.taxon_key)
        if nr is None or (nr.status == "draft" and nr.source.startswith("heuristic:")):
            props = fc["features"][0]["properties"]
            pipeline.repo.save_native_range(
                NativeRange(
                    taxon_key=sp.taxon_key,
                    geometry=validate_geojson_polygon(fc["features"][0]["geometry"]),
                    status="confirmed",
                    source=f"curated:{props['source']}",
                    note=props["note"],
                    confirmed_ts=utcnow(),
                )
            )


def evaluate_regions(
    suitability_path: Any, threshold: float, regions: tuple[Region, ...]
) -> list[dict[str, Any]]:
    out = []
    for r in regions:
        st = raster_window_stats(suitability_path, r.bbox, threshold)
        frac = st["fraction_at_or_above"]
        ok = (
            (
                frac >= SUITABLE_MIN_FRACTION
                if r.expect == "suitable"
                else frac <= UNSUITABLE_MAX_FRACTION
            )
            if frac == frac
            else False
        )  # NaN → no land in box
        out.append(
            {
                "name": r.name,
                "bbox": list(r.bbox),
                "expect": r.expect,
                "ref": r.ref,
                "fraction_suitable": frac,
                "mean_suitability": st["mean"],
                "passed": ok,
            }
        )
    return out


def transferability_check(pipeline: SpeciesPipeline, taxon_key: int) -> dict[str, Any] | None:
    """Train a native-only Model A and score it on the introduced records (independent test)."""
    species = pipeline.repo.get_species(taxon_key)
    mv = pipeline.repo.get_model_version(taxon_key, species.model_version) if species else None
    nr = pipeline.repo.get_native_range(taxon_key)
    if mv is None or nr is None:
        return None
    m = mv.metrics
    stack = BioclimStack.open_version(
        pipeline.settings.bioclim_root, m["reproducibility"]["bioclim_version_used"]
    )
    predictors = list(m["predictors"])
    occ = pipeline.repo.load_occurrences(taxon_key)
    feats = stack.sample(occ["longitude"], occ["latitude"], predictors)
    occ = occ.drop(columns=[c for c in predictors if c in occ.columns]).join(feats)
    occ = occ[~occ[predictors].isna().any(axis=1)].reset_index(drop=True)
    labels = effective_range_labels(occ, nr.geometry)
    cfg = replace(
        pipeline.training_config,
        tune=False,
        feature_classes=m["feature_classes"],
        beta_multiplier=m["beta_multiplier"],
    )

    def thinned(df):
        df = df.iloc[cell_dedupe(df["longitude"], df["latitude"], stack.transform)]
        df = df.reset_index(drop=True)
        keep = distance_thin(df["longitude"], df["latitude"], cfg.thin_km, seed=cfg.seed)
        return df.iloc[keep].reset_index(drop=True)

    native = thinned(occ[labels == "native"])
    intro = thinned(occ[labels == "introduced"])
    if len(native) < cfg.min_presences or len(intro) < MIN_INTRODUCED_FOR_TRANSFER:
        return {"skipped": f"{len(native)} native / {len(intro)} introduced thinned records"}
    bg = sample_buffered_background(
        native["longitude"],
        native["latitude"],
        stack,
        cfg.n_background,
        cfg.background_buffer_km,
        cfg.seed + 1,
        predictors,
    )
    data = TrainingData(
        predictors,
        native["longitude"].to_numpy(),
        native["latitude"].to_numpy(),
        native[predictors],
        bg.lon,
        bg.lat,
        bg.features,
        bg.method,
    )
    result = train_maxent(data, "A", cfg)
    eval_bg = sample_buffered_background(
        intro["longitude"],
        intro["latitude"],
        stack,
        5000,
        cfg.background_buffer_km,
        cfg.seed + 7,
        predictors,
    )
    p = np.asarray(result.model.predict(intro[predictors])).ravel()
    b = np.asarray(result.model.predict(eval_bg.features[predictors])).ravel()
    return {
        "auc": round(auc(p, b), 4),
        "cbi": round(continuous_boyce_index(p, b), 4),
        "n_native_train": len(native),
        "n_introduced_test": len(intro),
        "literature_mean_auc": LITERATURE_TRANSFER_AUC,
        "method": "Model A trained on native records only; introduced presences vs background "
        f"within {cfg.background_buffer_km:.0f} km of them",
    }


def run_suite(
    pipeline: SpeciesPipeline, taxon_keys: list[int] | None = None, train: bool = True
) -> dict[str, Any]:
    species = [s for s in REFERENCE_SPECIES if not taxon_keys or s.taxon_key in taxon_keys]
    install_reference_polygons(pipeline, species)
    results = []
    for i, sp in enumerate(species):
        pipeline.progress("validation", i / len(species), f"{sp.scientific_name}")
        entry: dict[str, Any] = {
            "taxon_key": sp.taxon_key,
            "scientific_name": sp.scientific_name,
            "native_range_reference": sp.native_ref,
        }
        try:
            reg = pipeline.repo.get_species(sp.taxon_key)
            if train and (reg is None or not reg.model_version):
                pipeline.run(sp.taxon_key)
                install_reference_polygons(pipeline, [sp])
                reg = pipeline.repo.get_species(sp.taxon_key)
                if reg is not None and not reg.model_version:
                    pipeline.run(sp.taxon_key)
            reg = pipeline.repo.get_species(sp.taxon_key)
            if reg is None or not reg.model_version:
                entry.update(status="no_model", passed=False)
                results.append(entry)
                continue
            mv = pipeline.repo.get_model_version(sp.taxon_key, reg.model_version)
            assert mv is not None
            suit = pipeline.artifacts.local_path(mv.artifacts["suitability"])
            regions = evaluate_regions(suit, float(mv.metrics["threshold"]), sp.regions)
            transfer = transferability_check(pipeline, sp.taxon_key)
            invaded = [r for r in regions if r["expect"] == "suitable"]
            controls = [r for r in regions if r["expect"] == "unsuitable"]
            captured = sum(r["passed"] for r in invaded)
            entry.update(
                status="evaluated",
                model_version=mv.model_version,
                model_type=mv.model_type,
                bioclim_version=mv.metrics["reproducibility"]["bioclim_version_used"],
                auc_mean=mv.metrics.get("auc_mean"),
                cbi_mean=mv.metrics.get("cbi_mean"),
                tss_mean=mv.metrics.get("tss_mean"),
                n_occurrences=reg.n_occurrences_total,
                regions=regions,
                invaded_captured=captured,
                invaded_total=len(invaded),
                controls_correct=sum(r["passed"] for r in controls),
                controls_total=len(controls),
                transferability=transfer,
                passed=bool(invaded)
                and captured / len(invaded) >= 0.75
                and all(r["passed"] for r in controls),
            )
        except Exception as exc:  # one species must not abort the suite
            log.exception("Validation failed for %s", sp.scientific_name)
            entry.update(status="error", error=f"{type(exc).__name__}: {exc}", passed=False)
        results.append(entry)

    transfer_aucs = [
        r["transferability"]["auc"]
        for r in results
        if isinstance(r.get("transferability"), dict) and "auc" in r["transferability"]
    ]
    report = {
        "generated_ts": datetime.now(UTC).isoformat(),
        "criteria": {
            "invaded_region_captured_if_fraction_suitable_at_least": SUITABLE_MIN_FRACTION,
            "control_region_correct_if_fraction_suitable_at_most": UNSUITABLE_MAX_FRACTION,
            "species_passes_if": "≥75% of invaded regions captured and all controls correct",
            "caveat": "Region checks on Model B are plausibility checks (invaded records were "
            "used in training); the native-only transferability AUC is the "
            "independent test.",
        },
        "species": results,
        "summary": {
            "n_species": len(results),
            "n_passed": sum(bool(r.get("passed")) for r in results),
            "mean_transfer_auc": round(float(np.mean(transfer_aucs)), 4) if transfer_aucs else None,
            "literature_mean_transfer_auc": LITERATURE_TRANSFER_AUC,
        },
    }
    write_report(pipeline, report)
    pipeline.progress("done", 1.0, f"{report['summary']['n_passed']}/{len(results)} passed")
    return report


def write_report(pipeline: SpeciesPipeline, report: dict[str, Any]) -> None:
    root = pipeline.artifacts.local_path("validation")
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    text = json.dumps(report, indent=2, default=str)
    (root / f"validation_{stamp}.json").write_text(text, encoding="utf-8")
    (root / "latest.json").write_text(text, encoding="utf-8")


def main() -> None:
    from src.orchestration.factory import build_pipeline

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--species", type=int, nargs="*", help="taxon keys (default: all)")
    r.add_argument("--no-train", action="store_true", help="only evaluate existing models")
    args = ap.parse_args()
    report = run_suite(build_pipeline(), args.species, train=not args.no_train)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
