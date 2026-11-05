"""Global projection of a trained model: suitability, MESS, and range-zone rasters (COGs).

Zone codes (zones.tif, uint8, nodata=255):
    0 = not suitable
    1 = suitable, inside the confirmed native range
    2 = suitable, outside native range, already occupied (within buffer of known records)
    3 = candidate invasion / expansion zone (suitable, outside native range, not yet occupied)

extrapolation.tif (uint8, nodata=255) is a 0/1 mask of MESS < 0, served as a toggleable map
overlay. MESS is always produced alongside suitability (CLAUDE.md constraint 2).

Models with an extrapolation reference also get the advanced diagnostics (float32,
nodata=-9999): exdet.tif (NT1 where < 0, else NT2), mop.tif (MOP distance), shape.tif (Shape
value) and aoa_di.tif (AOA dissimilarity index), plus consensus.tif (uint8, nodata=255): how
many of the five diagnostics (MESS, exDet, MOP, Shape, AOA) flag the cell as extrapolation.

Further products (U4, U3, U6, U12):
    limiting.tif   (uint8, nodata=255) the limiting factor (Elith et al. 2010): index of the
                   predictor whose change to its mean over the training presences would raise
                   suitability most; 254 = no single predictor raises it;
    agreement.tif  (float32) ensembles only: weighted share of members whose own binary call
                   agrees with the ensemble's (1 = unanimous);
    change_*.tif   (uint8, nodata=255) binary change between two surfaces on the same grid:
                   0 stable unsuitable, 1 gain, 2 loss, 3 stable suitable;
    priority.tif   (float32) survey priority = suitability × (1 − effort) × (0.5 + 0.5 ×
                   uncertainty), each term 0–1 (see `priority_map`);
    binary_*.tif   (uint8) suitable (1) / unsuitable (0) at one threshold, for exports.

Multi-GCM ensembles (U7, `gcm_ensemble`): scenarios registered as `<gcm>_<ssp>_<period>` are
grouped by SSP and period; each group with ≥ 2 GCMs gets
    suitability_ensemble_<ssp>_<period>.tif    unweighted mean suitability over the GCMs;
    extrapolation_ensemble_<ssp>_<period>.tif  1 where MESS < 0 under *any* GCM (conservative);
    agreement_ensemble_<ssp>_<period>.tif      (uint8) number of GCMs calling the cell
                                               suitable at the model threshold (0…n).
"""

from __future__ import annotations

import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import geometry_mask
from rasterio.shutil import copy as rio_copy
from rasterio.windows import Window

from src.domain import CHANGE_LABELS, ZONE_LABELS  # noqa: F401  (re-exported)
from src.features.background_sampler import buffer_region
from src.features.raster_sampler import (  # noqa: F401  (write_binary re-exported)
    BioclimStack,
    RasterBlock,
    cell_area_km2,
    write_binary,
)
from src.modeling.evaluation import (
    EXTRAPOLATION_DIAGNOSTICS,
    ExtrapolationDiagnostics,
    ExtrapolationReference,
    MessReference,
    extrapolation_diagnostics,
    mess,
)
from src.modeling.maxent_trainer import TrainingResult

FLOAT_NODATA = -9999.0
ZONE_NODATA = 255
_PREDICT_CHUNK = 200_000
_COG_OPTS = {"compress": "deflate", "blocksize": "512", "overview_resampling": "average"}
# Continuous diagnostic rasters (artifact key → file name) and the consensus count raster.
DIAGNOSTIC_RASTERS = {
    "exdet": "exdet.tif",
    "mop": "mop.tif",
    "shape": "shape.tif",
    "aoa": "aoa_di.tif",
}
CONSENSUS_RASTER = "consensus.tif"
LIMITING_RASTER = "limiting.tif"
AGREEMENT_RASTER = "agreement.tif"
PRIORITY_RASTER = "priority.tif"
LIMITING_NONE = 254


@dataclass
class ProjectionSummary:
    land_area_km2: float = 0.0
    nonnative_land_area_km2: float = 0.0
    native_suitable_area_km2: float = 0.0
    established_outside_area_km2: float = 0.0
    candidate_area_km2: float = 0.0
    candidate_suitability_area_sum: float = 0.0
    candidate_mess_ok_area_km2: float = 0.0
    candidate_envelope_area_km2: float = 0.0
    extrapolation_area_km2: float = 0.0
    candidate_by_tile: Counter[str] = field(default_factory=Counter)
    limiting_variables: Counter[str] = field(default_factory=Counter)
    # Advanced extrapolation diagnostics (only when the model has an extrapolation reference)
    extrapolation_config: dict[str, Any] | None = None
    diagnostic_land_area: Counter[str] = field(default_factory=Counter)
    diagnostic_candidate_area: Counter[str] = field(default_factory=Counter)
    consensus_land_area: list[float] = field(default_factory=lambda: [0.0] * 6)
    candidate_consensus_area: list[float] = field(default_factory=lambda: [0.0] * 6)
    exdet_univariate_area_km2: float = 0.0
    exdet_combinatorial_area_km2: float = 0.0
    combinatorial_variables: Counter[str] = field(default_factory=Counter)
    # Limiting factor (area per predictor) and ensemble agreement
    limiting_land_area: Counter[str] = field(default_factory=Counter)
    limiting_candidate_area: Counter[str] = field(default_factory=Counter)
    agreement_area_sum: float = 0.0
    agreement_candidate_area_sum: float = 0.0
    low_agreement_area_km2: float = 0.0
    has_agreement: bool = False

    def add_diagnostics(
        self,
        dg: ExtrapolationDiagnostics,
        area: np.ndarray,
        cand: np.ndarray,
        predictors: Sequence[str],
    ) -> None:
        for name, flag in dg.flags.items():
            self.diagnostic_land_area[name] += float(area[flag].sum())
            self.diagnostic_candidate_area[name] += float(area[cand & flag].sum())
        consensus = dg.consensus
        for n in range(len(EXTRAPOLATION_DIAGNOSTICS) + 1):
            self.consensus_land_area[n] += float(area[consensus == n].sum())
            self.candidate_consensus_area[n] += float(area[cand & (consensus == n)].sum())
        self.exdet_univariate_area_km2 += float(area[dg.univariate].sum())
        comb = dg.combinatorial
        self.exdet_combinatorial_area_km2 += float(area[comb].sum())
        if comb.any():
            for idx, a in pd.Series(area[comb]).groupby(dg.exdet_mic[comb]).sum().items():
                self.combinatorial_variables[predictors[int(idx)]] += float(a)

    def extrapolation_dict(self, top_n: int = 5) -> dict[str, Any] | None:
        """Per-diagnostic verdicts with their thresholds — the documented basis of every
        extrapolation flag shown in the UI and the bulletin."""
        cfg = self.extrapolation_config
        if cfg is None:
            return None
        land, cand = self.land_area_km2, self.candidate_area_km2
        thresholds = cfg["thresholds"]
        majority = len(EXTRAPOLATION_DIAGNOSTICS) // 2 + 1
        return {
            **{k: v for k, v in cfg.items() if k != "thresholds"},
            "diagnostics": [
                {
                    "id": name,
                    "label": thresholds[name]["label"],
                    "name": thresholds[name]["name"],
                    "reference": thresholds[name]["reference"],
                    "flag_rule": thresholds[name]["flag_rule"],
                    "threshold": thresholds[name]["value"],
                    "threshold_rule": thresholds[name]["rule"],
                    "land_area_km2": round(self.diagnostic_land_area[name], 1),
                    "land_fraction": (
                        round(self.diagnostic_land_area[name] / land, 4) if land else 0.0
                    ),
                    "candidate_flagged_fraction": (
                        round(self.diagnostic_candidate_area[name] / cand, 4) if cand else 0.0
                    ),
                }
                for name in EXTRAPOLATION_DIAGNOSTICS
            ],
            "consensus_land_area_km2": [round(a, 1) for a in self.consensus_land_area],
            "candidate_consensus_ok_fraction": (
                round(self.candidate_consensus_area[0] / cand, 4) if cand else 1.0
            ),
            "candidate_consensus_majority_fraction": (
                round(sum(self.candidate_consensus_area[majority:]) / cand, 4) if cand else 0.0
            ),
            "exdet_univariate_area_km2": round(self.exdet_univariate_area_km2, 1),
            "exdet_combinatorial_area_km2": round(self.exdet_combinatorial_area_km2, 1),
            "top_combinatorial_variables": [
                {"variable": k, "area_km2": round(v, 1)}
                for k, v in self.combinatorial_variables.most_common(top_n)
            ],
        }

    def to_dict(self, top_n: int = 5) -> dict[str, Any]:
        cand = self.candidate_area_km2
        land = self.land_area_km2
        extrapolation = self.extrapolation_dict(top_n)
        limiting = (
            {
                "rule": "predictor whose change to its mean over the training presences "
                "raises suitability most (Elith et al. 2010)",
                "land": (
                    [
                        {"variable": k, "area_km2": round(v, 1), "fraction": round(v / land, 4)}
                        for k, v in self.limiting_land_area.most_common()
                    ]
                    if land
                    else []
                ),
                "candidate": (
                    [
                        {"variable": k, "area_km2": round(v, 1), "fraction": round(v / cand, 4)}
                        for k, v in self.limiting_candidate_area.most_common()
                    ]
                    if cand
                    else []
                ),
            }
            if self.limiting_land_area
            else None
        )
        agreement = (
            {
                "rule": "weighted share of ensemble members whose own binary call agrees with "
                "the ensemble",
                "land_mean": round(self.agreement_area_sum / land, 4) if land else None,
                "candidate_mean": (
                    round(self.agreement_candidate_area_sum / cand, 4) if cand else None
                ),
                "low_agreement_land_fraction": (
                    round(self.low_agreement_area_km2 / land, 4) if land else None
                ),
                "low_agreement_below": 0.6,
            }
            if self.has_agreement
            else None
        )
        return {
            **({"extrapolation": extrapolation} if extrapolation else {}),
            **({"limiting_factor": limiting} if limiting else {}),
            **({"ensemble_agreement": agreement} if agreement else {}),
            "land_area_km2": round(self.land_area_km2, 1),
            "nonnative_land_area_km2": round(self.nonnative_land_area_km2, 1),
            "native_suitable_area_km2": round(self.native_suitable_area_km2, 1),
            "established_outside_area_km2": round(self.established_outside_area_km2, 1),
            "candidate_area_km2": round(cand, 1),
            "candidate_mean_suitability": (
                round(self.candidate_suitability_area_sum / cand, 4) if cand else 0.0
            ),
            "candidate_fraction_of_nonnative_land": (
                round(cand / self.nonnative_land_area_km2, 5)
                if self.nonnative_land_area_km2
                else 0.0
            ),
            "candidate_mess_ok_fraction": (
                round(self.candidate_mess_ok_area_km2 / cand, 4) if cand else 1.0
            ),
            "candidate_envelope_fraction": (
                round(self.candidate_envelope_area_km2 / cand, 4) if cand else 0.0
            ),
            "extrapolation_area_km2": round(self.extrapolation_area_km2, 1),
            "top_candidate_regions": [
                {"tile": k, "area_km2": round(v, 1)}
                for k, v in self.candidate_by_tile.most_common(top_n)
            ],
            "top_limiting_variables": [
                {"variable": k, "area_km2": round(v, 1)}
                for k, v in self.limiting_variables.most_common(top_n)
            ],
        }


@dataclass
class ProjectionOutputs:
    suitability: Path
    mess: Path
    zones: Path
    extrapolation: Path
    scenarios: dict[str, ScenarioOutputs]
    summary: dict[str, Any]
    # exdet / mop / shape / aoa / consensus (empty for models without an extrapolation reference)
    diagnostics: dict[str, Path] = field(default_factory=dict)
    # limiting / agreement (when produced)
    products: dict[str, Path] = field(default_factory=dict)


def _tile_label(lat_top: float, lon_left: float) -> str:
    """Human-readable 10°×10° tile label, e.g. '30–40°N, 110–120°E'."""

    def fmt(a: float, b: float, pos: str, neg: str) -> str:
        lo, hi = sorted((a, b))
        if lo >= 0:
            return f"{lo:.0f}–{hi:.0f}°{pos}"
        if hi <= 0:
            return f"{abs(hi):.0f}–{abs(lo):.0f}°{neg}"
        return f"{abs(lo):.0f}°{neg}–{hi:.0f}°{pos}"

    return f"{fmt(lat_top - 10, lat_top, 'N', 'S')}, {fmt(lon_left, lon_left + 10, 'E', 'W')}"


def _predict(result: TrainingResult, x: np.ndarray) -> np.ndarray:
    out = np.empty(len(x), dtype="float32")
    for s in range(0, len(x), _PREDICT_CHUNK):
        chunk = pd.DataFrame(x[s : s + _PREDICT_CHUNK], columns=result.predictors)
        out[s : s + _PREDICT_CHUNK] = np.asarray(result.model.predict(chunk)).ravel()
    return out


class _Writer:
    """Tiled GTiff writer converted to a COG on close (COG driver cannot write by window)."""

    def __init__(
        self,
        final_path: Path,
        stack: BioclimStack,
        dtype: str,
        nodata: float,
        overview_resampling: str = "average",
    ):
        self.final_path = final_path
        self.overview_resampling = overview_resampling
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name) / final_path.name
        self.ds = rasterio.open(
            self.tmp_path,
            "w",
            driver="GTiff",
            width=stack.width,
            height=stack.height,
            count=1,
            dtype=dtype,
            crs=stack.crs,
            transform=stack.transform,
            nodata=nodata,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
        )

    def write(self, block: RasterBlock, arr: np.ndarray) -> None:
        win = Window(0, block.row_off, block.width, block.height)
        self.ds.write(arr, 1, window=win)

    def close(self) -> None:
        self.ds.close()
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        opts = {**_COG_OPTS, "overview_resampling": self.overview_resampling}
        rio_copy(self.tmp_path, self.final_path, driver="COG", **opts)
        self._tmpdir.cleanup()


class _DiagnosticWriters:
    """Writers for the advanced extrapolation rasters of one projection (keys of
    DIAGNOSTIC_RASTERS and/or "consensus")."""

    def __init__(self, stack: BioclimStack, paths: dict[str, Path]):
        self.paths = paths
        self.writers: dict[str, _Writer] = {}
        try:
            for key, path in paths.items():
                self.writers[key] = (
                    _Writer(path, stack, "uint8", ZONE_NODATA, overview_resampling="mode")
                    if key == "consensus"
                    else _Writer(path, stack, "float32", FLOAT_NODATA)
                )
        except Exception:
            self.close()
            raise

    def write(
        self, block: RasterBlock, valid: np.ndarray, dg: ExtrapolationDiagnostics | None
    ) -> None:
        n = block.height * block.width
        values: dict[str, np.ndarray] = {}
        if dg is not None:
            values = {"exdet": dg.exdet, "mop": dg.mop, "shape": dg.shape, "aoa": dg.aoa_di}
            values["consensus"] = dg.consensus
        for key, wr in self.writers.items():
            nodata, dtype = (
                (ZONE_NODATA, "uint8") if key == "consensus" else (FLOAT_NODATA, "float32")
            )
            arr = np.full(n, nodata, dtype=dtype)
            if key in values:
                arr[valid] = values[key]
            wr.write(block, arr.reshape(block.height, block.width))

    def close(self) -> None:
        for wr in self.writers.values():
            wr.close()


def project_model(
    result: TrainingResult,
    stack: BioclimStack,
    out_dir: Path,
    native_geom: shapely.Geometry,
    occ_lon: Sequence[float],
    occ_lat: Sequence[float],
    occupied_buffer_km: float = 50.0,
    scenario_stacks: dict[str, BioclimStack] | None = None,
    block_rows: int | None = None,
    limiting_factor: bool = True,
) -> ProjectionOutputs:
    out_dir.mkdir(parents=True, exist_ok=True)
    occupied = buffer_region(occ_lon, occ_lat, occupied_buffer_km) if len(occ_lon) else None
    lo_env = np.array([result.presence_envelope[p][0] for p in result.predictors])
    hi_env = np.array([result.presence_envelope[p][1] for p in result.predictors])

    paths = {
        "suitability": out_dir / "suitability.tif",
        "mess": out_dir / "mess.tif",
        "zones": out_dir / "zones.tif",
        "extrapolation": out_dir / "extrapolation.tif",
    }
    w_suit = _Writer(paths["suitability"], stack, "float32", FLOAT_NODATA)
    w_mess = _Writer(paths["mess"], stack, "float32", FLOAT_NODATA)
    w_zone = _Writer(paths["zones"], stack, "uint8", ZONE_NODATA)
    w_extra = _Writer(paths["extrapolation"], stack, "uint8", ZONE_NODATA)
    summary = ProjectionSummary()
    ext_ref: ExtrapolationReference | None = getattr(result, "extrapolation_reference", None)
    w_diag = _DiagnosticWriters(
        stack,
        (
            {
                **{k: out_dir / f for k, f in DIAGNOSTIC_RASTERS.items()},
                "consensus": out_dir / CONSENSUS_RASTER,
            }
            if ext_ref is not None
            else {}
        ),
    )
    if ext_ref is not None:
        summary.extrapolation_config = ext_ref.describe()
    presence_mean = getattr(result, "presence_mean", None) or {}
    do_limiting = limiting_factor and all(p in presence_mean for p in result.predictors)
    do_agreement = hasattr(result.model, "member_agreement")
    products: dict[str, Path] = {}
    w_lim = w_agree = None
    if do_limiting:
        products["limiting"] = out_dir / LIMITING_RASTER
        w_lim = _Writer(products["limiting"], stack, "uint8", ZONE_NODATA, "mode")
    if do_agreement:
        products["agreement"] = out_dir / AGREEMENT_RASTER
        w_agree = _Writer(products["agreement"], stack, "float32", FLOAT_NODATA)
        summary.has_agreement = True
    try:
        for block in stack.iter_blocks(result.predictors, block_rows=block_rows):
            h, w = block.height, block.width
            x = block.data.reshape(len(result.predictors), -1).T
            valid = ~np.isnan(x).any(axis=1)

            suit = np.full(h * w, FLOAT_NODATA, dtype="float32")
            mess_v = np.full(h * w, FLOAT_NODATA, dtype="float32")
            zones = np.full(h * w, ZONE_NODATA, dtype="uint8")
            extra = np.full(h * w, ZONE_NODATA, dtype="uint8")
            lim_codes = np.full(h * w, ZONE_NODATA, dtype="uint8")
            agree = np.full(h * w, FLOAT_NODATA, dtype="float32")
            dg: ExtrapolationDiagnostics | None = None
            if valid.any():
                xv = x[valid]
                suit[valid] = _predict(result, xv)
                if do_limiting:
                    lim_codes[valid] = _limiting_codes(result, xv, suit[valid], presence_mean)
                if do_agreement:
                    agree[valid] = _agreement(result, xv)
                mv, mod = mess(result.mess_reference, xv)
                mess_v[valid] = mv
                if ext_ref is not None:
                    dg = extrapolation_diagnostics(ext_ref, xv, mv)

                shape_ = (h, w)
                native = geometry_mask(
                    [native_geom], out_shape=shape_, transform=block.transform, invert=True
                ).ravel()[valid]
                occ_mask = (
                    geometry_mask(
                        [occupied], out_shape=shape_, transform=block.transform, invert=True
                    ).ravel()[valid]
                    if occupied is not None and not occupied.is_empty
                    else np.zeros(valid.sum(), dtype=bool)
                )
                suitable = suit[valid] >= result.threshold
                z = np.zeros(valid.sum(), dtype="uint8")
                z[suitable & native] = 1
                z[suitable & ~native & occ_mask] = 2
                cand = suitable & ~native & ~occ_mask
                z[cand] = 3
                zones[valid] = z
                extra[valid] = (mv < 0).astype("uint8")

                rows = np.repeat(np.arange(block.row_off, block.row_off + h), w)[valid]
                cols = np.tile(np.arange(w), h)[valid]
                area = cell_area_km2(stack.transform, rows)
                in_env = ((xv >= lo_env) & (xv <= hi_env)).all(axis=1)
                summary.land_area_km2 += float(area.sum())
                summary.nonnative_land_area_km2 += float(area[~native].sum())
                summary.native_suitable_area_km2 += float(area[suitable & native].sum())
                summary.established_outside_area_km2 += float(
                    area[suitable & ~native & occ_mask].sum()
                )
                summary.candidate_area_km2 += float(area[cand].sum())
                summary.candidate_suitability_area_sum += float(
                    (area[cand] * suit[valid][cand]).sum()
                )
                summary.candidate_mess_ok_area_km2 += float(area[cand & (mv >= 0)].sum())
                summary.candidate_envelope_area_km2 += float(area[cand & in_env].sum())
                summary.extrapolation_area_km2 += float(area[mv < 0].sum())
                if dg is not None:
                    summary.add_diagnostics(dg, area, cand, result.predictors)
                if do_limiting:
                    codes = lim_codes[valid]
                    for code in np.unique(codes):
                        name = "none" if code == LIMITING_NONE else result.predictors[int(code)]
                        sel = codes == code
                        summary.limiting_land_area[name] += float(area[sel].sum())
                        summary.limiting_candidate_area[name] += float(area[sel & cand].sum())
                if do_agreement:
                    av = agree[valid].astype("float64")
                    summary.agreement_area_sum += float((area * av).sum())
                    summary.agreement_candidate_area_sum += float((area[cand] * av[cand]).sum())
                    summary.low_agreement_area_km2 += float(area[av < 0.6].sum())
                if cand.any():
                    lon = stack.transform.c + (cols[cand] + 0.5) * stack.transform.a
                    lat = stack.transform.f + (rows[cand] + 0.5) * stack.transform.e
                    tiles = (
                        pd.Series(area[cand])
                        .groupby([np.floor(lat / 10) * 10 + 10, np.floor(lon / 10) * 10])
                        .sum()
                    )
                    for (lat_top, lon_left), a in tiles.items():
                        summary.candidate_by_tile[_tile_label(lat_top, lon_left)] += float(a)
                    extrap = cand & (mv < 0)
                    if extrap.any():
                        lim = pd.Series(area[extrap]).groupby(mod[extrap]).sum()
                        for idx, a in lim.items():
                            summary.limiting_variables[result.predictors[int(idx)]] += float(a)

            w_suit.write(block, suit.reshape(h, w))
            w_mess.write(block, mess_v.reshape(h, w))
            w_zone.write(block, zones.reshape(h, w))
            w_extra.write(block, extra.reshape(h, w))
            w_diag.write(block, valid, dg)
            if w_lim is not None:
                w_lim.write(block, lim_codes.reshape(h, w))
            if w_agree is not None:
                w_agree.write(block, agree.reshape(h, w))
    finally:
        for wr in (w_suit, w_mess, w_zone, w_extra, w_diag, w_lim, w_agree):
            if wr is not None:
                wr.close()

    scenario_out = {
        name: project_scenario(result, sc_stack, out_dir, name, block_rows)
        for name, sc_stack in (scenario_stacks or {}).items()
    }
    return ProjectionOutputs(
        suitability=paths["suitability"],
        mess=paths["mess"],
        zones=paths["zones"],
        extrapolation=paths["extrapolation"],
        scenarios=scenario_out,
        summary=summary.to_dict(),
        diagnostics=w_diag.paths,
        products=products,
    )


def _limiting_codes(
    result: TrainingResult, xv: np.ndarray, base: np.ndarray, presence_mean: dict[str, float]
) -> np.ndarray:
    """Elith et al. (2010): per cell, the predictor whose change to its mean over the training
    presences would raise suitability most (LIMITING_NONE if none raises it)."""
    gains = np.empty((len(xv), len(result.predictors)), dtype="float32")
    for j, p in enumerate(result.predictors):
        xj = xv.copy()
        xj[:, j] = presence_mean[p]
        gains[:, j] = _predict(result, xj) - base
    codes = np.argmax(gains, axis=1).astype("uint8")
    codes[gains.max(axis=1) <= 1e-6] = LIMITING_NONE
    return codes


def _agreement(result: TrainingResult, xv: np.ndarray) -> np.ndarray:
    out = np.empty(len(xv), dtype="float32")
    for s in range(0, len(xv), _PREDICT_CHUNK):
        chunk = pd.DataFrame(xv[s : s + _PREDICT_CHUNK], columns=result.predictors)
        out[s : s + _PREDICT_CHUNK] = result.model.member_agreement(chunk, result.threshold)
    return out


# ---------------------------------------------------------------------------
# Products derived from projected rasters (binary, change, survey priority)
# ---------------------------------------------------------------------------
def _write_like(src: Any, path: Path, dtype: str, nodata: float, resampling: str) -> Any:
    return rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=src.width,
        height=src.height,
        count=1,
        dtype=dtype,
        crs=src.crs,
        transform=src.transform,
        nodata=nodata,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
    )


def _to_cog(tmp_path: Path, path: Path, resampling: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rio_copy(tmp_path, path, driver="COG", **{**_COG_OPTS, "overview_resampling": resampling})


def same_grid(a: Path, b: Path) -> bool:
    with rasterio.open(a) as ra, rasterio.open(b) as rb:
        return (ra.width, ra.height) == (rb.width, rb.height) and ra.transform.almost_equals(
            rb.transform
        )


def change_map(
    before: Path, before_threshold: float, after: Path, after_threshold: float, path: Path
) -> dict[str, Any]:
    """Binary change between two suitability surfaces on the same grid (U6): codes 0 stable
    unsuitable, 1 gain, 2 loss, 3 stable suitable; area (km²) of each."""
    areas = dict.fromkeys(CHANGE_LABELS, 0.0)
    with (
        rasterio.open(before) as rb,
        rasterio.open(after) as ra,
        tempfile.TemporaryDirectory() as tmp,
    ):
        tmp_path = Path(tmp) / path.name
        with _write_like(ra, tmp_path, "uint8", ZONE_NODATA, "mode") as dst:
            for _, win in ra.block_windows(1):
                a = ra.read(1, window=win)
                b = rb.read(1, window=win)
                valid = (a != ra.nodata) & (b != rb.nodata)
                code = (a >= after_threshold).astype("uint8") * 1 + (b >= before_threshold).astype(
                    "uint8"
                ) * 2
                # after + 2 × before: 0 stable unsuitable, 1 gain, 2 loss, 3 stable suitable
                out = np.full(a.shape, ZONE_NODATA, dtype="uint8")
                out[valid] = code[valid]
                rows = np.arange(win.row_off, win.row_off + win.height)
                area = np.broadcast_to(cell_area_km2(ra.transform, rows)[:, None], a.shape)
                for c in CHANGE_LABELS:
                    areas[c] += float(area[out == c].sum())
                dst.write(out, 1, window=win)
        _to_cog(tmp_path, path, "mode")
    before_area = areas[2] + areas[3]
    return {
        "codes": {str(k): v for k, v in CHANGE_LABELS.items()},
        "gain_km2": round(areas[1], 1),
        "loss_km2": round(areas[2], 1),
        "stable_suitable_km2": round(areas[3], 1),
        "stable_unsuitable_km2": round(areas[0], 1),
        "net_change_fraction": (
            round((areas[1] - areas[2]) / before_area, 4) if before_area else None
        ),
        "before_threshold": before_threshold,
        "after_threshold": after_threshold,
    }


PRIORITY_FORMULA = (
    "priority = suitability × (1 − effort) × (0.5 + 0.5 × uncertainty); effort = log10(target-"
    "group records + 1) / log10(max + 1); uncertainty from the {source}"
)


def priority_map(
    suitability: Path,
    threshold: float,
    effort: np.ndarray,
    effort_pixel_deg: float,
    path: Path,
    uncertainty: Path | None = None,
    uncertainty_kind: str | None = None,
    top_n: int = 25,
) -> dict[str, Any]:
    """Survey prioritisation (U12, "where to look next"): suitable, little-surveyed, uncertain
    cells score highest. `uncertainty_kind`: "agreement" (1 − ensemble agreement),
    "consensus" (diagnostics flagging extrapolation / 5) or "mess" (MESS < 0 mask)."""
    ny, nx = effort.shape
    log_max = float(np.log10(effort.max() + 1.0)) if effort.size and effort.max() > 0 else 1.0
    top: list[tuple[float, float, float, float, float, float]] = []
    total_area = 0.0
    with rasterio.open(suitability) as src, tempfile.TemporaryDirectory() as tmp:
        unc_src = rasterio.open(uncertainty) if uncertainty is not None else None
        tmp_path = Path(tmp) / path.name
        try:
            with _write_like(src, tmp_path, "float32", FLOAT_NODATA, "average") as dst:
                t = src.transform
                for _, win in src.block_windows(1):
                    s = src.read(1, window=win).astype("float64")
                    valid = s != src.nodata
                    rows = np.arange(win.row_off, win.row_off + win.height)
                    cols = np.arange(win.col_off, win.col_off + win.width)
                    lat = t.f + (rows + 0.5) * t.e
                    lon = t.c + (cols + 0.5) * t.a
                    gr = np.clip(np.floor((90.0 - lat) / effort_pixel_deg).astype(int), 0, ny - 1)
                    gc = np.clip(np.floor((lon + 180.0) / effort_pixel_deg).astype(int), 0, nx - 1)
                    e = np.log10(effort[np.ix_(gr, gc)] + 1.0) / log_max
                    u = np.zeros_like(s)
                    if unc_src is not None:
                        raw = unc_src.read(1, window=win).astype("float64")
                        ok = raw != unc_src.nodata
                        if uncertainty_kind == "agreement":
                            u = np.where(ok, 1.0 - raw, 0.0)
                        elif uncertainty_kind == "consensus":
                            u = np.where(ok, raw / 5.0, 0.0)
                        else:
                            u = np.where(ok, raw, 0.0)
                    score = np.clip(s, 0, 1) * (1 - np.clip(e, 0, 1)) * (0.5 + 0.5 * u)
                    out = np.where(valid, score, FLOAT_NODATA).astype("float32")
                    dst.write(out, 1, window=win)
                    area = cell_area_km2(t, rows)
                    total_area += float(
                        np.broadcast_to(area[:, None], s.shape)[valid & (s >= threshold)].sum()
                    )
                    cand = valid & (s >= threshold)
                    if cand.any():
                        r_i, c_i = np.nonzero(cand)
                        sc = score[cand]
                        best = np.argsort(sc)[-top_n:]
                        for b in best:
                            top.append(
                                (
                                    float(sc[b]),
                                    float(lon[c_i[b]]),
                                    float(lat[r_i[b]]),
                                    float(s[r_i[b], c_i[b]]),
                                    float(e[r_i[b], c_i[b]]),
                                    float(u[r_i[b], c_i[b]]),
                                )
                            )
                        top = sorted(top, reverse=True)[:top_n]
        finally:
            if unc_src is not None:
                unc_src.close()
        _to_cog(tmp_path, path, "average")
    return {
        "formula": PRIORITY_FORMULA.format(
            source={
                "agreement": "ensemble disagreement (1 − agreement)",
                "consensus": "number of extrapolation diagnostics flagging the cell / 5",
                "mess": "MESS < 0 mask",
            }.get(uncertainty_kind or "", "— none available (term = 0)")
        ),
        "uncertainty_source": uncertainty_kind,
        "suitable_area_km2": round(total_area, 1),
        "top_cells": [
            {
                "rank": i + 1,
                "longitude": round(lo, 4),
                "latitude": round(la, 4),
                "priority": round(sc, 4),
                "suitability": round(su, 4),
                "effort": round(ef, 4),
                "uncertainty": round(un, 4),
            }
            for i, (sc, lo, la, su, ef, un) in enumerate(top)
        ],
    }


class Projectable(Protocol):
    """What scenario projection needs from a model: a fitted TrainingResult or a stored model
    (`extrapolation_reference` may be None for models stored before it existed)."""

    model: Any
    predictors: list[str]
    mess_reference: MessReference
    extrapolation_reference: ExtrapolationReference | None


@dataclass
class ScenarioOutputs:
    suitability: Path
    extrapolation: Path
    extrapolated_land_fraction: float
    # Consensus of the five diagnostics under this climate, and each one's extrapolated share
    # of land (None / empty for models without an extrapolation reference).
    consensus: Path | None = None
    diagnostics_land_fraction: dict[str, float] = field(default_factory=dict)


def project_scenario(
    result: Projectable,
    stack: BioclimStack,
    out_dir: Path,
    name: str,
    block_rows: int | None = None,
) -> ScenarioOutputs:
    """Suitability under an alternative (e.g. CMIP6 future) climate — no retraining — always
    paired with that scenario's own MESS extrapolation mask (future climates extrapolate more
    than the present, so the current-climate mask must not be reused), plus its own consensus
    of the five extrapolation diagnostics when the model has an extrapolation reference."""
    suit_path = out_dir / f"suitability_{name}.tif"
    extra_path = out_dir / f"extrapolation_{name}.tif"
    w_suit = _Writer(suit_path, stack, "float32", FLOAT_NODATA)
    w_extra = _Writer(extra_path, stack, "uint8", ZONE_NODATA)
    ext_ref = getattr(result, "extrapolation_reference", None)
    w_diag = _DiagnosticWriters(
        stack, {"consensus": out_dir / f"consensus_{name}.tif"} if ext_ref is not None else {}
    )
    land = extrapolated = 0.0
    flagged: Counter[str] = Counter()
    try:
        for block in stack.iter_blocks(result.predictors, block_rows=block_rows):
            h, w = block.height, block.width
            x = block.data.reshape(len(result.predictors), -1).T
            valid = ~np.isnan(x).any(axis=1)
            suit = np.full(len(x), FLOAT_NODATA, dtype="float32")
            extra = np.full(len(x), ZONE_NODATA, dtype="uint8")
            dg: ExtrapolationDiagnostics | None = None
            if valid.any():
                suit[valid] = _predict(result, x[valid])
                mv, _ = mess(result.mess_reference, x[valid])
                extra[valid] = (mv < 0).astype("uint8")
                rows = np.repeat(np.arange(block.row_off, block.row_off + h), w)[valid]
                area = cell_area_km2(stack.transform, rows)
                land += float(area.sum())
                extrapolated += float(area[mv < 0].sum())
                if ext_ref is not None:
                    dg = extrapolation_diagnostics(ext_ref, x[valid], mv)
                    for key, flag in dg.flags.items():
                        flagged[key] += float(area[flag].sum())
            w_suit.write(block, suit.reshape(h, w))
            w_extra.write(block, extra.reshape(h, w))
            w_diag.write(block, valid, dg)
    finally:
        w_suit.close()
        w_extra.close()
        w_diag.close()
    return ScenarioOutputs(
        suit_path,
        extra_path,
        round(extrapolated / land, 4) if land else 0.0,
        consensus=w_diag.paths.get("consensus"),
        diagnostics_land_fraction=(
            {k: round(flagged[k] / land, 4) if land else 0.0 for k in EXTRAPOLATION_DIAGNOSTICS}
            if ext_ref is not None
            else {}
        ),
    )


# ---------------------------------------------------------------------------
# Multi-GCM scenario ensembles (U7)
# ---------------------------------------------------------------------------
GCM_ENSEMBLE_PREFIX = "ensemble_"


def scenario_groups(names: Iterable[str]) -> dict[str, list[str]]:
    """CMIP6 scenarios named `<gcm>_<ssp>_<period>` (bioclim_store add-scenario) grouped by
    `<ssp>_<period>`; only groups with at least two GCMs form an ensemble."""
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        parts = name.rsplit("_", 2)
        if name.startswith(GCM_ENSEMBLE_PREFIX) or len(parts) != 3 or not all(parts):
            continue
        groups[f"{parts[1]}_{parts[2]}"].append(name)
    return {k: sorted(v) for k, v in sorted(groups.items()) if len(v) >= 2}


@dataclass
class GcmEnsembleOutputs:
    suitability: Path
    extrapolation: Path
    agreement: Path
    summary: dict[str, Any]


def gcm_ensemble(
    members: dict[str, tuple[Path, Path]], threshold: float, out_dir: Path, key: str
) -> GcmEnsembleOutputs:
    """Combine one SSP × period across GCMs: mean suitability, the union of the members'
    MESS < 0 masks and the number of GCMs calling each cell suitable at `threshold`.
    `members` maps scenario name → (suitability, extrapolation) on one grid."""
    names = sorted(members)
    if len(names) < 2:
        raise ValueError("a GCM ensemble needs at least two scenario projections")
    first = members[names[0]][0]
    for name in names[1:]:
        if not same_grid(first, members[name][0]):
            raise ValueError(f"scenario {name} is not on the grid of {names[0]}")
    n = len(names)
    paths = {
        "suitability": out_dir / f"suitability_{GCM_ENSEMBLE_PREFIX}{key}.tif",
        "extrapolation": out_dir / f"extrapolation_{GCM_ENSEMBLE_PREFIX}{key}.tif",
        "agreement": out_dir / f"agreement_{GCM_ENSEMBLE_PREFIX}{key}.tif",
    }
    area_by_votes = np.zeros(n + 1)
    land = extrapolated_any = extrapolated_all = 0.0
    srcs = [(rasterio.open(members[m][0]), rasterio.open(members[m][1])) for m in names]
    try:
        ref = srcs[0][0]
        with tempfile.TemporaryDirectory() as tmp:
            tmps = {k: Path(tmp) / p.name for k, p in paths.items()}
            with (
                _write_like(ref, tmps["suitability"], "float32", FLOAT_NODATA, "average") as w_s,
                _write_like(ref, tmps["extrapolation"], "uint8", ZONE_NODATA, "mode") as w_e,
                _write_like(ref, tmps["agreement"], "uint8", ZONE_NODATA, "mode") as w_a,
            ):
                for _, win in ref.block_windows(1):
                    suit = np.stack([s.read(1, window=win) for s, _ in srcs]).astype("float64")
                    extra = np.stack([e.read(1, window=win) for _, e in srcs])
                    valid = (suit != FLOAT_NODATA).all(axis=0) & (extra != ZONE_NODATA).all(axis=0)
                    mean = np.full(valid.shape, FLOAT_NODATA, dtype="float32")
                    mean[valid] = suit.mean(axis=0)[valid]
                    votes = (suit >= threshold).sum(axis=0)
                    agree = np.full(valid.shape, ZONE_NODATA, dtype="uint8")
                    agree[valid] = votes[valid]
                    n_extra = (extra == 1).sum(axis=0)
                    ext = np.full(valid.shape, ZONE_NODATA, dtype="uint8")
                    ext[valid] = (n_extra[valid] > 0).astype("uint8")
                    rows = np.arange(win.row_off, win.row_off + win.height)
                    area = np.broadcast_to(cell_area_km2(ref.transform, rows)[:, None], valid.shape)
                    land += float(area[valid].sum())
                    extrapolated_any += float(area[valid & (n_extra > 0)].sum())
                    extrapolated_all += float(area[valid & (n_extra == n)].sum())
                    for v in range(n + 1):
                        area_by_votes[v] += float(area[valid & (votes == v)].sum())
                    w_s.write(mean, 1, window=win)
                    w_e.write(ext, 1, window=win)
                    w_a.write(agree, 1, window=win)
            _to_cog(tmps["suitability"], paths["suitability"], "average")
            _to_cog(tmps["extrapolation"], paths["extrapolation"], "mode")
            _to_cog(tmps["agreement"], paths["agreement"], "mode")
    finally:
        for s, e in srcs:
            s.close()
            e.close()

    def frac(a: float) -> float:
        return round(a / land, 4) if land else 0.0

    suitable_any = float(area_by_votes[1:].sum())
    suitable_all = float(area_by_votes[n])
    majority = float(area_by_votes[n // 2 + 1 :].sum())
    ssp, period = key.split("_", 1)
    summary = {
        "ssp": ssp,
        "period": period,
        "gcms": names,
        "n_gcms": n,
        "threshold": threshold,
        "combination": "unweighted mean of the GCMs' suitability (model democracy); "
        "extrapolation = MESS < 0 under any GCM",
        "suitable_all_gcms_land_fraction": frac(suitable_all),
        "suitable_majority_land_fraction": frac(majority),
        "suitable_any_gcm_land_fraction": frac(suitable_any),
        "gcm_disagreement_land_fraction": frac(suitable_any - suitable_all),
        "extrapolated_any_gcm_land_fraction": frac(extrapolated_any),
        "extrapolated_all_gcms_land_fraction": frac(extrapolated_all),
        "land_fraction_by_votes": {str(v): frac(float(a)) for v, a in enumerate(area_by_votes)},
    }
    return GcmEnsembleOutputs(
        paths["suitability"], paths["extrapolation"], paths["agreement"], summary
    )


# ---------------------------------------------------------------------------
# Sampling effort of the target group (map layer + per-zone summary)
# ---------------------------------------------------------------------------
EFFORT_RASTER = "effort.tif"


def effort_grid(
    lon: Sequence[float], lat: Sequence[float], count: Sequence[float], pixel_deg: float
) -> np.ndarray:
    """Global record-count grid at the effort data's own pixel size (rows from 90°N)."""
    ny, nx = round(180.0 / pixel_deg), round(360.0 / pixel_deg)
    grid = np.zeros((ny, nx), dtype="float64")
    lon_a, lat_a = np.asarray(lon, dtype="float64"), np.asarray(lat, dtype="float64")
    cols = np.clip(np.floor((lon_a + 180.0) / pixel_deg).astype(int), 0, nx - 1)
    rows = np.clip(np.floor((90.0 - lat_a) / pixel_deg).astype(int), 0, ny - 1)
    np.add.at(grid, (rows, cols), np.asarray(count, dtype="float64"))
    return grid


def write_effort_raster(grid: np.ndarray, pixel_deg: float, path: Path) -> Path:
    """COG of log10(records + 1); cells without records are nodata (transparent on the map)."""
    values = np.where(grid > 0, np.log10(grid + 1.0), FLOAT_NODATA).astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / path.name
        with rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            width=grid.shape[1],
            height=grid.shape[0],
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=rasterio.transform.from_origin(-180.0, 90.0, pixel_deg, pixel_deg),
            nodata=FLOAT_NODATA,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
        ) as dst:
            dst.write(values, 1)
        rio_copy(tmp_path, path, driver="COG", **_COG_OPTS)
    return path


def effort_zone_summary(
    zones_path: Path, stack: BioclimStack, grid: np.ndarray, pixel_deg: float
) -> dict[str, Any]:
    """Share of land and of each range zone (area-weighted) where the target group has no
    GBIF records at all: there, missing presences say nothing about suitability."""
    ny, nx = grid.shape
    totals: Counter[str] = Counter()
    unsampled: Counter[str] = Counter()
    names = {1: "native_suitable", 2: "established_outside", 3: "candidate"}
    with rasterio.open(zones_path) as src:
        t = src.transform
        for _, win in src.block_windows(1):
            zones = src.read(1, window=win)
            rows = np.arange(win.row_off, win.row_off + win.height)
            cols = np.arange(win.col_off, win.col_off + win.width)
            lat = t.f + (rows + 0.5) * t.e
            lon = t.c + (cols + 0.5) * t.a
            gr = np.clip(np.floor((90.0 - lat) / pixel_deg).astype(int), 0, ny - 1)
            gc = np.clip(np.floor((lon + 180.0) / pixel_deg).astype(int), 0, nx - 1)
            empty = grid[np.ix_(gr, gc)] <= 0
            area = np.broadcast_to(cell_area_km2(stack.transform, rows)[:, None], zones.shape)
            land = zones != ZONE_NODATA
            totals["land"] += float(area[land].sum())
            unsampled["land"] += float(area[land & empty].sum())
            for code, name in names.items():
                z = zones == code
                totals[name] += float(area[z].sum())
                unsampled[name] += float(area[z & empty].sum())
    return {
        f"{name}_unsampled_fraction": (
            round(unsampled[name] / totals[name], 4) if totals[name] else None
        )
        for name in ("land", *names.values())
    }
