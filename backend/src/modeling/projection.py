"""Global projection of a trained model: suitability, MESS, and range-zone rasters (COGs).

Zone codes (zones.tif, uint8, nodata=255):
    0 = not suitable
    1 = suitable, inside the confirmed native range
    2 = suitable, outside native range, already occupied (within buffer of known records)
    3 = candidate invasion / expansion zone (suitable, outside native range, not yet occupied)

extrapolation.tif (uint8, nodata=255) is a 0/1 mask of MESS < 0, served as a toggleable map
overlay. MESS is always produced alongside suitability (CLAUDE.md constraint 2).
"""

from __future__ import annotations

import tempfile
from collections import Counter
from collections.abc import Sequence
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

from src.domain import ZONE_LABELS  # noqa: F401  (re-exported for convenience)
from src.features.background_sampler import buffer_region
from src.features.raster_sampler import BioclimStack, RasterBlock, cell_area_km2
from src.modeling.evaluation import MessReference, mess
from src.modeling.maxent_trainer import TrainingResult

FLOAT_NODATA = -9999.0
ZONE_NODATA = 255
_PREDICT_CHUNK = 200_000
_COG_OPTS = {"compress": "deflate", "blocksize": "512", "overview_resampling": "average"}


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

    def to_dict(self, top_n: int = 5) -> dict[str, Any]:
        cand = self.candidate_area_km2
        return {
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

    def __init__(self, final_path: Path, stack: BioclimStack, dtype: str, nodata: float):
        self.final_path = final_path
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
        rio_copy(self.tmp_path, self.final_path, driver="COG", **_COG_OPTS)
        self._tmpdir.cleanup()


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
    try:
        for block in stack.iter_blocks(result.predictors, block_rows=block_rows):
            h, w = block.height, block.width
            x = block.data.reshape(len(result.predictors), -1).T
            valid = ~np.isnan(x).any(axis=1)

            suit = np.full(h * w, FLOAT_NODATA, dtype="float32")
            mess_v = np.full(h * w, FLOAT_NODATA, dtype="float32")
            zones = np.full(h * w, ZONE_NODATA, dtype="uint8")
            extra = np.full(h * w, ZONE_NODATA, dtype="uint8")
            if valid.any():
                xv = x[valid]
                suit[valid] = _predict(result, xv)
                mv, mod = mess(result.mess_reference, xv)
                mess_v[valid] = mv

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
    finally:
        for wr in (w_suit, w_mess, w_zone, w_extra):
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
    )


class Projectable(Protocol):
    """What scenario projection needs from a model: a fitted TrainingResult or a stored model."""

    model: Any
    predictors: list[str]
    mess_reference: MessReference


@dataclass
class ScenarioOutputs:
    suitability: Path
    extrapolation: Path
    extrapolated_land_fraction: float


def project_scenario(
    result: Projectable,
    stack: BioclimStack,
    out_dir: Path,
    name: str,
    block_rows: int | None = None,
) -> ScenarioOutputs:
    """Suitability under an alternative (e.g. CMIP6 future) climate — no retraining — always
    paired with that scenario's own MESS extrapolation mask (future climates extrapolate more
    than the present, so the current-climate mask must not be reused)."""
    suit_path = out_dir / f"suitability_{name}.tif"
    extra_path = out_dir / f"extrapolation_{name}.tif"
    w_suit = _Writer(suit_path, stack, "float32", FLOAT_NODATA)
    w_extra = _Writer(extra_path, stack, "uint8", ZONE_NODATA)
    land = extrapolated = 0.0
    try:
        for block in stack.iter_blocks(result.predictors, block_rows=block_rows):
            h, w = block.height, block.width
            x = block.data.reshape(len(result.predictors), -1).T
            valid = ~np.isnan(x).any(axis=1)
            suit = np.full(len(x), FLOAT_NODATA, dtype="float32")
            extra = np.full(len(x), ZONE_NODATA, dtype="uint8")
            if valid.any():
                suit[valid] = _predict(result, x[valid])
                mv, _ = mess(result.mess_reference, x[valid])
                extra[valid] = (mv < 0).astype("uint8")
                rows = np.repeat(np.arange(block.row_off, block.row_off + h), w)[valid]
                area = cell_area_km2(stack.transform, rows)
                land += float(area.sum())
                extrapolated += float(area[mv < 0].sum())
            w_suit.write(block, suit.reshape(h, w))
            w_extra.write(block, extra.reshape(h, w))
    finally:
        w_suit.close()
        w_extra.close()
    return ScenarioOutputs(suit_path, extra_path, round(extrapolated / land, 4) if land else 0.0)
