"""Single access point for reading rasters (CLAUDE.md: never open bioclim GeoTIFFs ad hoc).

The global bioclim stack is a versioned, 19-band Cloud-Optimized GeoTIFF:

    data/bioclim/<version>/bioclim.tif
    data/bioclim/<version>/metadata.json
    data/bioclim/<version>/scenarios/<name>.tif      (optional CMIP6 future stacks)
    data/bioclim/<version>/hires/bioclim_<res>.tif   (optional finer stack, same source)
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window

from src.domain import BIOCLIM_BANDS

STACK_FILENAME = "bioclim.tif"
METADATA_FILENAME = "metadata.json"
# Cells per projection block: 256 global rows at 10' (2160 columns); fewer rows on finer grids.
BLOCK_CELLS = 552_960
# Cell size (degrees) of the WorldClim resolution codes.
RESOLUTION_DEG = {"10m": 1 / 6, "5m": 1 / 12, "2.5m": 1 / 24, "30s": 1 / 120}


@dataclass(frozen=True)
class BioclimMetadata:
    version: str
    source: str
    resolution: str
    crs: str
    bands: tuple[str, ...]
    created: str
    sha256: str | None = None
    selected_predictors: tuple[str, ...] = ()
    vif: dict[str, float] = field(default_factory=dict)
    scenarios: dict[str, str] = field(default_factory=dict)
    citation: str | None = None
    core_predictors: tuple[str, ...] = ()
    derived_from: str | None = None
    # Finer-resolution stacks of the same source, e.g. {"30s": "hires/bioclim_30s.tif"}, for
    # regional high-resolution projection of models trained on this version.
    hires: dict[str, str] = field(default_factory=dict)
    # Version whose grid and land mask this stack was resampled onto (e.g. a CHELSA
    # cross-check stack aligned to the active WorldClim version).
    aligned_to: str | None = None

    @classmethod
    def load(cls, path: Path) -> BioclimMetadata:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            version=raw["version"],
            source=raw["source"],
            resolution=raw["resolution"],
            crs=raw.get("crs", "EPSG:4326"),
            bands=tuple(raw.get("bands", BIOCLIM_BANDS)),
            created=raw["created"],
            sha256=raw.get("sha256"),
            selected_predictors=tuple(raw.get("selected_predictors", ())),
            vif={k: float(v) for k, v in raw.get("vif", {}).items()},
            scenarios=dict(raw.get("scenarios", {})),
            citation=raw.get("citation"),
            core_predictors=tuple(raw.get("core_predictors", ())),
            derived_from=raw.get("derived_from"),
            hires=dict(raw.get("hires", {})),
            aligned_to=raw.get("aligned_to"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "resolution": self.resolution,
            "crs": self.crs,
            "bands": list(self.bands),
            "created": self.created,
            "sha256": self.sha256,
            "selected_predictors": list(self.selected_predictors),
            "vif": self.vif,
            "scenarios": self.scenarios,
            "citation": self.citation,
            "core_predictors": list(self.core_predictors),
            "derived_from": self.derived_from,
            "hires": self.hires,
            "aligned_to": self.aligned_to,
        }

    def finest_hires(self) -> str | None:
        """Resolution code of the finest registered high-resolution stack, if any."""
        if not self.hires:
            return None
        return min(self.hires, key=lambda r: RESOLUTION_DEG.get(r, float("inf")))


@dataclass(frozen=True)
class RasterBlock:
    row_off: int
    height: int
    width: int
    transform: Affine
    data: np.ndarray  # (bands, rows, cols) float32, NaN where nodata


class BioclimStack:
    """Read-only view of one bioclim stack version (current climate or a named scenario)."""

    def __init__(self, path: Path, metadata: BioclimMetadata, scenario: str | None = None):
        if not path.exists():
            raise FileNotFoundError(f"Bioclim stack not found: {path}")
        self.path = path
        self.metadata = metadata
        self.scenario = scenario
        with rasterio.open(path) as src:
            self.width, self.height = src.width, src.height
            self.transform: Affine = src.transform
            self.crs = src.crs
            self.nodata = src.nodata
            self.res: tuple[float, float] = src.res
        # Grid offset of a regional subset within the file (None = the full grid).
        self._window: Window | None = None

    @classmethod
    def open_version(
        cls,
        bioclim_root: Path,
        version: str,
        scenario: str | None = None,
        hires: str | None = None,
    ) -> BioclimStack:
        root = bioclim_root / version
        meta = BioclimMetadata.load(root / METADATA_FILENAME)
        if hires is not None:
            if hires not in meta.hires:
                raise KeyError(f"No {hires} high-resolution stack registered for {version}")
            return cls(root / meta.hires[hires], meta)
        if scenario is None:
            return cls(root / STACK_FILENAME, meta)
        if scenario not in meta.scenarios:
            raise KeyError(f"Scenario {scenario!r} not registered for {version}")
        return cls(root / meta.scenarios[scenario], meta, scenario=scenario)

    def subset(self, bbox: tuple[float, float, float, float]) -> BioclimStack:
        """View of the cells inside (west, south, east, north): projection over it writes a
        regional raster. Point sampling is unaffected."""
        win = rasterio.windows.from_bounds(*bbox, transform=self.transform)
        win = win.round_offsets(op="floor").round_lengths(op="ceil")
        win = win.intersection(Window(0, 0, self.width, self.height))
        view = BioclimStack.__new__(BioclimStack)
        view.__dict__.update(self.__dict__)
        view._window = win
        view.width, view.height = int(win.width), int(win.height)
        view.transform = rasterio.windows.transform(win, self.transform)
        return view

    @property
    def version(self) -> str:
        return self.metadata.version

    def band_indexes(self, bands: Sequence[str]) -> list[int]:
        order = list(self.metadata.bands)
        return [order.index(b) + 1 for b in bands]

    def _masked(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype("float32", copy=False)
        if self.nodata is not None and not np.isnan(self.nodata):
            arr[arr == self.nodata] = np.nan
        # WorldClim uses a large negative fill value in some releases.
        arr[arr < -3.0e38] = np.nan
        return arr

    # -------------------------------------------------------------- points
    def sample(
        self, lon: Sequence[float], lat: Sequence[float], bands: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """Sample band values at points. Rows outside the grid or on nodata are NaN."""
        bands = list(bands or self.metadata.bands)
        idx = self.band_indexes(bands)
        lon_a = np.asarray(lon, dtype="float64")
        lat_a = np.asarray(lat, dtype="float64")
        out = np.full((lon_a.size, len(bands)), np.nan, dtype="float32")
        if lon_a.size == 0:
            return pd.DataFrame(out, columns=bands)
        with rasterio.open(self.path) as src:
            rows, cols = rasterio.transform.rowcol(src.transform, lon_a, lat_a)
            rows = np.asarray(rows)
            cols = np.asarray(cols)
            inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
            if inside.any():
                vals = np.array(
                    list(src.sample(zip(lon_a[inside], lat_a[inside], strict=True), indexes=idx)),
                    dtype="float32",
                )
                out[inside] = self._masked(vals)
        return pd.DataFrame(out, columns=bands)

    # -------------------------------------------------------------- grids
    def iter_blocks(
        self, bands: Sequence[str], block_rows: int | None = None
    ) -> Iterator[RasterBlock]:
        """Row-block iterator over the grid (for projection). `row_off` is relative to this
        view; by default blocks hold ~0.5M cells whatever the resolution."""
        idx = self.band_indexes(bands)
        rows = block_rows or max(1, BLOCK_CELLS // max(self.width, 1))
        col0 = int(self._window.col_off) if self._window is not None else 0
        row0 = int(self._window.row_off) if self._window is not None else 0
        with rasterio.open(self.path) as src:
            for row_off in range(0, self.height, rows):
                h = min(rows, self.height - row_off)
                win = Window(col0, row0 + row_off, self.width, h)
                data = self._masked(src.read(idx, window=win))
                yield RasterBlock(row_off, h, self.width, src.window_transform(win), data)

    def read_overview(
        self, bands: Sequence[str], max_width: int = 1440
    ) -> tuple[np.ndarray, Affine]:
        """Decimated read of the whole grid (for VIF sampling, previews)."""
        return read_raster_overview(self.path, self.band_indexes(bands), max_width, self._masked)

    def random_valid_samples(
        self, n: int, bands: Sequence[str] | None = None, seed: int = 0, max_width: int = 2160
    ) -> pd.DataFrame:
        bands = list(bands or self.metadata.bands)
        arr, _ = self.read_overview(bands, max_width=max_width)
        flat = arr.reshape(len(bands), -1).T
        valid = flat[~np.isnan(flat).any(axis=1)]
        rng = np.random.default_rng(seed)
        take = rng.choice(len(valid), size=min(n, len(valid)), replace=False)
        return pd.DataFrame(valid[take], columns=bands)


# ---------------------------------------------------------------------------
# Generic helpers for derived rasters (suitability, MESS, zones)
# ---------------------------------------------------------------------------
def _default_mask(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    if nodata is not None and not np.isnan(nodata):
        arr[arr == nodata] = np.nan
    return arr


def read_raster_overview(
    path: Path,
    indexes: Sequence[int] | int = 1,
    max_width: int = 1440,
    mask_fn: Any = None,
    resampling: Resampling = Resampling.average,
) -> tuple[np.ndarray, Affine]:
    """Decimated read. Use Resampling.nearest for categorical rasters (e.g. zones)."""
    with rasterio.open(path) as src:
        scale = max(1.0, src.width / max_width)
        out_w = max(1, int(round(src.width / scale)))
        out_h = max(1, int(round(src.height / scale)))
        idx = [indexes] if isinstance(indexes, int) else list(indexes)
        arr = src.read(idx, out_shape=(len(idx), out_h, out_w), resampling=resampling)
        transform = src.transform @ Affine.scale(src.width / out_w, src.height / out_h)
        arr = mask_fn(arr) if mask_fn else _default_mask(arr, src.nodata)
    if isinstance(indexes, int):
        arr = arr[0]
    return arr, transform


def sample_raster_file(
    path: Path, lon: Sequence[float], lat: Sequence[float], band: int = 1
) -> np.ndarray:
    """Point-sample a single-band derived raster; NaN outside grid / on nodata."""
    lon_a = np.asarray(lon, dtype="float64")
    lat_a = np.asarray(lat, dtype="float64")
    out = np.full(lon_a.size, np.nan, dtype="float32")
    if lon_a.size == 0:
        return out
    with rasterio.open(path) as src:
        rows, cols = rasterio.transform.rowcol(src.transform, lon_a, lat_a)
        rows, cols = np.asarray(rows), np.asarray(cols)
        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        if inside.any():
            vals = np.array(
                [
                    v[0]
                    for v in src.sample(
                        zip(lon_a[inside], lat_a[inside], strict=True), indexes=[band]
                    )
                ],
                dtype="float32",
            )
            out[inside] = _default_mask(vals, src.nodata)
    return out


def cell_area_km2(transform: Affine, rows: np.ndarray) -> np.ndarray:
    """Approximate area (km²) of a geographic (EPSG:4326) grid cell for the given row indices."""
    dx, dy = abs(transform.a), abs(transform.e)
    lat = transform.f - (rows + 0.5) * dy
    km_per_deg = 111.32
    return (dx * km_per_deg) * (dy * km_per_deg) * np.cos(np.deg2rad(lat))


def raster_window_stats(
    path: Path, bbox: tuple[float, float, float, float], threshold: float
) -> dict[str, float]:
    """Area-weighted stats of a single-band raster inside a lon/lat box.

    Returns land area (km²), mean value and the fraction of land area with value ≥ threshold.
    """
    minx, miny, maxx, maxy = bbox
    with rasterio.open(path) as src:
        win = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        arr = _default_mask(src.read(1, window=win), src.nodata)
        rows = np.arange(int(win.row_off), int(win.row_off) + arr.shape[0])
        area = np.repeat(cell_area_km2(src.transform, rows)[:, None], arr.shape[1], axis=1)
    valid = ~np.isnan(arr)
    land = float(area[valid].sum())
    if land == 0:
        return {"land_area_km2": 0.0, "mean": float("nan"), "fraction_at_or_above": float("nan")}
    return {
        "land_area_km2": round(land, 1),
        "mean": round(float((arr[valid] * area[valid]).sum() / land), 4),
        "fraction_at_or_above": round(float(area[valid & (arr >= threshold)].sum() / land), 4),
    }
