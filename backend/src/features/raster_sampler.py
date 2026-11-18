"""Single access point for reading rasters (CLAUDE.md: never open bioclim GeoTIFFs ad hoc).

The global bioclim stack is a versioned, 19-band Cloud-Optimized GeoTIFF:

    data/bioclim/<version>/bioclim.tif
    data/bioclim/<version>/metadata.json
    data/bioclim/<version>/scenarios/<name>.tif      (optional CMIP6 future stacks)
    data/bioclim/<version>/hires/bioclim_<res>.tif   (optional finer stack, same source)
    data/bioclim/<version>/static_layers/<name>.tif  (optional static predictors, e.g. soil)

Static predictor layers (`metadata.static_layers`, e.g. SoilGrids topsoil properties added by
`bioclim_store add-soil`) are single-band rasters on the version grid. `BioclimStack` serves
them next to the climate bands under their own names, so sampling, projection, MESS and the
extrapolation diagnostics see one predictor set. They do not change with the climate: a
future scenario or a high-resolution view reuses the same layers (resampled with
nearest-neighbour onto a finer grid unless a layer registers its own file for that
resolution under `hires`).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
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
    # Static observer-bias rasters on this grid (B6: travel time to cities, distance to roads,
    # …): name → {path, direction, transform, citation, source, sha256}. `direction` says
    # whether larger values mean *more* ("increasing") or *less* ("decreasing") effort.
    bias_layers: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Static environmental predictors on this grid (e.g. SoilGrids soil properties): name →
    # {path, kind, description, units, source, citation, sha256, hires: {res: path}, …}.
    # Unlike bias layers they are ordinary predictors, kept unchanged for prediction.
    static_layers: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Extra *climate* predictors outside the 19 BIO bands (e.g. CHELSA-BIOCLIM+ relative
    # humidity, VPD, wind): name → {path, scenarios: {scenario: path}, hires: {res: path},
    # description, group, units, source, citation, sha256}. They change with the climate, so a
    # scenario view needs the layer's own file for that scenario; without one the layer is
    # present-only for it (see `unavailable_in_scenario`).
    climate_layers: dict[str, dict[str, Any]] = field(default_factory=dict)
    # "worldclim" / "chelsa" (None: inferred from `source` for older versions) and the
    # climate baseline period, e.g. "1970-2000".
    source_family: str | None = None
    baseline_period: str | None = None

    @property
    def predictor_bands(self) -> tuple[str, ...]:
        """Every predictor this version offers: the stack's climate bands, extra climate
        layers, then static layers (each in registration order)."""
        extra = [*self.climate_layers, *self.static_layers]
        return (*self.bands, *dict.fromkeys(n for n in extra if n not in self.bands))

    @property
    def family(self) -> str:
        """Climate data family of the version ("worldclim", "chelsa" or "other")."""
        if self.source_family:
            return self.source_family
        src = self.source.lower()
        if src.startswith("worldclim"):
            return "worldclim"
        if src.startswith("chelsa"):
            return "chelsa"
        return "other"

    @property
    def baseline(self) -> str | None:
        """Climate baseline period ("1970-2000" for WorldClim v2.1, "1981-2010" for CHELSA)."""
        return self.baseline_period or {"worldclim": "1970-2000", "chelsa": "1981-2010"}.get(
            self.family
        )

    def unavailable_in_scenario(self, scenario: str, predictors: Sequence[str]) -> list[str]:
        """Predictors that have no data for `scenario` (present-only climate layers). Stack
        bands and static layers (soil, held constant) are always available."""
        return [
            p
            for p in predictors
            if p in self.climate_layers
            and scenario not in (self.climate_layers[p].get("scenarios") or {})
        ]

    def present_only_layers(self) -> list[str]:
        """Climate layers without data for any registered scenario of this version."""
        return [
            n
            for n, e in self.climate_layers.items()
            if not set(e.get("scenarios") or {}) & set(self.scenarios)
        ]

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
            bias_layers={k: dict(v) for k, v in raw.get("bias_layers", {}).items()},
            static_layers={k: dict(v) for k, v in raw.get("static_layers", {}).items()},
            climate_layers={k: dict(v) for k, v in raw.get("climate_layers", {}).items()},
            source_family=raw.get("source_family"),
            baseline_period=raw.get("baseline_period"),
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
            "bias_layers": self.bias_layers,
            "static_layers": self.static_layers,
            "climate_layers": self.climate_layers,
            "source_family": self.source_family,
            "baseline_period": self.baseline_period,
        }

    def finest_hires(self) -> str | None:
        """Resolution code of the finest registered high-resolution stack, if any."""
        if not self.hires:
            return None
        return min(self.hires, key=lambda r: RESOLUTION_DEG.get(r, float("inf")))


class PresentOnlyPredictorError(KeyError):
    """A climate layer (e.g. relative humidity) has no data for a future scenario."""

    def __init__(self, predictor: str, scenario: str):
        super().__init__(
            f"{predictor} has no data for scenario {scenario} (present-climate only): a model "
            "using it cannot be projected onto this scenario"
        )
        self.predictor = predictor
        self.scenario = scenario


@dataclass(frozen=True)
class RasterBlock:
    row_off: int
    height: int
    width: int
    transform: Affine
    data: np.ndarray  # (bands, rows, cols) float32, NaN where nodata


class BioclimStack:
    """Read-only view of one bioclim stack version (current climate or a named scenario),
    including the version's static predictor layers (see the module docstring)."""

    def __init__(
        self,
        path: Path,
        metadata: BioclimMetadata,
        scenario: str | None = None,
        root: Path | None = None,
        hires: str | None = None,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Bioclim stack not found: {path}")
        self.path = path
        self.metadata = metadata
        self.scenario = scenario
        # Version directory (static layer paths are relative to it) and, for a
        # high-resolution view, its resolution code.
        self.root = root or (path.parent if path.name == STACK_FILENAME else path.parent.parent)
        self.hires = hires
        with rasterio.open(path) as src:
            self.width, self.height = src.width, src.height
            self.transform: Affine = src.transform
            self.crs = src.crs
            self.nodata = src.nodata
            self.res: tuple[float, float] = src.res
        # The file's full grid (a regional subset keeps reading in its coordinates).
        self._full_shape = (self.height, self.width)
        self._full_transform = self.transform
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
            return cls(root / meta.hires[hires], meta, root=root, hires=hires)
        if scenario is None:
            return cls(root / STACK_FILENAME, meta, root=root)
        if scenario not in meta.scenarios:
            raise KeyError(f"Scenario {scenario!r} not registered for {version}")
        return cls(root / meta.scenarios[scenario], meta, scenario=scenario, root=root)

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

    @property
    def predictor_bands(self) -> tuple[str, ...]:
        """Climate bands of the stack file plus the version's static layers."""
        return self.metadata.predictor_bands

    def band_indexes(self, bands: Sequence[str]) -> list[int]:
        """1-based indexes of climate bands in the stack file (static layers excluded)."""
        order = list(self.metadata.bands)
        return [order.index(b) + 1 for b in bands]

    def _split(self, bands: Sequence[str]) -> tuple[list[str], list[str]]:
        """(climate bands in the stack file, static layers), each in request order."""
        climate: list[str] = []
        static: list[str] = []
        for b in bands:
            if b in self.metadata.bands:
                climate.append(b)
            elif b in self.metadata.static_layers or b in self.metadata.climate_layers:
                static.append(b)
            else:
                raise KeyError(
                    f"Predictor {b!r} is neither a band nor a layer of "
                    f"{self.metadata.version} (have {', '.join(self.predictor_bands)})"
                )
        return climate, static

    def static_layer_path(self, name: str) -> Path:
        """File of a single-band layer for this view.

        Static layers (soil): the view's own high-resolution file when registered, otherwise
        the version-grid file, for every scenario. Climate layers: the scenario's own file
        (PresentOnlyPredictorError when the scenario has none), else as static layers."""
        if name in self.metadata.climate_layers:
            entry = self.metadata.climate_layers[name]
            if self.scenario is not None:
                rel = (entry.get("scenarios") or {}).get(self.scenario)
                if rel is None:
                    raise PresentOnlyPredictorError(name, self.scenario)
                return self.root / rel
        else:
            entry = self.metadata.static_layers[name]
        rel = (entry.get("hires") or {}).get(self.hires) if self.hires else None
        return self.root / (rel or entry["path"])

    @contextmanager
    def _open_static(self, name: str) -> Iterator[Any]:
        """A static layer as a dataset aligned to this stack's full grid (a nearest-neighbour
        warped view when the layer lives on another grid, e.g. under a high-resolution view)."""
        with rasterio.open(self.static_layer_path(name)) as src:
            h, w = self._full_shape
            if (src.height, src.width) == (h, w) and src.transform.almost_equals(
                self._full_transform
            ):
                yield src
                return
            with WarpedVRT(
                src,
                crs=self.crs,
                transform=self._full_transform,
                width=w,
                height=h,
                resampling=Resampling.nearest,
                src_nodata=src.nodata,
                nodata=src.nodata,
            ) as vrt:
                yield vrt

    def _masked(self, arr: np.ndarray, nodata: float | None = None) -> np.ndarray:
        arr = arr.astype("float32", copy=False)
        nodata = self.nodata if nodata is None else nodata
        if nodata is not None and not np.isnan(nodata):
            arr[arr == nodata] = np.nan
        # WorldClim uses a large negative fill value in some releases.
        arr[arr < -3.0e38] = np.nan
        return arr

    # -------------------------------------------------------------- points
    def sample(
        self, lon: Sequence[float], lat: Sequence[float], bands: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """Sample predictor values at points (climate bands and/or static layers). Rows
        outside the grid or on nodata are NaN."""
        bands = list(bands or self.predictor_bands)
        climate, static = self._split(bands)
        lon_a = np.asarray(lon, dtype="float64")
        lat_a = np.asarray(lat, dtype="float64")
        cols: dict[str, np.ndarray] = {}
        if climate:
            out = np.full((lon_a.size, len(climate)), np.nan, dtype="float32")
            if lon_a.size:
                with rasterio.open(self.path) as src:
                    out = _sample_dataset(src, lon_a, lat_a, self.band_indexes(climate), out)
                out = self._masked(out)
            cols.update({b: out[:, i] for i, b in enumerate(climate)})
        for name in static:
            # Point values come from the layer's own grid (no resampling needed).
            cols[name] = sample_raster_file(self.static_layer_path(name), lon_a, lat_a)
        return pd.DataFrame({b: cols[b] for b in bands}, columns=bands).astype("float32")

    # -------------------------------------------------------------- grids
    def iter_blocks(
        self, bands: Sequence[str], block_rows: int | None = None
    ) -> Iterator[RasterBlock]:
        """Row-block iterator over the grid (for projection). `row_off` is relative to this
        view; by default blocks hold ~0.5M cells whatever the resolution."""
        climate, static = self._split(bands)
        idx = self.band_indexes(climate)
        order = [("c", climate.index(b)) if b in climate else ("s", static.index(b)) for b in bands]
        rows = block_rows or max(1, BLOCK_CELLS // max(self.width, 1))
        col0 = int(self._window.col_off) if self._window is not None else 0
        row0 = int(self._window.row_off) if self._window is not None else 0
        with ExitStack() as es:
            src = es.enter_context(rasterio.open(self.path))
            layers = [es.enter_context(self._open_static(n)) for n in static]
            for row_off in range(0, self.height, rows):
                h = min(rows, self.height - row_off)
                win = Window(col0, row0 + row_off, self.width, h)
                clim = self._masked(src.read(idx, window=win)) if climate else None
                stat = [self._masked(ds.read(1, window=win), ds.nodata) for ds in layers]
                parts = [
                    clim[i] if kind == "c" and clim is not None else stat[i] for kind, i in order
                ]
                data = (
                    np.stack(parts).astype("float32", copy=False)
                    if parts
                    else np.empty((0, h, self.width), dtype="float32")
                )
                yield RasterBlock(row_off, h, self.width, src.window_transform(win), data)

    def read_overview(
        self, bands: Sequence[str], max_width: int = 1440
    ) -> tuple[np.ndarray, Affine]:
        """Decimated read of the whole grid (for VIF sampling, previews)."""
        climate, static = self._split(bands)
        if not static:
            return read_raster_overview(
                self.path, self.band_indexes(bands), max_width, self._masked
            )
        h, w = self._full_shape
        scale = max(1.0, w / max_width)
        out_w, out_h = max(1, int(round(w / scale))), max(1, int(round(h / scale)))
        cols: dict[str, np.ndarray] = {}
        if climate:
            with rasterio.open(self.path) as src:
                arr = src.read(
                    self.band_indexes(climate),
                    out_shape=(len(climate), out_h, out_w),
                    resampling=Resampling.average,
                )
            arr = self._masked(arr)
            cols.update({b: arr[i] for i, b in enumerate(climate)})
        for name in static:
            with self._open_static(name) as ds:
                a = ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.average)
                cols[name] = self._masked(a, ds.nodata)
        transform = self._full_transform @ Affine.scale(w / out_w, h / out_h)
        return np.stack([cols[b] for b in bands]).astype("float32", copy=False), transform

    def random_valid_samples(
        self, n: int, bands: Sequence[str] | None = None, seed: int = 0, max_width: int = 2160
    ) -> pd.DataFrame:
        bands = list(bands or self.predictor_bands)
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


def _sample_dataset(
    src: Any, lon: np.ndarray, lat: np.ndarray, indexes: list[int], out: np.ndarray
) -> np.ndarray:
    """Fill `out` (n, len(indexes)) with raw values at the points inside the dataset grid."""
    rows, cols = rasterio.transform.rowcol(src.transform, lon, lat)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
    if inside.any():
        out[inside] = np.array(
            list(src.sample(zip(lon[inside], lat[inside], strict=True), indexes=indexes)),
            dtype="float32",
        )
    return out


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


def list_versions(bioclim_root: Path) -> list[BioclimMetadata]:
    """Installed stack versions (directories with a metadata.json), sorted by tag."""
    if not bioclim_root.exists():
        return []
    return [
        BioclimMetadata.load(d / METADATA_FILENAME)
        for d in sorted(bioclim_root.iterdir())
        if (d / METADATA_FILENAME).is_file() and (d / STACK_FILENAME).is_file()
    ]


def choose_crosscheck_version(
    bioclim_root: Path, trained_version: str, candidates: Sequence[str | None]
) -> str | None:
    """Climate stack to cross-check a model trained on `trained_version` against: the first
    installed candidate from a *different* climate family (a CHELSA-trained model is checked
    against WorldClim and vice versa), else the first installed candidate that differs."""
    root = Path(bioclim_root)
    trained_path = root / trained_version / METADATA_FILENAME
    trained = BioclimMetadata.load(trained_path).family if trained_path.exists() else None
    installed = []
    for c in dict.fromkeys(c for c in candidates if c and c != trained_version):
        path = root / c / METADATA_FILENAME
        if path.exists():
            installed.append((c, BioclimMetadata.load(path).family))
    for c, family in installed:
        if family != trained:
            return c
    return installed[0][0] if installed else None


# Fields of a static layer entry copied verbatim into a model version's reproducibility record.
STATIC_LAYER_LINEAGE_FIELDS = (
    "kind",
    "description",
    "units",
    "depth",
    "source",
    "source_resolution_m",
    "citation",
    "sha256",
    "land_coverage",
    "static_under_scenarios",
)


def static_layer_lineage(meta: BioclimMetadata, predictors: Sequence[str]) -> list[dict[str, Any]]:
    """Every static layer of the version with its provenance and checksum, and whether the
    model used it (the VIF screen may have dropped it)."""
    return [
        {
            "name": name,
            "used": name in predictors,
            **{k: entry.get(k) for k in STATIC_LAYER_LINEAGE_FIELDS},
        }
        for name, entry in meta.static_layers.items()
    ]


BIAS_LAYER_TRANSFORMS = ("none", "log1p")


def sample_bias_layer(
    bioclim_root: Path, version: str, name: str, lon: Sequence[float], lat: Sequence[float]
) -> tuple[np.ndarray, dict[str, Any]]:
    """A registered static bias layer (with its transform) at points; returns the values and
    the layer's metadata entry. Raises KeyError when the layer is not registered."""
    root = bioclim_root / version
    meta = BioclimMetadata.load(root / METADATA_FILENAME)
    entry = meta.bias_layers[name]
    values = sample_raster_file(root / entry["path"], lon, lat).astype("float64")
    if entry.get("transform", "none") == "log1p":
        values = np.log1p(np.clip(values, 0, None))
    return values, entry


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


def write_binary(suitability: Path, threshold: float, path: Path) -> Path:
    """Suitable (1) / unsuitable (0) COG at `threshold`; nodata 255 off land."""
    import tempfile

    from rasterio.shutil import copy as rio_copy

    with rasterio.open(suitability) as src, tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / path.name
        profile = {
            "driver": "GTiff",
            "width": src.width,
            "height": src.height,
            "count": 1,
            "dtype": "uint8",
            "crs": src.crs,
            "transform": src.transform,
            "nodata": 255,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "deflate",
        }
        with rasterio.open(tmp_path, "w", **profile) as dst:
            for _, win in src.block_windows(1):
                v = src.read(1, window=win)
                out = np.where(v == src.nodata, 255, (v >= threshold).astype("uint8"))
                dst.write(out.astype("uint8"), 1, window=win)
        path.parent.mkdir(parents=True, exist_ok=True)
        rio_copy(
            tmp_path,
            path,
            driver="COG",
            compress="deflate",
            blocksize="512",
            overview_resampling="mode",
        )
    return path
