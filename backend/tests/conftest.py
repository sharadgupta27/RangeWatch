"""Shared fixtures: a small synthetic global bioclim stack and offline GBIF/iNat fakes.

No test touches the live network (CLAUDE.md): external clients are replaced by fakes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.config import Settings
from src.connectors.gbif_client import EffortDensity, TaxonSuggestion
from src.domain import BIOCLIM_BANDS, OccurrenceRecord, ReferenceRecord
from src.features.raster_sampler import METADATA_FILENAME, STACK_FILENAME
from src.modeling.maxent_trainer import TrainingConfig

TAXON_KEY = 5_000_001
BIOCLIM_VERSION = "bioclim_test"
NODATA = -9999.0


def is_land(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Two synthetic continents: 'America' and 'Eurasia'."""
    america = (lon >= -125) & (lon <= -35) & (lat >= -50) & (lat <= 70)
    eurasia = (lon >= -10) & (lon <= 145) & (lat >= -35) & (lat <= 70)
    return america | eurasia


def synthetic_bands(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(7)
    bio1 = 28.0 - 0.45 * np.abs(lat)  # temperature ~ latitude
    bio12 = 2200 * np.exp(-((lat / 32.0) ** 2)) + 4 * ((lon + 180) % 60)  # precipitation
    bio4 = 12 * np.abs(lat) + 0.8 * np.abs(lon) + 150  # seasonality
    bands = []
    for i in range(1, 20):
        if i == 1:
            b = bio1
        elif i == 12:
            b = bio12
        elif i == 4:
            b = bio4
        else:
            a, c = rng.uniform(-1, 1, 2)
            b = a * bio1 * 10 + c * bio12 * 0.1 + rng.normal(0, 1, size=lat.shape)
        bands.append(b)
    return np.stack(bands).astype("float32")


def build_stack(root: Path, res: float = 1.0) -> Path:
    vdir = root / BIOCLIM_VERSION
    vdir.mkdir(parents=True, exist_ok=True)
    w, h = int(360 / res), int(180 / res)
    lon = -180 + (np.arange(w) + 0.5) * res
    lat = 90 - (np.arange(h) + 0.5) * res
    lon2, lat2 = np.meshgrid(lon, lat)
    data = synthetic_bands(lon2, lat2)
    data[:, ~is_land(lon2, lat2)] = NODATA
    profile = dict(
        driver="GTiff",
        width=w,
        height=h,
        count=19,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-180, 90, res, res),
        nodata=NODATA,
    )
    with rasterio.open(vdir / STACK_FILENAME, "w", **profile) as dst:
        dst.write(data)
    meta = {
        "version": BIOCLIM_VERSION,
        "source": "synthetic test stack",
        "resolution": f"{res}deg",
        "crs": "EPSG:4326",
        "bands": list(BIOCLIM_BANDS),
        "created": "2026-01-01T00:00:00+00:00",
        "sha256": "test",
        "selected_predictors": ["bio1", "bio4", "bio12"],
        "vif": {"bio1": 1.5, "bio4": 1.4, "bio12": 1.2},
        "scenarios": {},
    }
    (vdir / METADATA_FILENAME).write_text(json.dumps(meta), encoding="utf-8")
    return vdir


def add_synthetic_scenario(
    bioclim_root: Path, name: str, warming: float = 4.0, version: str = BIOCLIM_VERSION
) -> None:
    """Register a future-climate stack: same grid, BIO1 (and derived bands) warmer."""
    vdir = bioclim_root / version
    with rasterio.open(vdir / STACK_FILENAME) as src:
        data = src.read()
        profile = src.profile
    land = data[0] != NODATA
    data[0][land] += warming
    rel = f"scenarios/{name}.tif"
    (vdir / "scenarios").mkdir(exist_ok=True)
    with rasterio.open(vdir / rel, "w", **profile) as dst:
        dst.write(data)
    meta = json.loads((vdir / METADATA_FILENAME).read_text(encoding="utf-8"))
    meta["scenarios"] = {**meta.get("scenarios", {}), name: rel}
    (vdir / METADATA_FILENAME).write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture(scope="session")
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-wide data root holding the synthetic global static layer (built once)."""
    root = tmp_path_factory.mktemp("data")
    build_stack(root / "bioclim")
    (root / "native_range_polygons").mkdir()
    return root


@pytest.fixture
def settings(tmp_path: Path, data_root: Path) -> Settings:
    # Isolated from the developer's .env / exported SDM_* variables: explicit values win.
    return Settings(
        _env_file=None,
        crosscheck_bioclim_version=None,
        database_url="sqlite://",  # unused: tests use the in-memory repository
        data_root=data_root,
        artifact_root=tmp_path / "artifacts",
        bioclim_version=BIOCLIM_VERSION,
        inat_enabled=True,
        random_seed=42,
        mlflow_tracking_uri=None,
    )


FAST_TRAINING = TrainingConfig(
    tune=False,
    feature_classes="LQ",
    n_background=1500,
    n_folds=3,
    importance_repeats=2,
    thin_km=50,
    min_presences=10,
    seed=42,
)


def box_records(
    n: int,
    lon: tuple[float, float],
    lat: tuple[float, float],
    start_id: int,
    label: str = "unknown",
    year: int = 2015,
    seed: int = 0,
    source: str = "GBIF",
) -> list[OccurrenceRecord]:
    rng = np.random.default_rng(seed)
    xs, ys = rng.uniform(*lon, n), rng.uniform(*lat, n)
    return [
        OccurrenceRecord(
            taxon_key=TAXON_KEY,
            source=source,
            record_id=start_id + i,  # type: ignore[arg-type]
            longitude=float(x),
            latitude=float(y),
            event_date=date(year + (i % 8), 6, 1),
            coordinate_uncertainty_m=100.0,
            basis_of_record="HUMAN_OBSERVATION",
            range_label=label,  # type: ignore[arg-type]
        )
        for i, (x, y) in enumerate(zip(xs, ys, strict=True))
    ]


@dataclass
class FakeGbif:
    """Offline GBIF: initial records + a queue of deltas returned by incremental searches."""

    initial: list[OccurrenceRecord]
    deltas: list[list[OccurrenceRecord]] = field(default_factory=list)
    download_enabled: bool = False
    calls: list[datetime | None] = field(default_factory=list)
    effort_error: Exception | None = None
    effort_calls: list[tuple[int, tuple[float, ...]]] = field(default_factory=list)
    absences: list[ReferenceRecord] = field(default_factory=list)
    absence_calls: list[datetime | None] = field(default_factory=list)
    absence_error: Exception | None = None
    effort_filters: list[dict] = field(default_factory=list)

    def get_taxon(self, taxon_key: int) -> TaxonSuggestion:
        return TaxonSuggestion(
            taxon_key,
            "Testus invasivus (L.)",
            "Testus invasivus",
            "SPECIES",
            "Plantae",
            "Tracheophyta",
            "Magnoliopsida",
            "Lamiales",
            "Verbenaceae",
            "Testus",
            class_key=220,
            order_key=408,
            family_key=6689,
            genus_key=9_000,
            phylum_key=7_707_728,
        )

    def match_name(self, name: str):
        """Backbone match: the test species (fuzzy for misspellings), nothing otherwise."""
        low = name.lower()
        if low.startswith("testus invasivus"):
            return self.get_taxon(TAXON_KEY), "EXACT", 99
        if low.startswith("testus invasivs"):
            return self.get_taxon(TAXON_KEY), "FUZZY", 90
        return None

    def fetch_effort_density(
        self,
        taxon_key: int,
        bbox: tuple[float, float, float, float],
        max_pixel_deg: float,
        basis_of_record=None,
        years=None,
    ) -> EffortDensity:
        """Synthetic observer bias: effort on every 0.5° pixel, but only south of 50°N."""
        self.effort_calls.append((taxon_key, tuple(bbox)))
        self.effort_filters.append({"basis_of_record": basis_of_record, "years": years})
        if self.effort_error is not None:
            raise self.effort_error
        west, south, east, north = bbox
        lon, lat = np.meshgrid(
            np.arange(np.floor(west) + 0.25, east, 0.5),
            np.arange(np.floor(south) + 0.25, north, 0.5),
        )
        keep = lat.ravel() < 50
        return EffortDensity(
            taxon_key=taxon_key,
            zoom=2,
            pixel_deg=0.5,
            n_tiles=1,
            lon=lon.ravel()[keep].tolist(),
            lat=lat.ravel()[keep].tolist(),
            count=[3] * int(keep.sum()),
            query="fake://density",
        )

    def suggest_species(self, query: str, limit: int = 10) -> list[TaxonSuggestion]:
        return [self.get_taxon(TAXON_KEY)]

    def search_occurrences(
        self,
        taxon_key: int,
        since: datetime | None = None,
        until: datetime | None = None,
        on_progress=None,
    ) -> list[OccurrenceRecord]:
        self.calls.append(since)
        if since is None:
            return list(self.initial)
        return self.deltas.pop(0) if self.deltas else []

    def search_absences(
        self,
        taxon_key: int,
        since: datetime | None = None,
        until: datetime | None = None,
        on_progress=None,
    ) -> list[ReferenceRecord]:
        self.absence_calls.append(since)
        if self.absence_error is not None:
            raise self.absence_error
        return list(self.absences) if since is None else []

    def download_occurrences(self, taxon_key: int, workdir: Path):  # pragma: no cover
        raise AssertionError("download API must not be used without credentials")


@dataclass
class FakeInatTaxon:
    inat_taxon_id: int = 777
    name: str = "Testus invasivus"
    common_name: str | None = "Test weed"
    rank: str | None = "species"
    thumbnail_url: str | None = "https://example.org/t.jpg"
    observations_count: int | None = 10


@dataclass
class FakeInat:
    observations: list[OccurrenceRecord] = field(default_factory=list)

    def find_taxon(self, scientific_name: str) -> FakeInatTaxon:
        return FakeInatTaxon()

    def autocomplete(self, query: str, limit: int = 10) -> list[FakeInatTaxon]:
        return [FakeInatTaxon()]

    def get_observations(
        self,
        inat_taxon_id: int,
        taxon_key: int,
        updated_since: datetime | None = None,
        on_progress=None,
    ) -> list[OccurrenceRecord]:
        return list(self.observations) if updated_since is None else []


def native_box_geojson() -> dict:
    return {"type": "Polygon", "coordinates": [[[-2, 30], [45, 30], [45, 60], [-2, 60], [-2, 30]]]}


def now_utc() -> datetime:
    return datetime.now(UTC)


TEMPERATURE_BANDS = {1, 5, 6, 8, 9, 10, 11}


def write_synthetic_chelsa(src_dir: Path, res: float = 0.5) -> Path:
    """CHELSA-like files: own grid (84°N–56°S), values everywhere incl. oceans, packed with a
    declared scale/offset (Kelvin for temperatures) and BIO3 as a ratio instead of percent."""
    src_dir.mkdir(parents=True, exist_ok=True)
    w, h = int(360 / res), int(140 / res)
    lon = -180 + (np.arange(w) + 0.5) * res
    lat = 84 - (np.arange(h) + 0.5) * res
    data = synthetic_bands(*np.meshgrid(lon, lat))
    for i in range(1, 20):
        offset = -273.15 if i in TEMPERATURE_BANDS else 0.0
        physical = data[i - 1] / (100.0 if i == 3 else 1.0)
        raw = ((physical - offset) / 0.1).astype("float32")
        with rasterio.open(
            src_dir / f"CHELSA_bio{i}_1981-2010_V.2.1.tif",
            "w",
            driver="GTiff",
            width=w,
            height=h,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(-180, 84, res, res),
        ) as dst:
            dst.write(raw, 1)
            dst.scales = (0.1,)
            dst.offsets = (offset,)
    return src_dir


def write_synthetic_stack_tif(path: Path, res: float = 0.5) -> Path:
    """A 19-band stack of the synthetic climate at a finer resolution (a 'high-res' layer)."""
    w, h = int(360 / res), int(180 / res)
    lon2, lat2 = np.meshgrid(-180 + (np.arange(w) + 0.5) * res, 90 - (np.arange(h) + 0.5) * res)
    data = synthetic_bands(lon2, lat2)
    data[:, ~is_land(lon2, lat2)] = NODATA
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=w,
        height=h,
        count=19,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-180, 90, res, res),
        nodata=NODATA,
    ) as dst:
        dst.write(data)
    return path


# Interrupted Goode Homolosine, the native projection of the SoilGrids rasters.
IGH = "+proj=igh +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
SOIL_DEPTHS = ("0-5cm", "5-15cm", "15-30cm")
# A region with no SoilGrids prediction (inside the synthetic 'Eurasia').
SOIL_GAP = ((100.0, 120.0), (20.0, 40.0))


def synthetic_soil(prop: str, depth: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Mapped (integer-unit) SoilGrids values. pH is constant in space but differs by depth
    (6.0 / 7.0 / 8.0), so the 0–30 cm weighting is checkable: (5·6 + 10·7 + 15·8) / 30."""
    d = SOIL_DEPTHS.index(depth)
    if prop == "phh2o":
        return np.full_like(lon, 60.0 + 10.0 * d)
    if prop == "sand":
        return 300.0 + 2.0 * lon  # g/kg → % at 0.1
    if prop == "clay":
        return 250.0 + 2.0 * lat
    if prop == "soc":
        return 50.0 + 2.0 * np.abs(lat)
    if prop == "bdod":
        return 130.0 + 0.2 * lon
    if prop == "wv0033":
        return 300.0 + lat
    if prop == "wv1500":
        return np.full_like(lon, 100.0)
    raise ValueError(prop)


def write_synthetic_soilgrids(
    src_dir: Path,
    props: tuple[str, ...] = ("phh2o", "sand", "clay", "soc", "bdod", "wv0033", "wv1500"),
    res_tag: str = "5000",
    pixel_m: float = 50_000.0,
) -> Path:
    """SoilGrids-like files named as on files.isric.org: int16 in IGH, nodata −32768 in the
    projection's interruptions, over the oceans' far south and in SOIL_GAP."""
    from pyproj import Transformer
    from rasterio.crs import CRS

    src_dir.mkdir(parents=True, exist_ok=True)
    x0, x1, y0, y1 = -20_037_508.0, 20_037_508.0, -8_683_259.0, 8_683_259.0
    w, h = int((x1 - x0) / pixel_m), int((y1 - y0) / pixel_m)
    xs = x0 + (np.arange(w) + 0.5) * pixel_m
    ys = y1 - (np.arange(h) + 0.5) * pixel_m
    xx, yy = np.meshgrid(xs, ys)
    crs = CRS.from_string(IGH)
    # pyproj (not GDAL) so the projection's interruptions come back as inf, not an error.
    to_geo = Transformer.from_crs(IGH, "EPSG:4326", always_xy=True)
    lon, lat = to_geo.transform(xx.ravel(), yy.ravel(), errcheck=False)
    lon = np.asarray(lon).reshape(h, w)
    lat = np.asarray(lat).reshape(h, w)
    with np.errstate(invalid="ignore"):
        bad = ~np.isfinite(lon) | ~np.isfinite(lat) | (np.abs(lon) > 180) | (np.abs(lat) > 90)
        (gx0, gx1), (gy0, gy1) = SOIL_GAP
        bad |= (lon >= gx0) & (lon <= gx1) & (lat >= gy0) & (lat <= gy1)
    lon_f, lat_f = np.where(bad, 0.0, lon), np.where(bad, 0.0, lat)
    for prop in props:
        for depth in SOIL_DEPTHS:
            vals = np.round(synthetic_soil(prop, depth, lon_f, lat_f)).astype("int16")
            vals[bad] = -32768
            with rasterio.open(
                src_dir / f"{prop}_{depth}_mean_{res_tag}.tif",
                "w",
                driver="GTiff",
                width=w,
                height=h,
                count=1,
                dtype="int16",
                crs=crs,
                transform=from_origin(x0, y1, pixel_m, pixel_m),
                nodata=-32768,
            ) as dst:
                dst.write(vals, 1)
    return src_dir


def write_synthetic_chelsa_extras(
    src_dir: Path,
    variables: tuple[str, ...] = ("hurs_mean", "gdd5"),
    future: tuple[str, str, str] | None = None,
    future_variables: tuple[str, ...] = ("gdd5",),
    res: float = 0.5,
) -> Path:
    """CHELSA-BIOCLIM+-like files on the CHELSA grid, packed with a declared scale (0.1).
    With `future=(gcm, ssp, period)`: also that CMIP6 future's BIO1–19 (BIO1 +3 °C) and the
    `future_variables` (CHELSA publishes e.g. gdd5 for futures, but not hurs)."""
    src_dir.mkdir(parents=True, exist_ok=True)
    w, h = int(360 / res), int(140 / res)
    lon2, lat2 = np.meshgrid(-180 + (np.arange(w) + 0.5) * res, 84 - (np.arange(h) + 0.5) * res)

    def write(name: str, physical: np.ndarray, offset: float = 0.0) -> None:
        with rasterio.open(
            src_dir / name,
            "w",
            driver="GTiff",
            width=w,
            height=h,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(-180, 84, res, res),
        ) as dst:
            dst.write(((physical - offset) / 0.1).astype("float32"), 1)
            dst.scales = (0.1,)
            dst.offsets = (offset,)

    def extra(var: str, warm: float = 0.0) -> np.ndarray:
        if var.startswith("hurs"):
            return 40.0 + 0.4 * np.abs(lat2) + 0.05 * lon2  # % — varies independently of bio1
        if var.startswith("gdd"):
            return 4000.0 - 60.0 * np.abs(lat2) + 300.0 * warm
        return 10.0 + 0.1 * lon2

    for var in variables:
        write(f"CHELSA_{var}_1981-2010_V.2.1.tif", extra(var))
    if future is not None:
        gcm, ssp, period = future
        data = synthetic_bands(lon2, lat2)
        data[0] += 3.0
        for i in range(1, 20):
            offset = -273.15 if i in TEMPERATURE_BANDS else 0.0
            write(
                f"CHELSA_bio{i}_{period}_{gcm.lower()}_{ssp}_V.2.1.tif",
                data[i - 1] / (100.0 if i == 3 else 1.0),
                offset,
            )
        for var in future_variables:
            write(f"CHELSA_{var}_{period}_{gcm.lower()}_{ssp}_V.2.1.tif", extra(var, warm=1.0))
    return src_dir
