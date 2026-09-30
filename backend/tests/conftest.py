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
from src.connectors.gbif_client import TaxonSuggestion
from src.domain import BIOCLIM_BANDS, OccurrenceRecord
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


def add_synthetic_scenario(bioclim_root: Path, name: str, warming: float = 4.0) -> None:
    """Register a future-climate stack: same grid, BIO1 (and derived bands) warmer."""
    vdir = bioclim_root / BIOCLIM_VERSION
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
    return Settings(
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
