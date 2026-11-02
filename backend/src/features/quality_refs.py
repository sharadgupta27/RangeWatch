"""Versioned reference gazetteers for the data-quality tests (features/quality.py).

Like the bioclim stack this is part of the global static layer: built once, versioned, never
modified after a model has used it, and recorded (version + SHA-256 per table) in every model
version's lineage.

    data/reference/<version>/
        capitals.csv            name, country, longitude, latitude       (Natural Earth)
        country_centroids.csv   name, iso_a3, kind, longitude, latitude  (Natural Earth admin-0)
        province_centroids.csv  name, iso_a3, kind, longitude, latitude  (Natural Earth admin-1)
        institutions.csv        key, name, types, country, longitude, latitude (GBIF GRSciColl)
        metadata.json

Centroids are stored twice per polygon, as the geometric centroid and as Natural Earth's
label point, because gazetteers geocode "country only" records to either.

Build (needs network; GRSciColl goes through the GBIF connector):

    python -m src.features.quality_refs build --version quality_refs_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

METADATA_FILE = "metadata.json"
TABLES = ("capitals", "country_centroids", "province_centroids", "institutions")
NATURAL_EARTH = "https://naciscdn.org/naturalearth/10m/cultural/{name}.zip"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class QualityReferences:
    version: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def table(self, name: str) -> pd.DataFrame | None:
        return self.tables.get(name)

    def describe(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tables": {
                name: {
                    "n_rows": len(df),
                    "sha256": (self.metadata.get("files", {}).get(name) or {}).get("sha256"),
                    "source": (self.metadata.get("files", {}).get(name) or {}).get("source"),
                }
                for name, df in self.tables.items()
            },
            "built_ts": self.metadata.get("built_ts"),
        }

    @classmethod
    def load(cls, root: Path, version: str) -> QualityReferences | None:
        """None if the version directory does not exist; missing tables are left out (the
        tests that need them are then reported as skipped)."""
        vdir = root / version
        if not vdir.is_dir():
            return None
        meta_path = vdir / METADATA_FILE
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        tables = {
            name: pd.read_csv(vdir / f"{name}.csv")
            for name in TABLES
            if (vdir / f"{name}.csv").exists()
        }
        return cls(version=version, tables=tables, metadata=meta)


def write_references(
    root: Path, version: str, tables: dict[str, pd.DataFrame], sources: dict[str, str]
) -> Path:
    vdir = root / version
    if vdir.exists() and any(vdir.iterdir()):
        raise FileExistsError(f"{vdir} exists: reference versions are immutable, pick a new tag")
    vdir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for name, df in tables.items():
        path = vdir / f"{name}.csv"
        df.to_csv(path, index=False)
        files[name] = {"rows": len(df), "sha256": _sha256(path), "source": sources.get(name)}
    meta = {"version": version, "built_ts": datetime.now(UTC).isoformat(), "files": files}
    (vdir / METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return vdir


# ------------------------------------------------------------------ builders
def _natural_earth(name: str, workdir: Path) -> Any:
    import geopandas as gpd
    import requests

    url = NATURAL_EARTH.format(name=name)
    path = workdir / f"{name}.zip"
    resp = requests.get(url, timeout=(10, 300), headers={"User-Agent": "RangeWatch"})
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return gpd.read_file(path)


def centroid_table(gdf: Any, name_col: str, iso_col: str) -> pd.DataFrame:
    """Geometric centroid and label point of every polygon (equal-area centroid)."""
    eq = gdf.to_crs("ESRI:54034")  # World Cylindrical Equal Area
    cent = eq.geometry.centroid.to_crs("EPSG:4326")
    rows = []
    for i, row in gdf.iterrows():
        base = {"name": row.get(name_col), "iso_a3": row.get(iso_col)}
        rows.append({**base, "kind": "centroid", "longitude": cent[i].x, "latitude": cent[i].y})
        lx, ly = row.get("LABEL_X"), row.get("LABEL_Y")
        if pd.notna(lx) and pd.notna(ly):
            rows.append({**base, "kind": "label_point", "longitude": lx, "latitude": ly})
    return pd.DataFrame(rows)


def build(root: Path, version: str) -> Path:
    from src.connectors.gbif_client import GbifClient

    tables: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        log.info("Natural Earth admin-0 countries")
        countries = _natural_earth("ne_10m_admin_0_countries", work)
        tables["country_centroids"] = centroid_table(countries, "NAME", "ADM0_A3")
        sources["country_centroids"] = NATURAL_EARTH.format(name="ne_10m_admin_0_countries")
        log.info("Natural Earth admin-1 states/provinces")
        provinces = _natural_earth("ne_10m_admin_1_states_provinces", work)
        tables["province_centroids"] = centroid_table(provinces, "name", "adm0_a3")
        sources["province_centroids"] = NATURAL_EARTH.format(name="ne_10m_admin_1_states_provinces")
        log.info("Natural Earth populated places (capitals)")
        places = _natural_earth("ne_10m_populated_places", work)
        caps = places[places["FEATURECLA"].astype(str).str.contains("Admin-0 capital")]
        tables["capitals"] = pd.DataFrame(
            {
                "name": caps["NAME"].to_numpy(),
                "country": caps["ADM0NAME"].to_numpy(),
                "longitude": caps.geometry.x.to_numpy(),
                "latitude": caps.geometry.y.to_numpy(),
            }
        )
        sources["capitals"] = NATURAL_EARTH.format(name="ne_10m_populated_places")
    log.info("GBIF GRSciColl institutions")
    inst = GbifClient().fetch_institutions()
    tables["institutions"] = pd.DataFrame(
        {
            "key": [i.key for i in inst],
            "name": [i.name for i in inst],
            "types": [";".join(i.types) for i in inst],
            "country": [i.country for i in inst],
            "longitude": [i.longitude for i in inst],
            "latitude": [i.latitude for i in inst],
        }
    )
    sources["institutions"] = "https://api.gbif.org/v1/grscicoll/institution"
    return write_references(root, version, tables, sources)


def main() -> None:
    from src.config import get_settings

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="download and write a new reference version")
    b.add_argument("--version", default=get_settings().quality_reference_version)
    args = parser.parse_args()
    if args.cmd == "build":
        out = build(get_settings().quality_reference_root, args.version)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
