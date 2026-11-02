"""User-uploaded occurrence / survey files → reference records (evaluation only).

Accepted formats: delimited text (.csv, .tsv, .txt), GeoJSON (.geojson, .json), GeoPackage
(.gpkg) and zipped shapefiles (.zip). Spatial formats are reprojected to EPSG:4326 from their
declared CRS; delimited text must carry decimal-degree WGS84 coordinates.

Columns are detected from common Darwin Core and agency names unless given explicitly:
coordinates (validated: a candidate whose values are not valid degrees — e.g. projected
metres in an "ITM - LONG" column — is skipped), species name, date, coordinate uncertainty,
presence/absence and a record id. Rows of other species are dropped when a species column
exists. Every decision (columns used, rows skipped and why) is returned in a parse report that
is stored with the dataset.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.domain import OccurrenceStatus, ReferenceRecord

TEXT_SUFFIXES = {".csv", ".tsv", ".txt"}
SPATIAL_SUFFIXES = {".geojson", ".json", ".gpkg", ".zip"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | SPATIAL_SUFFIXES
# Upper bound on a zipped shapefile's extracted size (the upload limit bounds only the zip).
MAX_UNZIPPED_BYTES = 500 * 1024 * 1024

# Candidate column names, normalised (lower case, alphanumerics only), in priority order.
LON_NAMES = ("decimallongitude", "longitude", "lon", "lng", "long", "x")
LAT_NAMES = ("decimallatitude", "latitude", "lat", "y")
NAME_NAMES = ("scientificname", "species", "taxon", "taxonname", "acceptedname", "name")
DATE_NAMES = ("eventdate", "date", "observationdate", "dateevent", "obsdate", "year")
UNCERTAINTY_NAMES = (
    "coordinateuncertaintyinmeters",
    "coordinateuncertainty",
    "uncertainty",
    "accuracy",
    "positionalaccuracy",
    "precision",
)
STATUS_NAMES = ("occurrencestatus", "presence", "presenceabsence", "status", "detected")
ID_NAMES = ("occurrenceid", "id", "recordid", "objectid", "fid", "catalognumber")

ABSENT_VALUES = {"absent", "absence", "0", "false", "no", "n", "not detected", "not found"}
PRESENT_VALUES = {"present", "presence", "1", "true", "yes", "y", "detected", "found"}


class UploadError(ValueError):
    """The file cannot be turned into reference records (shown to the user as-is)."""


@dataclass(frozen=True)
class ColumnMapping:
    """Explicit column names; any left None is detected."""

    longitude: str | None = None
    latitude: str | None = None
    species: str | None = None
    date: str | None = None
    uncertainty: str | None = None
    status: str | None = None
    record_id: str | None = None


@dataclass
class ParsedUpload:
    records: list[ReferenceRecord]
    file_sha256: str
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def n_present(self) -> int:
        return sum(r.occurrence_status == "present" for r in self.records)

    @property
    def n_absent(self) -> int:
        return sum(r.occurrence_status == "absent" for r in self.records)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _candidates(
    columns: list[str], names: tuple[str, ...], contains: tuple[str, ...] = ()
) -> list[str]:
    """Columns matching `names` exactly (normalised, in priority order), then columns whose
    normalised name contains any of `contains`."""
    normed = {c: _norm(c) for c in columns}
    exact = [c for n in names for c in columns if normed[c] == n]
    fuzzy = [c for c in columns if c not in exact and any(k in normed[c] for k in contains)]
    return exact + fuzzy


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce"
    )


def _pick_coordinate(
    df: pd.DataFrame, cands: list[str], limit: float
) -> tuple[str | None, list[str]]:
    """First candidate whose non-null values are (almost all) valid degrees within ±limit."""
    rejected: list[str] = []
    for c in cands:
        v = _numeric(df[c]).dropna()
        if len(v) and (v.abs() <= limit).mean() >= 0.99:
            return c, rejected
        rejected.append(c)
    return None, rejected


def _read_table(content: bytes, suffix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    info: dict[str, Any] = {"format": suffix.lstrip(".")}
    if suffix in TEXT_SUFFIXES:
        text = content.decode("utf-8-sig", errors="replace")
        try:
            dialect = csv.Sniffer().sniff(text[:20_000], delimiters=",;\t|")
            sep = dialect.delimiter
        except csv.Error:
            sep = "\t" if suffix == ".tsv" else ","
        info["delimiter"] = sep
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
        df = df.replace({"": None})
        info["crs"] = "EPSG:4326 (assumed for delimited text)"
        return df, info

    import geopandas as gpd

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(content)
        if suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                shp = [n for n in zf.namelist() if n.lower().endswith(".shp")]
                if not shp:
                    raise UploadError("The zip archive contains no .shp file")
                if sum(i.file_size for i in zf.infolist()) > MAX_UNZIPPED_BYTES:
                    raise UploadError(
                        f"The zip archive expands beyond {MAX_UNZIPPED_BYTES // (1024 * 1024)} MB"
                    )
                zf.extractall(tmp)
            path = Path(tmp) / shp[0]
        gdf = gpd.read_file(path)
    if gdf.crs is None:
        info["crs"] = "none declared (assumed EPSG:4326)"
    else:
        info["crs"] = gdf.crs.to_string()
        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")
            info["reprojected_to"] = "EPSG:4326"
    geom = gdf.geometry
    if not geom.geom_type.isin(["Point"]).all():
        geom = geom.representative_point()
        info["geometry"] = "non-point features reduced to a representative point"
    df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    df["__lon"] = geom.x.to_numpy()
    df["__lat"] = geom.y.to_numpy()
    return df, info


def _status(value: Any) -> OccurrenceStatus | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "present"
    v = str(value).strip().lower()
    if v in ABSENT_VALUES:
        return "absent"
    if v in PRESENT_VALUES or v == "":
        return "present"
    try:  # counts / abundances: > 0 present, 0 absent
        return "present" if float(v) > 0 else "absent"
    except ValueError:
        return None


def _species_match(value: Any, names: list[str]) -> bool:
    if value is None:
        return False
    v = " ".join(str(value).lower().split())
    return any(v == n or v.startswith(n + " ") for n in names)


def parse_upload(
    content: bytes,
    filename: str,
    taxon_key: int,
    dataset_id: str,
    species_names: list[str],
    mapping: ColumnMapping | None = None,
    max_rows: int = 200_000,
) -> ParsedUpload:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UploadError(
            f"Unsupported file type '{suffix or filename}': use "
            + ", ".join(sorted(SUPPORTED_SUFFIXES))
        )
    mapping = mapping or ColumnMapping()
    try:
        df, info = _read_table(content, suffix)
    except UploadError:
        raise
    except Exception as exc:
        raise UploadError(f"Could not read {filename}: {exc}") from exc
    if df.empty:
        raise UploadError("The file has no rows")
    if len(df) > max_rows:
        raise UploadError(f"{len(df):,} rows exceed the limit of {max_rows:,}")
    cols = [c for c in df.columns if not str(c).startswith("__")]

    def explicit(name: str | None) -> str | None:
        if name is None:
            return None
        if name not in df.columns:
            raise UploadError(f"Column '{name}' not found; columns are: {', '.join(cols)}")
        return name

    used: dict[str, str | None] = {}
    rejected: dict[str, list[str]] = {}
    if "__lon" in df.columns:
        used["longitude"], used["latitude"] = "geometry", "geometry"
        lon, lat = df["__lon"].astype(float), df["__lat"].astype(float)
    else:
        lon_col = explicit(mapping.longitude)
        lat_col = explicit(mapping.latitude)
        if lon_col is None:
            lon_col, rejected["longitude"] = _pick_coordinate(
                df, _candidates(cols, LON_NAMES, ("longitude", "long", "lon")), 180.0
            )
        if lat_col is None:
            lat_col, rejected["latitude"] = _pick_coordinate(
                df, _candidates(cols, LAT_NAMES, ("latitude", "lat")), 90.0
            )
        if lon_col is None or lat_col is None:
            raise UploadError(
                "No decimal-degree longitude/latitude columns found"
                + (
                    f" (rejected, values not in degrees: {sum(rejected.values(), [])})"
                    if any(rejected.values())
                    else ""
                )
                + "; name them explicitly"
            )
        used["longitude"], used["latitude"] = lon_col, lat_col
        lon, lat = _numeric(df[lon_col]), _numeric(df[lat_col])

    def detect(
        field_name: str,
        explicit_name: str | None,
        names: tuple[str, ...],
        contains: tuple[str, ...],
    ) -> str | None:
        col = explicit(explicit_name)
        if col is None:
            cands = _candidates(cols, names, contains)
            col = cands[0] if cands else None
        used[field_name] = col
        return col

    name_col = detect("species", mapping.species, NAME_NAMES, ("scientific", "taxon", "species"))
    date_col = detect("date", mapping.date, DATE_NAMES, ("date",))
    unc_col = detect(
        "uncertainty", mapping.uncertainty, UNCERTAINTY_NAMES, ("uncertainty", "accuracy")
    )
    status_col = detect("status", mapping.status, STATUS_NAMES, ("occurrencestatus", "presence"))
    id_col = detect("record_id", mapping.record_id, ID_NAMES, ())

    skipped: dict[str, int] = {}
    keep = pd.Series(True, index=df.index)

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal keep
        newly = mask & keep
        if newly.any():
            skipped[reason] = int(newly.sum())
        keep &= ~mask

    names = [" ".join(n.lower().split()) for n in species_names if n]
    if name_col is not None and names:
        drop(~df[name_col].map(lambda v: _species_match(v, names)), "other species")
    drop(lon.isna() | lat.isna(), "missing coordinates")
    drop((lon.abs() > 180) | (lat.abs() > 90), "coordinates out of range")
    drop((lon == 0) & (lat == 0), "zero coordinates")
    statuses = (
        df[status_col].map(_status) if status_col is not None else pd.Series("present", df.index)
    )
    drop(statuses.isna(), "unrecognised presence/absence value")

    if date_col is None:
        dates = pd.Series(pd.NaT, index=df.index)
    else:
        raw_dates = df[date_col]
        if pd.api.types.is_numeric_dtype(raw_dates):
            # A numeric column from a spatial format is a year; to_datetime would read
            # 2015 as nanoseconds since 1970.
            raw_dates = raw_dates.map(lambda v: None if pd.isna(v) else str(int(v)))
        dates = pd.to_datetime(raw_dates, errors="coerce", format="mixed")
    unc = _numeric(df[unc_col]) if unc_col is not None else pd.Series(np.nan, index=df.index)
    if id_col is not None:
        ids = pd.to_numeric(df[id_col], errors="coerce")
        if ids.isna().any() or ids.duplicated().any() or (ids % 1 != 0).any():
            used["record_id"] = None  # not a usable integer key: fall back to row numbers
            id_col = None
    record_ids = ids.astype("int64") if id_col is not None else pd.Series(df.index + 1, df.index)

    records = [
        ReferenceRecord(
            taxon_key=taxon_key,
            dataset_id=dataset_id,
            record_id=int(record_ids[i]),
            longitude=float(lon[i]),
            latitude=float(lat[i]),
            occurrence_status=statuses[i],
            event_date=None if pd.isna(dates[i]) else dates[i].date(),
            coordinate_uncertainty_m=None if pd.isna(unc[i]) else float(unc[i]),
        )
        for i in df.index[keep]
    ]
    if not records:
        raise UploadError(
            "No usable rows"
            + (
                f" (skipped: {', '.join(f'{v} {k}' for k, v in skipped.items())})"
                if skipped
                else ""
            )
            + (f"; species column '{name_col}' did not match {species_names}" if name_col else "")
        )
    report = {
        **info,
        "file_name": filename,
        "n_rows": len(df),
        "n_records": len(records),
        "skipped": skipped,
        "columns": used,
        "rejected_coordinate_columns": {k: v for k, v in rejected.items() if v},
        "species_filter": names if name_col else None,
    }
    return ParsedUpload(
        records=records, file_sha256=hashlib.sha256(content).hexdigest(), report=report
    )
