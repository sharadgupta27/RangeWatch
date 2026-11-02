"""Upload parsing (connectors/upload_client.py): column detection, species filter, CRS."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date

import geopandas as gpd
import pytest
from shapely.geometry import Point

from src.connectors.upload_client import ColumnMapping, UploadError, parse_upload

NAMES = ["Acacia saligna", "Acacia saligna (Labill.) H.L.Wendl."]

# Same layout as the Israel Nature and Parks Authority export: projected ITM columns come
# first and must be rejected as coordinates; a free-text column contains "lat" (relation).
INPA_CSV = """ID,Taxon (scientific),Date Event,ITM - LONG,ITM - LAT,WGS84 - LONG,WGS84 - LAT,\
Type of observation,Accuracy in meters (automatic),The location of the report in relation \
to the event
1105738,Acacia saligna,2016-12-08,192224.7513,716897.9527,34.91452842,32.54595434,Herd \
observation,,
1113392,Acacia saligna,2017-01-04,174171.1811,643757.3120,34.72579868,31.88582089,Herd \
observation,25,
2000001,Heterotheca subaxillaris,2018-03-01,174181.1655,643754.0300,34.7259,31.8858,,,
1145659,Acacia saligna,,188627.4932,699270.2049,,,,,
"""


def parse(content: bytes | str, name: str = "inpa.csv", **kw):
    data = content.encode() if isinstance(content, str) else content
    return parse_upload(data, name, 3190653, "upload-test", NAMES, **kw)


def test_agency_csv_columns_are_detected_and_other_species_dropped():
    out = parse(INPA_CSV)
    cols = out.report["columns"]
    assert cols["longitude"] == "WGS84 - LONG" and cols["latitude"] == "WGS84 - LAT"
    assert out.report["rejected_coordinate_columns"] == {
        "longitude": ["ITM - LONG"],
        "latitude": ["ITM - LAT"],
    }
    assert cols["species"] == "Taxon (scientific)" and cols["date"] == "Date Event"
    assert cols["uncertainty"] == "Accuracy in meters (automatic)"
    assert out.report["skipped"] == {"other species": 1, "missing coordinates": 1}
    assert [r.record_id for r in out.records] == [1105738, 1113392]
    first, second = out.records
    assert first.event_date == date(2016, 12, 8) and first.occurrence_status == "present"
    assert first.coordinate_uncertainty_m is None and second.coordinate_uncertainty_m == 25.0
    assert out.n_present == 2 and out.n_absent == 0
    assert len(out.file_sha256) == 64


def test_presence_absence_column_and_row_number_ids():
    csv = (
        "decimalLongitude;decimalLatitude;occurrenceStatus;scientificName\n"
        "34,8;32,1;present;Acacia saligna\n"
        "34.9;32.2;absent;Acacia saligna\n"
        "35.0;32.3;0;Acacia saligna\n"
        "35.1;32.4;maybe;Acacia saligna\n"
    )
    out = parse(csv)
    assert out.report["delimiter"] == ";"
    assert [r.occurrence_status for r in out.records] == ["present", "absent", "absent"]
    assert out.records[0].longitude == pytest.approx(34.8)  # decimal comma accepted
    assert [r.record_id for r in out.records] == [1, 2, 3]
    assert out.report["skipped"] == {"unrecognised presence/absence value": 1}


def test_explicit_mapping_overrides_detection_and_unknown_columns_fail():
    csv = "a,b,c\n34.9,32.5,x\n"
    with pytest.raises(UploadError, match="No decimal-degree"):
        parse(csv)
    out = parse(csv, mapping=ColumnMapping(longitude="a", latitude="b"))
    assert len(out.records) == 1 and out.report["columns"]["species"] is None
    with pytest.raises(UploadError, match="Column 'nope' not found"):
        parse(csv, mapping=ColumnMapping(longitude="nope", latitude="b"))


def test_spatial_upload_is_reprojected_from_its_crs(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"species": ["Acacia saligna"], "date": ["2019-04-02"]},
        geometry=[Point(34.9, 32.5)],
        crs="EPSG:4326",
    ).to_crs(
        "EPSG:2039"
    )  # Israeli TM Grid
    path = tmp_path / "survey.gpkg"
    gdf.to_file(path, driver="GPKG")
    out = parse(path.read_bytes(), "survey.gpkg")
    rec = out.records[0]
    assert rec.longitude == pytest.approx(34.9, abs=1e-6)
    assert rec.latitude == pytest.approx(32.5, abs=1e-6)
    assert out.report["reprojected_to"] == "EPSG:4326"
    assert out.report["columns"]["longitude"] == "geometry"


def test_zipped_shapefile_and_geojson(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"taxon": ["Acacia saligna", "Acacia saligna"]},
        geometry=[Point(34.9, 32.5), Point(35.0, 32.6)],
        crs="EPSG:4326",
    )
    gdf.to_file(tmp_path / "pts.shp")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for f in tmp_path.glob("pts.*"):
            zf.write(f, f.name)
    assert len(parse(buf.getvalue(), "pts.zip").records) == 2
    geojson = json.loads(gdf.to_json())
    assert len(parse(json.dumps(geojson), "pts.geojson").records) == 2


def test_numeric_year_column_from_spatial_format_is_a_year(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"species": ["Acacia saligna"], "year": [2015]},
        geometry=[Point(34.9, 32.5)],
        crs="EPSG:4326",
    )
    path = tmp_path / "survey.gpkg"
    gdf.to_file(path, driver="GPKG")
    out = parse(path.read_bytes(), "survey.gpkg")
    assert out.records[0].event_date == date(2015, 1, 1)


def test_zip_expanding_beyond_the_limit_is_rejected(monkeypatch):
    monkeypatch.setattr("src.connectors.upload_client.MAX_UNZIPPED_BYTES", 10)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("pts.shp", b"\0" * 1000)
    with pytest.raises(UploadError, match="expands beyond"):
        parse(buf.getvalue(), "pts.zip")


def test_rejections_are_explained():
    with pytest.raises(UploadError, match="Unsupported file type"):
        parse("x", "data.xlsx")
    with pytest.raises(UploadError, match="did not match"):
        parse("lon,lat,species\n34.9,32.5,Pinus halepensis\n")
    with pytest.raises(UploadError, match="exceed the limit"):
        parse("lon,lat\n34.9,32.5\n35.0,32.6\n", max_rows=1)
