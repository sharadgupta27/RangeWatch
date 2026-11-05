"""API contract of the priority-2/3 endpoints: roles, annotations, study region, alerts,
exports (COG / GeoPackage / ODMAP / STAC / citation), map products and the benchmark."""

from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from src.api import deps
from src.api.main import create_app
from src.domain import Alert
from tests.conftest import TAXON_KEY, FakeGbif, FakeInat
from tests.test_api import RecordingQueue, trained  # noqa: F401


class BenchmarkQueue(RecordingQueue):
    def enqueue_benchmark(self, job_id: str, request: dict) -> None:
        self.calls.append(("benchmark", job_id, request))


def make_client(trained, **settings_update):  # noqa: F811
    settings, repo, artifacts = trained
    s = settings.model_copy(update=settings_update)
    app = create_app()
    queue = BenchmarkQueue()
    app.dependency_overrides[deps.repository_dep] = lambda: repo
    app.dependency_overrides[deps.queue_dep] = lambda: queue
    app.dependency_overrides[deps.settings_dep] = lambda: s
    app.dependency_overrides[deps.artifacts_dep] = lambda: artifacts
    app.dependency_overrides[deps.gbif_search_dep] = lambda: FakeGbif(initial=[])
    app.dependency_overrides[deps.inat_search_dep] = lambda: FakeInat()
    c = TestClient(app)
    c.queue = queue  # type: ignore[attr-defined]
    c.repo = repo  # type: ignore[attr-defined]
    return c


@pytest.fixture
def client(trained):  # noqa: F811
    return make_client(trained)


def as_role(groups: str) -> dict[str, str]:
    return {"X-Auth-Request-User": "ana", "X-Auth-Request-Groups": groups}


# ------------------------------------------------------------------ roles (U14)
def test_header_auth_maps_groups_to_roles_and_enforces_them(trained):  # noqa: F811
    c = make_client(trained, auth_mode="header")
    assert c.get("/me").status_code == 401
    viewer = c.get("/me", headers=as_role("staff")).json()
    assert viewer["role"] == "viewer" and viewer["permissions"]["run"] is False
    assert c.post(f"/species/{TAXON_KEY}/runs", headers=as_role("staff")).status_code == 403
    analyst = as_role("staff,rangewatch-analysts")
    assert c.post(f"/species/{TAXON_KEY}/runs", headers=analyst).status_code == 202
    # confirming a native range is a curator's review step
    nr = c.get(f"/species/{TAXON_KEY}/native-range").json()
    body = {"geometry": nr["geometry"], "confirm": True}
    r = c.put(f"/species/{TAXON_KEY}/native-range", json=body, headers=analyst)
    assert r.status_code == 403 and "curator" in r.json()["detail"]
    ann = {"source": "GBIF", "record_id": 1, "kind": "exclude", "value": "true"}
    assert c.post(f"/species/{TAXON_KEY}/annotations", json=ann, headers=analyst).status_code == 403
    assert c.get(f"/species/{TAXON_KEY}", headers=as_role("")).status_code == 200  # reading is open


def test_without_auth_everyone_is_a_local_curator(client):
    me = client.get("/me").json()
    assert me == {
        "user": "local",
        "role": "curator",
        "auth_mode": "none",
        "permissions": {"view": True, "run": True, "curate": True},
    }


# ------------------------------------------------------------------ annotations (U11)
def test_annotations_are_append_only_and_flag_a_retrain(client):
    url = f"/species/{TAXON_KEY}/annotations"
    bad = client.post(url, json={"source": "GBIF", "record_id": 1, "kind": "exclude", "value": "x"})
    assert bad.status_code == 422
    missing = {"source": "GBIF", "record_id": 999_999, "kind": "exclude", "value": "true"}
    assert client.post(url, json=missing).status_code == 404
    first = client.post(
        url, json={"source": "GBIF", "record_id": 2, "kind": "cultivated", "value": "TRUE"}
    )
    assert first.status_code == 201
    assert first.json()["retrain_flagged"] is True
    assert first.json()["annotation"]["value"] == "true"
    client.post(
        url, json={"source": "GBIF", "record_id": 2, "kind": "cultivated", "value": "false"}
    )
    history = client.get(url, params={"record_id": 2}).json()
    assert [a["value"] for a in history] == ["false", "true"]  # newest first, nothing edited
    assert client.get(f"/species/{TAXON_KEY}").json()["status"] == "retrain_pending"
    rows = client.get(f"/species/{TAXON_KEY}/occurrences", params={"limit": 500}).json()["items"]
    row = next(r for r in rows if r["source"] == "GBIF" and r["record_id"] == 2)
    assert row["annotation"]["cultivated"] is False and row["annotation"]["n_annotations"] == 2


# ------------------------------------------------------------------ study region (U2)
def test_study_region_is_validated_and_flags_a_retrain(client):
    url = f"/species/{TAXON_KEY}/study-region"
    assert client.put(url, json={"bbox": [10, 50, 0, 40]}).status_code == 422
    r = client.put(url, json={"bbox": [10, 50, 0, 40], "resolution": "30s"})
    assert r.status_code == 422
    ok = client.put(url, json={"bbox": [-20, 25, 50, 65]})
    assert ok.status_code == 200
    assert ok.json()["study_region"] == {"bbox": [-20, 25, 50, 65], "resolution": None}
    assert ok.json()["status"] == "retrain_pending"
    cleared = client.put(url, json={"bbox": None}).json()
    assert cleared["study_region"] is None


# ------------------------------------------------------------------ alerts (U10)
def test_alert_listing_feed_and_subscriptions(client):
    client.repo.add_alert(
        Alert(
            TAXON_KEY,
            "outside_known_range",
            1,
            [
                {
                    "source": "GBIF",
                    "record_id": 5,
                    "longitude": 1.0,
                    "latitude": 2.0,
                    "effort_class": "discovery",
                }
            ],
            "1 new record far from every known record",
            model_version=1,
            effort_classes={"expansion": 0, "discovery": 1, "unknown": 0},
        )
    )
    alerts = client.get(f"/species/{TAXON_KEY}/alerts").json()
    assert alerts[0]["effort_classes"]["discovery"] == 1
    feed = client.get("/alerts/feed.xml")
    assert feed.headers["content-type"].startswith("application/rss+xml")
    assert "<rss" in feed.text and "Testus invasivus" in feed.text
    subs = "/alerts/subscriptions"
    for target in ("http://example.org/hook", "https://127.0.0.1/hook", "https://u:p@x.org/h"):
        r = client.post(subs, json={"channel": "webhook", "target": target})
        assert r.status_code == 422, target
    assert client.post(subs, json={"channel": "email", "target": "nope"}).status_code == 422
    made = client.post(
        subs, json={"channel": "email", "target": "team@example.org", "taxon_key": TAXON_KEY}
    ).json()
    assert made["kinds"] == ["outside_envelope", "outside_known_range"]
    assert [s["target"] for s in client.get(subs, params={"taxon_key": TAXON_KEY}).json()] == [
        "team@example.org"
    ]
    assert client.delete(f"{subs}/{made['subscription_id']}").status_code == 204
    assert client.delete(f"{subs}/{made['subscription_id']}").status_code == 404


# ------------------------------------------------------------------ exports (U8)
def test_export_index_and_downloads(client):
    base = f"/species/{TAXON_KEY}/models/1/exports"
    items = {i["id"]: i for i in client.get(base).json()["items"]}
    assert {"suitability", "mess", "zones", "limiting", "binary:p10", "occurrences"} <= set(items)
    tif = client.get(items["suitability"]["url"])
    assert tif.status_code == 200 and tif.content[:2] in (b"II", b"MM")
    binary = client.get(items["binary:max_tss"]["url"])
    assert binary.status_code == 200
    assert client.get(f"{base}/raster/binary:nope").status_code == 404
    gpkg = client.get(f"{base}/occurrences.gpkg")
    assert gpkg.status_code == 200 and gpkg.content[:15] == b"SQLite format 3"
    odmap = client.get(f"{base}/odmap").json()
    assert [s["section"] for s in odmap["sections"]] == [
        "Overview",
        "Data",
        "Model",
        "Assessment",
        "Prediction",
    ]
    md = client.get(f"{base}/odmap", params={"format": "md"}).text
    assert md.startswith("# ODMAP protocol") and "transferability" in md.lower()
    stac = client.get(f"{base}/stac").json()
    assert stac["stac_version"] == "1.0.0" and "suitability" in stac["assets"]
    assert stac["properties"]["rangewatch:threshold_rule"]
    cite = client.get(f"{base}/citation").text
    assert "GBIF.org" in cite and "elapid" in cite
    z = zipfile.ZipFile(io.BytesIO(client.get(f"{base}/bundle.zip").content))
    names = set(z.namelist())
    assert {"odmap.json", "stac.json", "CITATION.txt", "occurrences.gpkg"} <= names
    assert "rasters/suitability.tif" in names
    assert client.get(f"/species/{TAXON_KEY}/models/99/exports").status_code == 404


# ------------------------------------------------------------------ layers / detail
def test_layers_offer_binary_maps_and_products(client):
    layers = client.get(f"/species/{TAXON_KEY}/layers", params={"threshold": 0.42}).json()
    ids = [p["id"] for p in layers["products"]]
    assert "binary:p10" in ids and "binary:custom" in ids and "limiting" in ids
    assert "priority" in ids
    lim = next(p for p in layers["products"] if p["id"] == "limiting")
    assert any("bio1" in e["label"] for e in lim["legend"])
    detail = client.get(f"/species/{TAXON_KEY}/models/1").json()
    assert set(detail["thresholds"]) >= {"p10", "max_tss"}
    assert detail["response_curves"]["curves"][0]["predictor"] == "bio1"
    assert detail["projection"]["limiting_factor"]["land"]
    assert detail["algorithm"] == "maxent" and detail["survey_priority"]["status"] == "available"


# ------------------------------------------------------------------ benchmark (§4.6)
def test_benchmark_is_enqueued_and_validated(client):
    assert client.get("/validation/benchmark").status_code == 404
    r = client.post("/validation/benchmark", json={"bias": "gbif_effort"})
    assert r.status_code == 422
    job = client.post("/validation/benchmark", json={"n_presences": 200})
    assert job.status_code == 202 and job.json()["kind"] == "benchmark"
    kind, _, request = client.queue.calls[-1]
    assert kind == "benchmark" and request["variants"][0] == "no_correction"
