"""API contract tests (FastAPI TestClient; repository/queue/search dependencies overridden)."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from src.api import deps
from src.api.main import create_app
from src.config import Settings
from src.orchestration.pipeline import SpeciesPipeline
from src.orchestration.registry_ops import update_native_range
from src.persistence.artifact_store import ArtifactStore
from src.persistence.model_registry import NullModelRegistry
from src.persistence.repository import InMemoryRepository
from tests.conftest import (
    BIOCLIM_VERSION,
    FAST_TRAINING,
    TAXON_KEY,
    FakeGbif,
    FakeInat,
    box_records,
    native_box_geojson,
)


class RecordingQueue:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def enqueue_pipeline(
        self, job_id: str, taxon_key: int, force_retrain: bool, full_resync: bool = False
    ) -> None:
        self.calls.append(("pipeline", job_id, taxon_key, force_retrain))
        self.last_full_resync = full_resync

    def enqueue_bulletin(self, job_id: str, taxon_key: int, model_version: int | None) -> None:
        self.calls.append(("bulletin", job_id, taxon_key, model_version))

    def enqueue_scenarios(self, job_id: str, taxon_key: int) -> None:
        self.calls.append(("scenarios", job_id, taxon_key))

    def enqueue_validation(self, job_id: str, taxon_keys, train: bool) -> None:
        self.calls.append(("validation", job_id, taxon_keys, train))


@pytest.fixture(scope="module")
def trained(tmp_path_factory, data_root):
    """One trained species shared by the module (training is the slow part)."""
    tmp = tmp_path_factory.mktemp("api")
    settings = Settings(
        data_root=data_root,
        artifact_root=tmp / "artifacts",
        bioclim_version=BIOCLIM_VERSION,
        mlflow_tracking_uri=None,
    )
    recs = box_records(300, (0, 40), (35, 55), 1, seed=1) + box_records(
        60, (-100, -80), (35, 50), 10_000, label="introduced", seed=2
    )
    repo = InMemoryRepository()
    artifacts = ArtifactStore(settings.artifact_root, "/data/artifacts")
    p = SpeciesPipeline(
        repo=repo,
        gbif=FakeGbif(initial=recs),
        inat=FakeInat(),
        artifacts=artifacts,
        registry=NullModelRegistry(),
        settings=settings,
        training_config=FAST_TRAINING,
    )
    p.run(TAXON_KEY)
    update_native_range(repo, TAXON_KEY, native_box_geojson(), confirm=True)
    p.run(TAXON_KEY)
    return settings, repo, artifacts


@pytest.fixture
def client(trained):
    settings, repo, artifacts = trained
    app = create_app()
    queue = RecordingQueue()
    app.dependency_overrides[deps.repository_dep] = lambda: repo
    app.dependency_overrides[deps.queue_dep] = lambda: queue
    app.dependency_overrides[deps.settings_dep] = lambda: settings
    app.dependency_overrides[deps.artifacts_dep] = lambda: artifacts
    app.dependency_overrides[deps.gbif_search_dep] = lambda: FakeGbif(initial=[])
    app.dependency_overrides[deps.inat_search_dep] = lambda: FakeInat()
    c = TestClient(app)
    c.queue = queue  # type: ignore[attr-defined]
    return c


def test_health_and_bioclim(client):
    assert client.get("/health").json()["bioclim_available"] is True
    info = client.get("/bioclim").json()
    assert info["selected_predictors"] == ["bio1", "bio4", "bio12"]
    assert info["descriptions"]["bio1"] == "Annual mean temperature"


def test_search_merges_gbif_and_inat(client):
    res = client.get("/species/search", params={"q": "testus"}).json()
    assert res[0]["taxon_key"] == TAXON_KEY
    assert res[0]["common_name"] == "Test weed"
    assert res[0]["thumbnail_url"].startswith("https://")
    assert res[0]["in_registry"] is True
    assert res[0]["breadcrumb"][0] == "Plantae"


def test_species_detail_includes_caveat_and_severity(client):
    d = client.get(f"/species/{TAXON_KEY}").json()
    assert d["status"] == "up_to_date" and d["model_version"] == 1
    assert d["native_range_status"] == "confirmed"
    src = d["sources"]
    assert src["gbif"] + src["inaturalist_direct"] == d["n_occurrences_total"]
    assert 0 <= src["gbif_from_inaturalist"] <= src["gbif"]
    cm = d["current_model"]
    assert cm["model_type"] == "B"
    assert (
        "transferability" in cm["transferability_caveat"].lower()
        or "0.7" in cm["transferability_caveat"]
    )
    assert set(cm["severity"]["components"]) == {
        "suitability",
        "climate_analogy",
        "spread_rate",
        "ecological_impact_prior",
    }
    assert d["severity_config"]["weights"]["suitability"] == 0.35
    assert client.get("/species").json()[0]["severity_score"] is not None


def test_unknown_species_404(client):
    assert client.get("/species/1").status_code == 404
    assert client.get("/species/1/models").status_code == 404


def test_run_enqueues_job(client):
    r = client.post(f"/species/{TAXON_KEY}/runs", json={"force_retrain": True})
    assert r.status_code == 202
    job = r.json()
    assert job["status"] == "queued" and job["kind"] == "pipeline"
    assert client.queue.calls[-1] == ("pipeline", job["job_id"], TAXON_KEY, True)
    assert client.get(f"/jobs/{job['job_id']}").json()["job_id"] == job["job_id"]
    assert client.get(f"/jobs/{uuid.uuid4()}").status_code == 404


def test_native_range_roundtrip_and_train_requires_confirm(client):
    nr = client.get(f"/species/{TAXON_KEY}/native-range").json()
    assert nr["status"] == "confirmed" and nr["geometry"]["type"] == "MultiPolygon"
    r = client.put(
        f"/species/{TAXON_KEY}/native-range",
        json={"geometry": native_box_geojson(), "confirm": False, "train": True},
    )
    assert r.status_code == 409
    # The rejected request changed nothing: the range is still confirmed.
    assert client.get(f"/species/{TAXON_KEY}/native-range").json()["status"] == "confirmed"
    r = client.put(
        f"/species/{TAXON_KEY}/native-range",
        json={"geometry": native_box_geojson(), "confirm": True, "train": True},
    )
    assert r.status_code == 200 and r.json()["job"]["kind"] == "pipeline"
    bad = {"type": "Polygon", "coordinates": [[[0, 0], [400, 0], [400, 1], [0, 0]]]}
    assert (
        client.put(f"/species/{TAXON_KEY}/native-range", json={"geometry": bad}).status_code == 422
    )


def test_occurrence_page_and_timeline(client):
    page = client.get(
        f"/species/{TAXON_KEY}/occurrences", params={"limit": 10, "range_label": "introduced"}
    ).json()
    assert page["total"] == 60 and len(page["items"]) == 10
    assert all(i["range_label"] == "introduced" for i in page["items"])
    tl = client.get(f"/species/{TAXON_KEY}/occurrences/timeline").json()
    assert tl["min_year"] == 2015 and tl["years"][-1]["cumulative"] == 360


def test_model_lineage_exposes_reproducibility_verbatim(client):
    rows = client.get(f"/species/{TAXON_KEY}/models").json()
    assert rows[0]["model_version"] == 1
    repro = rows[0]["reproducibility"]
    for key in ("bioclim_version_used", "gbif_citation", "package_versions", "random_seeds"):
        assert repro[key]
    detail = client.get(f"/species/{TAXON_KEY}/models/1").json()
    assert detail["projection"]["candidate_area_km2"] >= 0
    assert set(detail["variable_importance"]) == {"bio1", "bio4", "bio12"}
    assert "spatial block" in detail["cv_method"]


def test_layers_always_pair_suitability_with_mess(client):
    layers = client.get(f"/species/{TAXON_KEY}/layers").json()
    assert layers["suitability"] and layers["mess"]
    assert layers["extrapolation"]["legend"][0]["value"] == 1
    assert "/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=" in layers["suitability"]["tile_url"]
    assert "%2Fdata%2Fartifacts%2Fspecies" in layers["suitability"]["tile_url"]
    assert "public.occurrence_tiles/{z}/{x}/{y}.pbf?taxon_key=" in layers["occurrences"]["tile_url"]
    assert layers["caveats"]["mess_required"] is True
    assert [e["value"] for e in layers["zones"]["legend"]] == [1, 2, 3]


def test_severity_config_update(client):
    body = {
        "weights": {
            "suitability": 0,
            "climate_analogy": 0,
            "spread_rate": 0,
            "ecological_impact_prior": 1,
        },
        "impact_prior": 0.8,
    }
    r = client.put(f"/species/{TAXON_KEY}/severity-config", json=body)
    assert r.status_code == 200 and r.json()["score"] == pytest.approx(0.8)
    body["weights"]["ecological_impact_prior"] = 0
    assert client.put(f"/species/{TAXON_KEY}/severity-config", json=body).status_code == 422


def test_bulletin_status_and_enqueue(client):
    st = client.get(f"/species/{TAXON_KEY}/bulletin").json()
    assert st["model_version"] == 1 and st["pdf_available"] is False
    r = client.post(f"/species/{TAXON_KEY}/bulletin")
    assert r.status_code == 202 and client.queue.calls[-1][0] == "bulletin"
    assert client.get(f"/species/{TAXON_KEY}/bulletin/file").status_code == 404


def test_openapi_schema_has_operation_ids(client):
    spec = client.get("/openapi.json").json()
    ops = {op["operationId"] for path in spec["paths"].values() for op in path.values()}
    assert {
        "searchSpecies",
        "getSpecies",
        "runSpeciesPipeline",
        "updateNativeRange",
        "listModelVersions",
        "getLayers",
        "generateBulletin",
        "getJob",
    } <= ops


def test_scenario_layers_pair_suitability_with_their_own_extrapolation(client, trained):
    _, repo, _ = trained
    mv = repo.get_model_version(TAXON_KEY, 1)
    arts = {
        **mv.artifacts,
        "scenarios": {
            "ssp245_2050": {
                "suitability": "species/x/v1/suitability_ssp245_2050.tif",
                "extrapolation": "species/x/v1/extrapolation_ssp245_2050.tif",
                "extrapolated_land_fraction": 0.3,
            },
            "legacy": "species/x/v1/suitability_legacy.tif",
        },
    }
    repo.update_model_artifacts(TAXON_KEY, 1, arts)
    try:
        sc = {s["name"]: s for s in client.get(f"/species/{TAXON_KEY}/layers").json()["scenarios"]}
        assert "extrapolation_ssp245_2050.tif" in sc["ssp245_2050"]["extrapolation"]["tile_url"]
        assert sc["ssp245_2050"]["extrapolated_land_fraction"] == 0.3
        assert sc["legacy"]["extrapolation"] is None
    finally:
        repo.update_model_artifacts(TAXON_KEY, 1, mv.artifacts)
    r = client.post(f"/species/{TAXON_KEY}/scenarios")
    assert r.status_code == 202 and client.queue.calls[-1][0] == "scenarios"


def test_validation_endpoints(client, trained):
    assert client.get("/validation").status_code == 404
    r = client.post("/validation/runs", json={"taxon_keys": [1316908], "train": False})
    assert r.status_code == 202
    assert client.queue.calls[-1][0] == "validation" and client.queue.calls[-1][2] == [1316908]


def test_full_resync_flag_is_forwarded(client):
    r = client.post(f"/species/{TAXON_KEY}/runs", json={"full_resync": True})
    assert r.status_code == 202 and client.queue.last_full_resync is True
