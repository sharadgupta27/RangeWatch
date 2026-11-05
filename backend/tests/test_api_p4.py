"""API contract of species lists / batch runs (U9), model cards (U13) and the bias-study
jobs (B1 target-group comparison, B9 shared bias)."""

from __future__ import annotations

import io
import zipfile

import pytest

from tests.conftest import TAXON_KEY
from tests.test_api import trained  # noqa: F401
from tests.test_api_p23 import BenchmarkQueue, as_role, make_client


class P4Queue(BenchmarkQueue):
    def enqueue_batch(self, job_id: str, taxon_keys: list[int], force_retrain: bool) -> None:
        self.calls.append(("batch", job_id, taxon_keys, force_retrain))

    def enqueue_target_groups(self, job_id: str, taxon_key: int) -> None:
        self.calls.append(("target_groups", job_id, taxon_key))

    def enqueue_shared_bias(self, job_id: str, taxon_key: int) -> None:
        self.calls.append(("shared_bias", job_id, taxon_key))


@pytest.fixture
def client(trained):  # noqa: F811
    c = make_client(trained)
    queue = P4Queue()
    from src.api import deps

    c.app.dependency_overrides[deps.queue_dep] = lambda: queue
    c.queue = queue  # type: ignore[attr-defined]
    return c


# ------------------------------------------------------------------ species lists (U9)
def test_species_list_resolves_names_filters_registry_and_runs_a_batch(client):
    body = {
        "name": "Watch list",
        "source": "https://example.org/list",
        "names": ["Testus invasivus", "testus  invasivs", "Nonexistentia fictiva"],
        "taxon_keys": [TAXON_KEY],
    }
    r = client.post("/lists", json=body)
    assert r.status_code == 201, r.text
    sl = r.json()
    by_query = {e["query"]: e for e in sl["entries"]}
    assert by_query["Testus invasivus"]["match_type"] == "EXACT"
    assert by_query["testus invasivs"]["match_type"] == "FUZZY"  # whitespace normalised
    assert by_query["Nonexistentia fictiva"]["taxon_key"] is None
    assert by_query[str(TAXON_KEY)]["match_type"] == "KEY"
    assert (sl["n_species"], sl["n_unresolved"], sl["n_registered"], sl["n_trained"]) == (
        1,
        1,
        1,
        1,
    )
    lid = sl["list_id"]
    assert [s["taxon_key"] for s in client.get("/species", params={"list_id": lid}).json()] == [
        TAXON_KEY
    ]
    assert client.get("/lists").json()[0]["list_id"] == lid
    run = client.post(f"/lists/{lid}/runs", json={"force_retrain": True})
    assert run.status_code == 202 and run.json()["kind"] == "batch"
    assert client.queue.calls[-1] == ("batch", run.json()["job_id"], [TAXON_KEY], True)

    upd = client.put(f"/lists/{lid}", json={"name": "Renamed", "names": ["Nonexistentia x"]})
    assert upd.status_code == 200 and upd.json()["n_species"] == 0
    assert client.post(f"/lists/{lid}/runs").status_code == 422  # nothing resolved to run
    assert client.delete(f"/lists/{lid}").status_code == 204
    assert client.get(f"/lists/{lid}").status_code == 404
    assert client.get("/species", params={"list_id": lid}).status_code == 404


def test_species_list_validation_and_roles(trained):  # noqa: F811
    c = make_client(trained, list_max_names=2)
    assert c.post("/lists", json={"name": "Empty"}).status_code == 422
    too_many = {"name": "Big", "names": ["a b", "c d", "e f"]}
    assert c.post("/lists", json=too_many).status_code == 422
    h = make_client(trained, auth_mode="header")
    body = {"name": "x", "taxon_keys": [TAXON_KEY]}
    assert h.post("/lists", json=body, headers=as_role("staff")).status_code == 403
    ok = h.post("/lists", json=body, headers=as_role("rangewatch-analysts"))
    assert ok.status_code == 201 and ok.json()["created_by"] == "ana"


# ------------------------------------------------------------------ model card (U13)
def test_model_card_json_markdown_and_exports(client):
    base = f"/species/{TAXON_KEY}/models/1/exports"
    card = client.get(f"{base}/model-card").json()
    assert card["caveats"][0] == card["transferability_caveat"]
    assert card["confidence_label"] in ("standard", "lower")
    assert {s["id"] for s in card["sections"]} >= {"what", "quality", "extrapolation", "data"}
    md = client.get(f"{base}/model-card", params={"format": "md"})
    assert md.headers["content-type"].startswith("text/markdown")
    assert md.text.startswith("# Model card")
    ids = [i["id"] for i in client.get(base).json()["items"]]
    assert "model_card" in ids
    z = zipfile.ZipFile(io.BytesIO(client.get(f"{base}/bundle.zip").content))
    assert "MODEL_CARD.md" in z.namelist()
    assert client.get(f"/species/{TAXON_KEY}/models/99/exports/model-card").status_code == 404


# ------------------------------------------------------------------ bias studies (B1, B9)
def test_bias_study_jobs_and_reports(client):
    base = f"/species/{TAXON_KEY}"
    assert client.get(f"{base}/target-group-comparison").status_code == 404
    r = client.post(f"{base}/target-group-comparison")
    assert r.status_code == 202 and r.json()["kind"] == "target_groups"
    assert client.queue.calls[-1] == ("target_groups", r.json()["job_id"], TAXON_KEY)
    assert client.get(f"{base}/shared-bias").status_code == 404
    r = client.post(f"{base}/shared-bias")
    assert r.status_code == 202 and r.json()["kind"] == "shared_bias"
    assert client.post("/species/123/shared-bias").status_code == 404
