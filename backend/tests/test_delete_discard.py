"""Deleting a species that is not needed, and discarding a queued or running job."""

from __future__ import annotations

import uuid

import pytest

from src.api import deps
from src.domain import Job, JobKind, JobStatus, RunDiscarded, SpeciesStatus
from src.orchestration import factory, tasks
from src.persistence.repository import InMemoryRepository, NotFoundError
from tests.conftest import TAXON_KEY, FakeGbif
from tests.test_api import RecordingQueue
from tests.test_pipeline import confirm_native, initial_records, make_pipeline


@pytest.fixture
def client(settings):
    """A freshly trained species per test (these tests delete it)."""
    from fastapi.testclient import TestClient

    from src.api.main import create_app
    from src.persistence.artifact_store import ArtifactStore

    repo = InMemoryRepository()
    p = make_pipeline(settings, FakeGbif(initial=initial_records()), repo=repo)
    p.run(TAXON_KEY)
    confirm_native(p)
    p.run(TAXON_KEY)
    artifacts: ArtifactStore = p.artifacts
    app = create_app()
    queue = RecordingQueue()
    app.dependency_overrides[deps.repository_dep] = lambda: repo
    app.dependency_overrides[deps.queue_dep] = lambda: queue
    app.dependency_overrides[deps.settings_dep] = lambda: settings
    app.dependency_overrides[deps.artifacts_dep] = lambda: artifacts
    c = TestClient(app)
    c.queue = queue  # type: ignore[attr-defined]
    return c


# --------------------------------------------------------------------------- delete
def test_delete_species_removes_everything(client):
    repo = client.app.dependency_overrides[deps.repository_dep]()
    artifacts = client.app.dependency_overrides[deps.artifacts_dep]()
    folder = artifacts.local_path(str(artifacts.species_dir(TAXON_KEY)))
    assert folder.exists() and repo.list_model_versions(TAXON_KEY)

    r = client.delete(f"/species/{TAXON_KEY}", params={"confirm": 1})
    assert r.status_code == 422  # must repeat the taxon key
    assert client.get(f"/species/{TAXON_KEY}").status_code == 200

    r = client.delete(f"/species/{TAXON_KEY}", params={"confirm": TAXON_KEY})
    assert r.status_code == 204
    assert client.get(f"/species/{TAXON_KEY}").status_code == 404
    assert repo.get_species(TAXON_KEY) is None
    assert repo.load_occurrences(TAXON_KEY).empty
    assert repo.list_model_versions(TAXON_KEY) == []
    assert repo.get_native_range(TAXON_KEY) is None
    assert repo.list_jobs(TAXON_KEY) == []
    assert not folder.exists()
    assert client.delete(f"/species/{TAXON_KEY}", params={"confirm": TAXON_KEY}).status_code == 404
    with pytest.raises(NotFoundError):
        repo.delete_species(TAXON_KEY)


def test_delete_is_refused_while_a_job_is_active(client):
    r = client.post(f"/species/{TAXON_KEY}/runs", json={})
    assert r.status_code == 202
    r = client.delete(f"/species/{TAXON_KEY}", params={"confirm": TAXON_KEY})
    assert r.status_code == 409 and "discard it first" in r.json()["detail"]


def test_species_can_be_registered_again_after_deletion(settings):
    repo = InMemoryRepository()
    p = make_pipeline(settings, FakeGbif(initial=initial_records()), repo=repo)
    p.run(TAXON_KEY)
    repo.delete_species(TAXON_KEY)
    p.run(TAXON_KEY)  # a fresh first ingest
    assert repo.get_species(TAXON_KEY) is not None
    assert len(repo.load_occurrences(TAXON_KEY)) > 0


# --------------------------------------------------------------------------- discard
def test_cancel_endpoint_discards_queued_job_and_revokes_it(client):
    job = client.post(f"/species/{TAXON_KEY}/runs", json={}).json()
    r = client.post(f"/jobs/{job['job_id']}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    assert ("revoke", job["job_id"]) in client.queue.calls
    assert client.post(f"/jobs/{job['job_id']}/cancel").status_code == 409
    assert client.post(f"/jobs/{uuid.uuid4()}/cancel").status_code == 404
    # Deleting is possible once nothing is active.
    r = client.delete(f"/species/{TAXON_KEY}", params={"confirm": TAXON_KEY})
    assert r.status_code == 204


@pytest.fixture
def worker_repo(monkeypatch):
    """The in-memory repository as the worker's repository (tasks + progress callback)."""
    repo = InMemoryRepository()
    monkeypatch.setattr(factory, "get_repository", lambda: repo)
    monkeypatch.setattr(tasks, "get_repository", lambda: repo)
    return repo


def test_job_discarded_before_start_never_runs(worker_repo):
    job = worker_repo.create_job(Job(kind=JobKind.PIPELINE, taxon_key=TAXON_KEY))
    worker_repo.update_job(job.job_id, status=JobStatus.CANCELLED)
    ran = []
    out = tasks._run_job(str(job.job_id), lambda **kw: ran.append(kw), taxon_key=TAXON_KEY)
    assert out == {"discarded": True} and ran == []
    assert worker_repo.get_job(job.job_id).status == JobStatus.CANCELLED


def test_running_pipeline_stops_at_the_next_checkpoint(worker_repo, settings):
    p = make_pipeline(settings, FakeGbif(initial=initial_records()), repo=worker_repo)
    job = worker_repo.create_job(Job(kind=JobKind.PIPELINE, taxon_key=TAXON_KEY))
    p.progress = factory.job_progress(str(job.job_id))

    def run(taxon_key: int) -> dict:
        # The user discards the job just after it started.
        worker_repo.update_job(job.job_id, status=JobStatus.CANCELLED)
        return p.run(taxon_key)

    out = tasks._run_job(str(job.job_id), run, taxon_key=TAXON_KEY)
    assert out == {"discarded": True}
    stored = worker_repo.get_job(job.job_id)
    assert stored.status == JobStatus.CANCELLED and stored.message == "Discarded by user"
    sp = worker_repo.get_species(TAXON_KEY)
    if sp is not None:  # registered before the first checkpoint
        assert sp.status not in (SpeciesStatus.INGESTING, SpeciesStatus.TRAINING)
        assert sp.last_error == "Run discarded by user — run again to continue"


def test_progress_callback_raises_only_once_discarded(worker_repo):
    job = worker_repo.create_job(Job(kind=JobKind.PIPELINE, taxon_key=TAXON_KEY))
    report = factory.job_progress(str(job.job_id))
    report("ingest", 0.1, "working")  # active job: just reports
    assert worker_repo.get_job(job.job_id).stage == "ingest"
    worker_repo.update_job(job.job_id, status=JobStatus.CANCELLED)
    report_fresh = factory.job_progress(str(job.job_id))
    with pytest.raises(RunDiscarded):
        report_fresh("model", 0.5, "training")


def test_late_discard_reports_completed_work(worker_repo):
    job = worker_repo.create_job(Job(kind=JobKind.BULLETIN, taxon_key=TAXON_KEY))

    def finishes_anyway(**_kw):
        worker_repo.update_job(job.job_id, status=JobStatus.CANCELLED)
        return {"ok": True}

    assert tasks._run_job(str(job.job_id), finishes_anyway) == {"ok": True}
    stored = worker_repo.get_job(job.job_id)
    assert stored.status == JobStatus.SUCCEEDED
    assert stored.message == "Completed before the discard took effect"
