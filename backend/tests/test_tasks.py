"""Celery task wrappers: job-status bookkeeping around the Prefect flows (flows mocked)."""

from __future__ import annotations

import pytest

from src.domain import Job, JobKind, JobStatus
from src.orchestration import flows, tasks
from src.persistence.repository import InMemoryRepository


@pytest.fixture
def repo(monkeypatch):
    r = InMemoryRepository()
    monkeypatch.setattr(tasks, "get_repository", lambda: r)
    return r


def test_pipeline_task_marks_job_succeeded_and_forwards_job_id(repo, monkeypatch):
    job = repo.create_job(Job(kind=JobKind.PIPELINE, taxon_key=7))
    seen = {}

    def fake_flow(**kwargs):
        seen.update(kwargs)
        return {"decision": "served_cached_results"}

    monkeypatch.setattr(flows, "species_pipeline_flow", fake_flow)
    out = tasks.run_species_pipeline.run(job_id=str(job.job_id), taxon_key=7)
    assert out == {"decision": "served_cached_results"}
    assert seen == {
        "taxon_key": 7,
        "force_retrain": False,
        "full_resync": False,
        "job_id": str(job.job_id),
    }
    done = repo.get_job(job.job_id)
    assert done.status == JobStatus.SUCCEEDED and done.progress == 1.0


def test_task_failure_is_recorded(repo, monkeypatch):
    job = repo.create_job(Job(kind=JobKind.BULLETIN, taxon_key=7))

    def boom(**kwargs):
        raise RuntimeError("no model")

    monkeypatch.setattr(flows, "bulletin_flow", boom)
    with pytest.raises(RuntimeError):
        tasks.generate_bulletin.run(job_id=str(job.job_id), taxon_key=7)
    failed = repo.get_job(job.job_id)
    assert failed.status == JobStatus.FAILED and "no model" in failed.message


def test_stale_jobs_are_reaped(repo):
    from dataclasses import replace
    from datetime import timedelta

    from src.domain import SpeciesRecord, SpeciesStatus, utcnow

    repo.create_species(
        SpeciesRecord(taxon_key=7, scientific_name="X", status=SpeciesStatus.INGESTING)
    )
    old = repo.create_job(Job(kind=JobKind.PIPELINE, taxon_key=7))
    fresh = repo.create_job(Job(kind=JobKind.PIPELINE, taxon_key=8))
    repo._jobs[old.job_id] = replace(old, updated_ts=utcnow() - timedelta(hours=5))
    assert tasks.reap_stale_jobs.run() == 1
    assert repo.get_job(old.job_id).status == JobStatus.FAILED
    assert repo.get_job(fresh.job_id).status == JobStatus.QUEUED
    assert repo.get_species(7).status == SpeciesStatus.FAILED  # not stuck "ingesting"


def test_pipeline_factory_trains_with_the_configured_seed(monkeypatch):
    from src.config import get_settings
    from src.orchestration import factory

    monkeypatch.setenv("SDM_RANDOM_SEED", "7")
    get_settings.cache_clear()
    try:
        assert factory.build_pipeline().training_config.seed == 7
    finally:
        get_settings.cache_clear()
