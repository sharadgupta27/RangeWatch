"""Storage-growth housekeeping: archiving feature tables of long-inactive species."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pandas as pd

from src.domain import Job, JobKind, SpeciesRecord, utcnow
from src.orchestration.maintenance import archive_inactive_features
from src.persistence.artifact_store import ArtifactStore
from src.persistence.repository import InMemoryRepository


def _species_with_features(repo, artifacts, key, registered_days_ago):
    repo.create_species(
        SpeciesRecord(
            taxon_key=key,
            scientific_name=f"Sp {key}",
            created_ts=utcnow() - timedelta(days=registered_days_ago),
        )
    )
    path = artifacts.local_path(artifacts.features_uri(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"longitude": [1.0, 2.0], "latitude": [3.0, 4.0], "bio1": [10.5, 11.5]})
    df.to_parquet(path, index=False)
    return df


def test_only_species_without_recent_requests_are_archived(tmp_path):
    repo = InMemoryRepository()
    artifacts = ArtifactStore(tmp_path, "/data/artifacts")
    old = _species_with_features(repo, artifacts, 1, registered_days_ago=400)
    _species_with_features(repo, artifacts, 2, registered_days_ago=400)
    _species_with_features(repo, artifacts, 3, registered_days_ago=10)
    # Species 2 was requested recently; species 1's only request is long ago.
    recent = repo.create_job(Job(kind=JobKind.PIPELINE, taxon_key=2))
    stale = repo.create_job(Job(kind=JobKind.PIPELINE, taxon_key=1))
    repo._jobs[stale.job_id] = replace(stale, created_ts=utcnow() - timedelta(days=300))
    assert recent.created_ts > utcnow() - timedelta(minutes=1)

    assert archive_inactive_features(repo, artifacts, timedelta(days=180)) == [1]

    assert not artifacts.local_path(artifacts.features_uri(1)).exists()
    archived = artifacts.local_path(artifacts.archived_features_uri(1))
    pd.testing.assert_frame_equal(pd.read_parquet(archived), old)  # lossless
    assert artifacts.local_path(artifacts.features_uri(2)).exists()
    assert artifacts.local_path(artifacts.features_uri(3)).exists()
    # Idempotent: nothing left to archive.
    assert archive_inactive_features(repo, artifacts, timedelta(days=180)) == []
