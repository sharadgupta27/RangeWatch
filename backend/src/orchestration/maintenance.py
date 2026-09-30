"""Storage-growth housekeeping (workplan §10, "archive/compress inactive species' feature
parquet files after N months of no re-query").

A species counts as re-queried whenever a user starts a job for it (pipeline run, bulletin,
scenario projection); the nightly monitoring sweep creates no job rows, so it does not keep a
species "active". Only the per-species feature table is archived: it is a derived cache that
the next training run rewrites. Training snapshots are left untouched — their bytes are
pinned by the SHA-256 recorded in the model lineage.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pyarrow.parquet as pq

from src.domain import utcnow
from src.persistence.artifact_store import ArtifactStore
from src.persistence.repository import Repository

log = logging.getLogger(__name__)

ZSTD_LEVEL = 19


def last_requested_ts(repo: Repository, taxon_key: int, registered_ts: datetime) -> datetime:
    """Latest user request for a species (its newest job), else its registration time."""
    jobs = repo.list_jobs(taxon_key, limit=1)
    return jobs[0].created_ts if jobs else registered_ts


def archive_inactive_features(
    repo: Repository,
    artifacts: ArtifactStore,
    inactive_for: timedelta,
    now: datetime | None = None,
) -> list[int]:
    """Move feature tables of species not requested within `inactive_for` into
    `species/<k>/archive/`, recompressed with zstd. Returns the archived taxon keys."""
    cutoff = (now or utcnow()) - inactive_for
    archived: list[int] = []
    for sp in repo.list_species():
        src = artifacts.local_path(artifacts.features_uri(sp.taxon_key))
        if not src.exists() or last_requested_ts(repo, sp.taxon_key, sp.created_ts) >= cutoff:
            continue
        dst = artifacts.local_path(artifacts.archived_features_uri(sp.taxon_key))
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".parquet.tmp")
        pq.write_table(pq.read_table(src), tmp, compression="zstd", compression_level=ZSTD_LEVEL)
        tmp.replace(dst)
        src.unlink()
        archived.append(sp.taxon_key)
        log.info("Archived feature table of inactive taxon %s", sp.taxon_key)
    return archived
