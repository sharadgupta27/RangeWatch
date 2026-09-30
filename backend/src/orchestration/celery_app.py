"""Celery application: long-running jobs never run on the FastAPI request thread."""

from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

from src.config import get_settings

settings = get_settings()

celery_app = Celery(
    "sdm",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["src.orchestration.tasks"],
)
celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    beat_schedule={
        # Continuous monitoring: daily incremental check + conditional retrain for all species.
        "nightly-refresh-all-species": {
            "task": "sdm.refresh_all_species",
            "schedule": crontab(hour=2, minute=0),
        },
        "reap-stale-jobs": {"task": "sdm.reap_stale_jobs", "schedule": crontab(minute="*/15")},
        # Storage growth (workplan §10): archive feature tables of long-inactive species.
        "archive-inactive-species": {
            "task": "sdm.archive_inactive_species",
            "schedule": crontab(hour=3, minute=30, day_of_week="sun"),
        },
    },
)
