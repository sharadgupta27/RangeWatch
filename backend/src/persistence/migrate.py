"""Apply `db/schema.sql` (idempotent: CREATE ... IF NOT EXISTS / CREATE OR REPLACE).

Docker Compose applies the schema through Postgres' init directory on first start; Kubernetes
(and any existing database) runs this instead, e.g. as the `db-migrate` Job:

    python -m src.persistence.migrate
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import psycopg2

from src.config import get_settings

log = logging.getLogger(__name__)
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "db" / "schema.sql"


def dsn_from_sqlalchemy_url(url: str) -> str:
    """psycopg2 accepts libpq URLs; strip the SQLAlchemy driver suffix."""
    return url.replace("postgresql+psycopg2://", "postgresql://", 1)


def apply_schema(database_url: str, schema_path: Path = SCHEMA_PATH, wait_s: int = 120) -> None:
    deadline = time.monotonic() + wait_s
    while True:
        try:
            conn = psycopg2.connect(dsn_from_sqlalchemy_url(database_url))
            break
        except psycopg2.OperationalError as exc:
            if time.monotonic() > deadline:
                raise
            log.info("Waiting for Postgres: %s", exc)
            time.sleep(3)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(schema_path.read_text(encoding="utf-8"))
        log.info("Schema applied from %s", schema_path)
    finally:
        conn.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    apply_schema(get_settings().database_url)


if __name__ == "__main__":
    main()
