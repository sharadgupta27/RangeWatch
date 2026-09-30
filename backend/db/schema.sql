-- SDM dashboard schema (PostgreSQL 15+ / PostGIS 3.3+).
-- Conventions (CLAUDE.md):
--   * species_registry: one row per GBIF taxon_key; only metadata rows are updated in place.
--   * occurrences: append-only, list-partitioned by taxon_key; bio1..bio19 nullable until extraction.
--   * artifacts are referenced by path/URI, never stored as blobs.

CREATE EXTENSION IF NOT EXISTS postgis;

-- ---------------------------------------------------------------------------
-- Species registry
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS species_registry (
    taxon_key               integer PRIMARY KEY,
    scientific_name         text NOT NULL,
    canonical_name          text,
    common_name             text,
    inat_taxon_id           integer,
    thumbnail_url           text,
    status                  text NOT NULL DEFAULT 'registered',
    last_gbif_fetch_ts      timestamptz,
    last_inat_fetch_ts      timestamptz,
    last_gbif_download_key  text,
    last_gbif_download_doi  text,
    n_occurrences_native    integer NOT NULL DEFAULT 0,
    n_occurrences_total     integer NOT NULL DEFAULT 0,
    n_pending_records       integer NOT NULL DEFAULT 0,
    bioclim_version_used    text,
    model_version           integer NOT NULL DEFAULT 0,
    model_path              text,
    model_trained_ts        timestamptz,
    model_metrics           jsonb,
    retrain_needed          boolean NOT NULL DEFAULT false,
    severity_config         jsonb,
    last_error              text,
    created_ts              timestamptz NOT NULL DEFAULT now(),
    updated_ts              timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Occurrence store (append-only, partitioned by taxon_key).
-- A partition per species is created on registration by the repository:
--   CREATE TABLE occurrences_<key> PARTITION OF occurrences FOR VALUES IN (<key>);
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS occurrences (
    taxon_key                 integer NOT NULL REFERENCES species_registry (taxon_key),
    source                    text NOT NULL CHECK (source IN ('GBIF', 'iNaturalist')),
    record_id                 bigint NOT NULL,
    gbif_id                   bigint GENERATED ALWAYS AS (CASE WHEN source = 'GBIF' THEN record_id END) STORED,
    inat_id                   bigint GENERATED ALWAYS AS (CASE WHEN source = 'iNaturalist' THEN record_id END) STORED,
    -- iNaturalist observation id for GBIF records that came from the iNat research-grade
    -- dataset; used to avoid double counting when supplementing directly from iNaturalist.
    inat_crossref_id          bigint,
    geom                      geometry(Point, 4326) NOT NULL,
    event_date                date,
    coordinate_uncertainty_m  double precision,
    basis_of_record           text,
    country_code              text,
    range_label               text NOT NULL DEFAULT 'unknown'
                              CHECK (range_label IN ('native', 'introduced', 'unknown')),
    ingested_ts               timestamptz NOT NULL DEFAULT now(),
    bio1 real, bio2 real, bio3 real, bio4 real, bio5 real, bio6 real, bio7 real,
    bio8 real, bio9 real, bio10 real, bio11 real, bio12 real, bio13 real, bio14 real,
    bio15 real, bio16 real, bio17 real, bio18 real, bio19 real,
    bioclim_version           text,
    PRIMARY KEY (taxon_key, source, record_id)
) PARTITION BY LIST (taxon_key);

CREATE TABLE IF NOT EXISTS occurrences_default PARTITION OF occurrences DEFAULT;

CREATE INDEX IF NOT EXISTS occurrences_geom_gist ON occurrences USING gist (geom);
CREATE INDEX IF NOT EXISTS occurrences_taxon_ingested ON occurrences (taxon_key, ingested_ts);

-- ---------------------------------------------------------------------------
-- Native-range polygons (user-reviewed; never auto-confirmed)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS native_ranges (
    taxon_key     integer PRIMARY KEY REFERENCES species_registry (taxon_key),
    geom          geometry(MultiPolygon, 4326) NOT NULL,
    status        text NOT NULL CHECK (status IN ('draft', 'confirmed')),
    source        text NOT NULL,
    note          text,
    updated_ts    timestamptz NOT NULL DEFAULT now(),
    confirmed_ts  timestamptz
);
CREATE INDEX IF NOT EXISTS native_ranges_geom_gist ON native_ranges USING gist (geom);

-- ---------------------------------------------------------------------------
-- Model lineage (one row per trained model version; species_registry holds the current one)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_versions (
    taxon_key        integer NOT NULL REFERENCES species_registry (taxon_key),
    model_version    integer NOT NULL,
    model_type       text NOT NULL CHECK (model_type IN ('A', 'B')),
    trained_ts       timestamptz NOT NULL DEFAULT now(),
    trigger          text NOT NULL,
    model_path       text NOT NULL,
    artifacts        jsonb NOT NULL,   -- relative artifact paths (suitability, mess, zones, ...)
    metrics          jsonb NOT NULL,   -- AUC/TSS/CBI + reproducibility block
    severity         jsonb,
    mlflow_run_id    text,
    PRIMARY KEY (taxon_key, model_version)
);

-- ---------------------------------------------------------------------------
-- Async jobs (Celery-backed); polled by the frontend
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS jobs (
    job_id       uuid PRIMARY KEY,
    taxon_key    integer,
    kind         text NOT NULL,
    status       text NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    stage        text,
    progress     real NOT NULL DEFAULT 0,
    message      text,
    result       jsonb,
    created_ts   timestamptz NOT NULL DEFAULT now(),
    updated_ts   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_taxon_created ON jobs (taxon_key, created_ts DESC);

-- ---------------------------------------------------------------------------
-- pg_tileserv function layer: occurrence vector tiles.
--   /public.occurrence_tiles/{z}/{x}/{y}.pbf?taxon_key=...&max_year=...
-- At low zoom points are aggregated server-side onto a screen-space grid (count +
-- per-label counts), so large point sets never reach the browser unaggregated.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.occurrence_tiles(
    z integer, x integer, y integer,
    taxon_key integer DEFAULT 0,
    max_year integer DEFAULT 9999
)
RETURNS bytea AS $$
DECLARE
    result bytea;
    bounds geometry := ST_TileEnvelope(z, x, y);
    cell double precision := (2 * 20037508.342789244) / (2 ^ z) / 64.0;
BEGIN
    IF z < 7 THEN
        WITH pts AS (
            SELECT ST_Transform(o.geom, 3857) AS g, o.range_label
            FROM occurrences o
            WHERE o.taxon_key = occurrence_tiles.taxon_key
              AND (o.event_date IS NULL OR EXTRACT(YEAR FROM o.event_date) <= max_year)
              AND o.geom && ST_Transform(bounds, 4326)
        ), agg AS (
            SELECT ST_SnapToGrid(g, cell) AS g,
                   count(*)::int AS n,
                   count(*) FILTER (WHERE range_label = 'native')::int AS n_native,
                   count(*) FILTER (WHERE range_label = 'introduced')::int AS n_introduced
            FROM pts GROUP BY 1
        ), mvtgeom AS (
            SELECT ST_AsMVTGeom(g, bounds) AS geom, n, n_native, n_introduced
            FROM agg
        )
        SELECT ST_AsMVT(mvtgeom, 'occurrences') INTO result FROM mvtgeom;
    ELSE
        WITH mvtgeom AS (
            SELECT ST_AsMVTGeom(ST_Transform(o.geom, 3857), bounds) AS geom,
                   1 AS n,
                   o.source,
                   o.record_id,
                   o.range_label,
                   EXTRACT(YEAR FROM o.event_date)::int AS year
            FROM occurrences o
            WHERE o.taxon_key = occurrence_tiles.taxon_key
              AND (o.event_date IS NULL OR EXTRACT(YEAR FROM o.event_date) <= max_year)
              AND o.geom && ST_Transform(bounds, 4326)
        )
        SELECT ST_AsMVT(mvtgeom, 'occurrences') INTO result FROM mvtgeom;
    END IF;
    RETURN result;
END;
$$ LANGUAGE plpgsql STABLE PARALLEL SAFE;

COMMENT ON FUNCTION public.occurrence_tiles IS
    'Occurrence vector tiles for one taxon; aggregated below z7, raw points above.';

-- pg_tileserv function layer: native-range polygon as vector tiles (display only; the
-- editor fetches the single polygon as GeoJSON from the API).
CREATE OR REPLACE FUNCTION public.native_range_tiles(
    z integer, x integer, y integer,
    taxon_key integer DEFAULT 0
)
RETURNS bytea AS $$
    WITH bounds AS (SELECT ST_TileEnvelope(z, x, y) AS geom),
    mvtgeom AS (
        SELECT ST_AsMVTGeom(ST_Transform(n.geom, 3857), b.geom) AS geom, n.status
        FROM native_ranges n, bounds b
        WHERE n.taxon_key = native_range_tiles.taxon_key
          AND ST_Transform(n.geom, 3857) && b.geom
    )
    SELECT ST_AsMVT(mvtgeom, 'native_range') FROM mvtgeom;
$$ LANGUAGE sql STABLE PARALLEL SAFE;
