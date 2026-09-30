# Invasion Risk — species distribution MLOps dashboard

Pick a species. Occurrences are pulled from GBIF (primary) and iNaturalist (supplementary), a
MaxEnt model (`elapid`) is trained on a versioned WorldClim bioclim stack with spatial block
cross-validation, and suitability, MESS extrapolation, range zones and a transparent severity
index are projected globally and published as map tiles and a PDF bulletin. Tracked species are
re-checked for new records and **retrained only when warranted**, with every model version
logged (MLflow + model-lineage table) for reproducibility.

The science is standard (MaxEnt + bioclim + MESS); the contribution is the MLOps layer —
persistence, incremental ingestion, conditional retraining, lineage, automated bulletins — and a
React frontend. See [CLAUDE.md](CLAUDE.md) for the non-negotiable constraints and
[SDM_Dashboard_Workplan.md](SDM_Dashboard_Workplan.md) for the full plan.

```
React SPA (TanStack Router/Query/Table/Form/Store, MapLibre + deck.gl, shadcn/ui)
   │  typed client generated from FastAPI's OpenAPI schema
   ▼
FastAPI ──► Celery (Redis) ──► Prefect flows: raw_occurrences → features → model → suitability_raster → bulletin
   │                                   │
   │                                   ├─ connectors: pygbif / pyinaturalist
   │                                   ├─ features: rasterio (versioned 19-band COG)
   │                                   ├─ modeling: elapid MaxEnt, spatial CV, MESS, severity
   │                                   └─ persistence: PostGIS · artifact volume · MLflow
   ▼
pg_tileserv (occurrence / native-range vector tiles) · titiler (suitability / MESS / zones COG tiles)
```

## Quick start (Docker)

```bash
cp .env.example .env            # optional: add GBIF credentials for DOI-backed downloads
docker compose up -d --build
# one-time global static layer (10′ ≈ 50 MB for a quick start; 2.5m recommended for production)
docker compose run --rm api python -m src.features.bioclim_store build --version bioclim_v1 --resolution 10m
```

The `frontend` Nginx container is the **single public entry point** (`SDM_GATEWAY_PORT`,
default 8080). It serves the SPA and proxies the API and both tile servers on the same origin;
every other service binds to `127.0.0.1` only. Upstreams are re-resolved per request (the
container's own nameserver, or `SDM_DNS_RESOLVER`), so recreating `api`, `titiler` or
`pg_tileserv` never needs a gateway restart; on Kubernetes the upstreams are therefore FQDNs.

| Path / service | URL |
|---|---|
| Dashboard | http://localhost:8080 |
| API + OpenAPI docs | http://localhost:8080/api/docs |
| Raster / vector tiles | `/tiles/raster/…` (titiler) · `/tiles/vector/…` (pg_tileserv) |
| MLflow · Prefect UI (localhost only) | http://127.0.0.1:5000 · http://127.0.0.1:4200 |

**Authentication:** set `SDM_AUTH_USER` and `SDM_AUTH_PASSWORD` in `.env` to require HTTP Basic
auth for the dashboard, API and tiles (the browser reuses the credentials for tile requests
because everything is same-origin). Leave them empty for local use.

Then search a species in the dashboard → **Start analysis** → review/edit the proposed native
range → **Confirm & train**. Progress, status badges and layers update live.

**GBIF credentials** (`SDM_GBIF_USER/PWD/EMAIL`) switch first-time ingestion to the
DOI-backed download API (the citable route). Without them the paged search API is used — it
works, but is capped (`SDM_GBIF_SEARCH_MAX_RECORDS`, default 100k) and not DOI-citable; the
bulletin says so explicitly.

## Development

```bash
# backend (Python 3.11+)
cd backend && python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # or bin/ on Unix
python -m pytest            # offline: GBIF/iNat are faked, a synthetic bioclim stack is generated
ruff check src tests && black --check src tests
uvicorn src.api.main:app --reload

# frontend
cd frontend && npm ci
npm run gen:api:backend     # export OpenAPI from the backend + regenerate src/api/schema.d.ts
npm run dev                 # http://localhost:5173
npm test && npm run typecheck && npm run build
```

Regenerate the API types after **every** backend schema change — frontend types are never
hand-written.

### Frontend dev server

`npm run dev` proxies `/api`, `/tiles/raster` and `/tiles/vector` to the locally published
service ports (see `vite.config.ts`), mirroring the production gateway, so the app only ever
uses same-origin relative URLs.

### Global static layer (versioned, immutable)

```bash
python -m src.features.bioclim_store build --version bioclim_v1 --resolution 2.5m
python -m src.features.bioclim_store derive --from bioclim_v1 --version bioclim_v2 --core bio1,bio12
python -m src.features.bioclim_store add-scenario --version bioclim_v2 --gcm ACCESS-CM2 --ssp ssp245 --period 2041-2060
```

`SDM_BIOCLIM_VERSION` selects the active version. A version is never modified once a model has
used it — change the predictor screening or source by creating a new tag (`derive`), which the
model lineage then records. Scenarios are the one additive exception: they are projection inputs
only and never change what a model was trained on.

### Data versioning (DVC + per-model training snapshots)

Two kinds of data, versioned by the tool that fits each:

| Data | Versioned by | Where |
|---|---|---|
| Global static layer (`data/bioclim/<version>/`: stack, metadata, scenario COGs) | **DVC** — `data/bioclim/<version>.dvc` pointers in git, content in an S3 remote | `dvc-remote` service (local) / S3 in production |
| Occurrences | append-only PostGIS rows with `ingested_ts` | `occurrences` |
| Exact training set of each model version | **immutable snapshot, SHA-256 in the lineage** | `artifacts/species/<taxon>/v<N>/training_data.parquet` + `occurrence_manifest.parquet`, also logged to MLflow |
| Models and projections | `model_versions` + MLflow registry | `artifacts/species/<taxon>/v<N>/` |

Per-species data is deliberately **not** in DVC: it is appended nightly by the service, is
independently versioned per species, and PostGIS is its system of record — a git commit per
species per night would duplicate that. What cannot be re-derived from append-only rows (the
labels implied by the native-range polygon at training time, the seeded thinning and background
sample) is what the per-version snapshot captures. Every model's lineage shows
`bioclim_sha256`, `bioclim_dvc_md5` and the two snapshot hashes, so a version can be reproduced
with `dvc pull` + its snapshot. Curated native-range polygons (a few KB of GeoJSON) stay in git,
which diffs them better than DVC would.

One-time setup (DVC as an isolated tool; keys live only in `.env` and `.dvc/config.local`, both
git-ignored):

```bash
uv tool install "dvc[s3]"                       # or: pipx install "dvc[s3]"
# .env: SDM_DVC_ACCESS_KEY / SDM_DVC_SECRET_KEY (random values; see .env.example)
docker compose --profile dvc up -d dvc-remote   # S3 gateway on 127.0.0.1:7070 (versitygw)
dvc remote modify --local storage access_key_id     "$SDM_DVC_ACCESS_KEY"
dvc remote modify --local storage secret_access_key "$SDM_DVC_SECRET_KEY"
dvc pull                                        # restore data/bioclim/* on a new machine
```

After building a new version (or adding a scenario to one):

```bash
dvc add data/bioclim/bioclim_v3 && dvc push
git add data/bioclim/bioclim_v3.dvc && git commit -m "bioclim_v3: <what changed>"
```

For production, point the remote at real object storage (`dvc remote modify storage url
s3://<bucket>/sdm-dvc` and drop `endpointurl`), with credentials from the environment.

### Climate scenarios ("what-if")

After `add-scenario`, project existing models without retraining — from the species page
(**Climate scenarios** card), via `POST /api/species/{taxon_key}/scenarios`, or automatically in
the nightly sweep. Every scenario gets its **own** MESS extrapolation mask
(`extrapolation_<scenario>.tif`); future climates leave the training range far more often, so the
current-climate mask is never reused.

### Validation suite (workplan phase 11)

```bash
docker compose run --rm worker python -m src.orchestration.validation_suite run   # or the UI: /validation
```

Six reference invaders (*Linepithema humile, Lantana camara, Ailanthus altissima, Vespa velutina,
Pueraria montana, Carpobrotus edulis*) are ingested and trained through the normal pipeline, then:

1. **Known-range check** — land in documented invaded regions should be suitable and control
   regions unsuitable (a plausibility check: Model B trains on invaded records).
2. **Native-only transferability test** — a Model A trained on native records only scores the
   introduced records (independent; compare with the literature's mean AUC ≈ 0.7).

The suite supplies coarse, literature-cited native-range outlines through the curated-polygon
route (`data/native_range_polygons/`); they are shown with their source in the editor and never
replace a polygon a user has edited or confirmed. Reports: `artifacts/validation/*.json`,
`GET /api/validation`, and the **Validation** page.

First run at 10′ (29 Sep 2026): 5/6 pass; mean native-only → invaded AUC 0.690 (per species
0.565–0.843), in line with the literature. *Carpobrotus edulis* is flagged for review: coastal
California scores 24 % suitable against the 25 % bar — the check box spans the Central Valley and
Sierra, while the plant is strictly coastal. The region was left as defined rather than tuned
after seeing the result; a coast-hugging outline would be the principled fix.

### Re-sync

**Re-sync all** on the species page (`full_resync: true`) re-fetches every record rather than
the watermark delta and appends what is missing — use it after an interrupted first ingest or
after the search-API cap changed. With GBIF credentials it is a fresh DOI-backed download (the
workplan's "periodic full refresh"), which also becomes the species' cited download; without
them it uses the capped search API. The retrain decision still only counts records actually new.

## Deployment on Kubernetes

`deploy/k8s/base` (kustomize) contains PostGIS, Redis, MLflow, Prefect, the API, Celery
worker/beat, titiler, pg_tileserv, the gateway + Ingress, an idempotent `db-migrate` Job
(`python -m src.persistence.migrate`) and a suspended `bioclim-build` Job (2.5′).
`deploy/k8s/overlays/kind` is a single-node smoke-test overlay (dev secrets, RWO volumes, 10′
layer). For production add an overlay with registry image tags, an out-of-band `sdm-secrets`
Secret, ReadWriteMany storage for `sdm-data` / `sdm-artifacts`, and the Ingress host/TLS.

```bash
kubectl kustomize deploy/k8s/overlays/kind | kubeconform -strict -summary   # schema check
kind create cluster --name sdm
# `kind load docker-image` fails on Docker Desktop's multi-platform image store
# ("content digest ... not found"); import a single platform instead:
for img in sdm-backend:dev sdm-frontend:dev; do
  docker save --platform linux/amd64 "$img" \
    | docker exec -i sdm-control-plane ctr -n k8s.io images import --snapshotter=overlayfs -
done
kubectl apply -k deploy/k8s/overlays/kind
kubectl -n sdm port-forward svc/gateway 8088:80
```

## Troubleshooting

- **Blank maps.** Browsers cap live WebGL contexts per page; maps release their contexts on
  unmount and show a *Reload map* overlay if the browser still drops one. If a tab has been open
  across a deployment, the banner *A newer version of the dashboard has been deployed* offers a
  reload (the build id is polled from `/version.json`).
- **Basemap missing** (offline / CARTO blocked): data layers keep rendering on a plain
  background and the map says so.

## How the lifecycle works

`backend/src/orchestration/pipeline.py` implements the CLAUDE.md contract verbatim:

1. **New species** → full ingest (download or sliced search + iNat, de-duplicated via the iNat
   id carried by GBIF's iNat dataset) → a **draft** native-range polygon → status
   `awaiting_native_range_review`. Nothing is trained until a user confirms the polygon.
2. **Known species** → incremental GBIF pull (`lastInterpreted` since the watermark) + iNat
   `updated_since`. No new records → cached results. Otherwise append (occurrences are
   append-only), advance the watermark, and **retrain if new/total > 5 % or any new record falls
   in a cell the current model rates unsuitable**; else flag `minor_update_pending`.
3. **Train & publish** → features → Model B (native + invaded) whenever introduced records exist,
   else Model A (lower confidence) → Wallace/ENMeval feature-class × regularisation grid selected
   by spatially block-cross-validated CBI → global suitability, MESS, extrapolation mask and zone
   COGs → severity → hashed training-data snapshot → MLflow run + `model_versions` row → bulletin.
   Runs are serialised per species (Postgres advisory lock): a second request for the same
   species waits, then sees the first run's result instead of training a duplicate version.

Celery beat runs the nightly refresh of all tracked species, a stale-job reaper and a weekly
storage-growth sweep. A retrain that fails keeps `retrain_needed` set, so the next run retries it
even though its triggering records are already stored. The frontend only polls job/registry
state; it never duplicates this decision logic.

**Storage growth.** Species with no user request (no job) for `SDM_ARCHIVE_INACTIVE_AFTER_DAYS`
(default 180; `0` disables) have their feature table moved to
`species/<taxon>/archive/features_<taxon>.parquet`, recompressed with zstd. It is a derived
cache that the next training run rewrites; training snapshots are never touched, since their
SHA-256 is part of the lineage. The nightly sweep does not count as a request.

## Where the scientific constraints live

| CLAUDE.md constraint | Implementation |
|---|---|
| Transferability caveat | `TRANSFERABILITY_CAVEAT` in every model's metrics → UI banner, map badge, bulletin |
| MESS alongside projections | `evaluation.mess` → `mess.tif` + `extrapolation.tif`; layers endpoint always pairs them; MESS overlay on by default; hatched in the bulletin |
| Prefer Model B | `maxent_trainer.select_model_type`; Model A labelled lower-confidence in API, UI, bulletin |
| Spatial block CV | `elapid.GeographicKFold` (`maxent_trainer.spatial_block_cv`) |
| CBI with AUC/TSS | `evaluation.continuous_boyce_index`; shown first in the UI |
| Native range never automated | draft-only heuristics; training gated on confirmation; editable deck.gl layer |
| Severity not a black box | `severity_index.py` documents each component; weights configurable in the UI and printed in the bulletin footnote; MESS affects *confidence*, never the score |

## Design decisions beyond the workplan

- **`backend/src/persistence/`** was added for the repository (PostGIS + in-memory for tests),
  artifact store and MLflow registry — the layer is in the architecture but had no folder.
- **GBIF search slicing.** The search API stalls beyond offset ~10 000 (responses trickle
  indefinitely), so the credential-less fallback bisects the query into lat/lon slices of
  ≤ 9 900 records (duplicates on boundaries removed by gbifID). Every GBIF call has a
  `(10 s, 60 s)` timeout and bounded retries. `lastInterpreted` only accepts day precision, so
  incremental windows are day-granular; re-seen records never count towards the 5 % delta.
- **Protected core predictors.** Plain VIF elimination on WorldClim drops every cold-limit
  variable (BIO1/5/6/10/11 are mutually collinear), which produced implausible boreal
  suitability for *Linepithema humile*. `bio1` + `bio12` are now always retained (VIF < 5 still
  holds). On that species spatial-CV AUC rose 0.77 → 0.84 and TSS 0.44 → 0.59, and boreal
  artefacts disappeared.
- **Seeded sampling.** `elapid.sample_geoseries` takes no seed and elapid 1.0.4 has no
  distance thinning, so both are implemented in `background_sampler.py` with logged seeds.
- **Retrain threshold unchanged** (5 % delta or ≥ 1 new record outside the envelope,
  `SDM_RETRAIN_DELTA_FRACTION` / `SDM_ENVELOPE_MIN_OUTSIDE_POINTS`). Changing it is a
  reviewed decision per CLAUDE.md.

## Known limitations / next steps

- Basic auth protects a single-team deployment; multi-user roles (e.g. who may confirm a native
  range) would need an identity provider in front of the gateway.
- Without GBIF credentials, species above `SDM_GBIF_SEARCH_MAX_RECORDS` are sampled
  proportionally per spatial slice — geographically balanced, but not the complete, DOI-citable
  dataset.
- The retrain trigger fires when ≥ 1 new record falls outside the suitability envelope
  (`SDM_ENVELOPE_MIN_OUTSIDE_POINTS=1`, per the CLAUDE.md contract). For heavily recorded species
  this retrains on most nightly checks; raising it is a reviewed decision.
- The local stack runs the 10′ climate layer; production should build 2.5′ (Kubernetes
  `bioclim-build` Job default).
