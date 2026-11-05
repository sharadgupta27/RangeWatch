# ADR 0001 — Effort-aware retrain trigger (B10)

- **Status:** Proposed — implemented behind a switch that is **off by default**; needs review
  before it becomes the default.
- **Date:** 2026-11-02
- **Touches:** the retrain-trigger contract in CLAUDE.md ("Any change to the retrain-trigger
  threshold … must be a documented, reviewed decision").

## Context

The contract retrains when the new-record delta exceeds 5 % of the store, or when new records
fall outside the current suitability envelope (`SDM_ENVELOPE_MIN_OUTSIDE_POINTS`, default 1).
Many outside-envelope records come from observers *reaching new areas* (the iNaturalist boom),
not from the species spreading. Today both look the same to the trigger, so heavily recorded
species retrain on most nightly checks.

## What is implemented

Every new record is classified with the model version's target-group effort raster
(`effort.tif`, GBIF records of the species' order per map pixel):

- **expansion** — the cell already had ≥ `SDM_EFFORT_SAMPLED_MIN_RECORDS` (10) target-group
  records: observers were there before, so the species is probably new there;
- **discovery** — fewer records: observers just reached the cell;
- **unknown** — the version has no effort layer.

The classes are **always reported**, whatever the mode: in the early-detection alerts (U10),
in the pipeline outcome (`details.early_detection`) and in the RSS feed and webhooks.

`SDM_RETRAIN_TRIGGER_MODE`:

| Mode | Behaviour |
|---|---|
| `envelope` (default) | The reviewed contract, unchanged: every outside-envelope record counts. |
| `effort_aware` (proposed) | Discoveries do not count towards `SDM_ENVELOPE_MIN_OUTSIDE_POINTS`; expansions and unknowns do. The 5 % delta rule is unchanged. |

Code: `SpeciesPipeline.delta_outside_suitability_envelope` / `_delta_suitability`
(`backend/src/orchestration/pipeline.py`), classification in `src/monitoring/alerts.py`.
Tests: `tests/test_pipeline_p23.py::test_effort_aware_trigger_is_opt_in`.

## Consequences

- Fewer retrains for species whose records grow mainly through new observer coverage.
- Risk: a real invasion front in a previously unsampled region is delayed until the 5 % rule
  fires. Discoveries are still alerted, so a human sees them.
- The effort layer is the version's own (fetched at training time), i.e. "effort before the
  new records", at GBIF map-API resolution, with no time filter: effort that arrived after
  training but before the record still counts as "previous".

## Review questions

1. Is 10 target-group records per map pixel the right "previously sampled" cut-off?
2. Should discoveries count once they exceed a share of the delta (e.g. a cluster)?
3. Should the mode be per species (e.g. only for iNaturalist-dominated species)?
