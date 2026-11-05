# ADR 0002 — Integrated SDM and presence–absence algorithms outside elapid

- **Status:** Accepted for experimental, opt-in use (requested November 2026); making either
  the default needs review.
- **Date:** 2026-11-02
- **Touches:** CLAUDE.md "Do not introduce a new SDM algorithm/library without first checking
  whether `elapid` already supports the needed feature"; improvements.md §4.4 ("R worker
  decision").

## Check against elapid 1.0.4

| Need | elapid | Decision |
|---|---|---|
| MaxEnt | `MaxentModel` | unchanged, still the default |
| Ensemble combination | `EnsembleModel` (unweighted mean / median / mode) | not used: the ensemble must be weighted by spatial-CV skill and expose per-member predictions for the agreement layer |
| GLM, GAM, RF, BRT | none | scikit-learn estimators (already an elapid dependency; version logged per model) |
| Environmental profiling | `NicheEnvelopeModel` | **used** for the `environmental_profile` pseudo-absence strategy |
| Integrated presence-only + presence–absence model | none | custom joint likelihood in numpy/scipy |
| Point process with an effort offset (B8) | `MaxentModel` has no offset term | penalised Poisson likelihood in numpy/scipy (`modeling/point_process.py`) |
| Bias shared across species (B9) | none | joint point process over the registry's species of one target group |
| Thinning (B3 / B4) | no thinning helper | numpy / scikit-learn (`features/thinning.py`) |

No new library was added. An R worker (`PointedSDMs`/INLA) was not introduced: it would add a
second runtime, container and Celery queue for a model that a ~150-line penalised likelihood
covers at this stage.

## Integrated model (`backend/src/modeling/integrated.py`)

- One environmental response β (linear + quadratic terms of standardised predictors).
- Presence-only GBIF/iNaturalist records: Poisson point process, log-intensity
  `a_po + f(x)·β + g·bias(x)`, quadrature = the uniform background (Berman & Turner 1992).
- Survey records (training uploads with absences): Bernoulli, cloglog link,
  `P = 1 − exp(−exp(a_pa + f(x)·β))`.
- Ridge penalty `l2` on β and g; L-BFGS with analytic gradients.
- The bias coefficient g of the target-group effort covariate is **estimated** (the survey
  data anchor β). Prediction drops it: the map is the probability of detecting the species in
  one survey unit, net of presence-only sampling bias.
- Falls back to MaxEnt, with the reason logged, when the training uploads have fewer than
  `min_survey_absences` (10) absences or 5 presences.

Limitations: no detection sub-model (repeat visits), no spatial random field, survey unit size
assumed constant, presence weights ignored. These are the reasons to revisit an INLA-based
model later.

## Point process and shared bias (`backend/src/modeling/point_process.py`)

- `log λ(x) = α + f(x)·β + Σ c_k z_k(x) + Σ g_j b_j(x)`: fixed-coefficient terms `z` are the
  offset (`log(target-group records + 1)` with c = 1, or the shared estimate), free terms `b`
  are estimated (the effort itself with `ppm_bias="estimated"`, static layers such as travel
  time) and held at their conditioning value for prediction. Uniform quadrature, ridge
  penalty, L-BFGS with analytic gradients.
- The offset coefficient is tested: the model is refitted with it free and the likelihood
  ratio against the assumed value is reported (`offset_test`).
- Shared bias: species-specific intercepts and climate responses, one set of bias
  coefficients, fitted on each species' presences with a fresh *uniform* quadrature (a
  target-group background already contains the bias being estimated). Stored per target
  group; a species without an estimate falls back to the plain offset, logged.
- Limitations: no presence weights, linear + quadratic features only, the effort surface is
  the GBIF map-API count without a time filter.

## Constraints kept

MESS / exDet / MOP / Shape / AOA, the transferability caveat and the confidence label are
attached to every algorithm's projection; spatial block CV is used for every member and model;
every choice (algorithm, members, weights, pseudo-absences, survey data) is logged per version
and shown in the lineage table.
