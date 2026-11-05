"""MaxEnt training via `elapid.MaxentModel` with spatial block cross-validation.

Modeling defaults are reused from Wallace / ENMeval rather than re-derived: the candidate
feature classes L, LQ, H, LQH with regularization multipliers 1 and 2, selected by spatially
cross-validated performance. Model selection uses mean test CBI (presence-only appropriate),
with mean test AUC as tie-breaker. Evaluation metrics come from `evaluation.py`.

Optional sampling-bias handling (features/sampling_bias.py), both off by default:
  * presence weights are passed to elapid as `sample_weight` (background weight 1) in every
    CV fold and in the final fit; CV test metrics stay unweighted;
  * bias covariates (e.g. target-group effort) are extra model columns. Spatial CV scores the
    model with the observed covariate values (it predicts the biased observations); the
    published model is a `BiasConditionedModel` that holds each covariate at its conditioning
    value, so the suitability surface, its threshold and every projection are bias-corrected.
    MESS / exDet / MOP / AOA and the presence envelope use the climate predictors only.

`algorithm` (MaxEnt unless configured otherwise):
  * "ensemble" (U3): MaxEnt plus presence–pseudo-absence GLM / GAM / RF / BRT members
    (modeling/algorithms.py), evaluated on shared spatial blocks against pseudo-absences and
    combined by CV-weighted mean; `predict_members` feeds the ensemble-agreement layer;
  * "integrated" (§4.4): presence-only + presence–absence survey data with one shared
    environmental response (modeling/integrated.py); needs training uploads with absences;
  * "ppm" (B8 / B9): Poisson point process with an explicit log-effort offset, an estimated
    effort coefficient, or the bias coefficients shared by the species' target group
    (modeling/point_process.py); the offset assumption is tested by likelihood ratio.

Every version also gets all threshold rules (`threshold_rule` picks the one used for the
binary zones) and partial-dependence response curves (modeling/response.py).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import elapid
import geopandas as gpd
import numpy as np
import pandas as pd

from src.domain import ModelType
from src.features.sampling_bias import (
    BIAS_CONDITIONS,
    BIAS_COVARIATES,
    DEFAULT_TIER_WEIGHTS,
    PRESENCE_WEIGHTINGS,
)
from src.features.thinning import THIN_METHODS
from src.modeling.algorithms import (
    ALGORITHMS,
    MemberResult,
    WeightedEnsemble,
    ensemble_weights,
    fit_pa_member,
    spatial_block_ids,
)
from src.modeling.evaluation import (
    THRESHOLD_RULES,
    ExtrapolationReference,
    FoldMetrics,
    MessReference,
    auc,
    choose_threshold,
    evaluate_fold,
    p10_threshold,
    permutation_importance,
    summarize_folds,
    threshold_candidates,
)
from src.modeling.integrated import IntegratedData, fit_integrated
from src.modeling.point_process import PpmData, fit_ppm, offset_test
from src.modeling.response import response_curves

log = logging.getLogger(__name__)

FEATURE_CLASSES: dict[str, list[str]] = {
    "L": ["linear"],
    "LQ": ["linear", "quadratic"],
    "H": ["hinge"],
    "LQH": ["linear", "quadratic", "hinge"],
    "LQHP": ["linear", "quadratic", "hinge", "product"],
}
WALLACE_TUNING_GRID: tuple[tuple[str, float], ...] = tuple(
    (fc, rm) for fc in ("L", "LQ", "H", "LQH") for rm in (1.0, 2.0)
)
MIN_INTRODUCED_FOR_MODEL_B = 5
ALGORITHM_MODES = ("maxent", "ensemble", "integrated", "ppm")
PPM_BIAS_MODES = ("offset", "estimated", "shared")
PSEUDO_ABSENCE_STRATEGIES = ("background", "disk", "environmental_profile", "completeness")

TRANSFERABILITY_CAVEAT = (
    "Native-range-only MaxEnt models show only moderate transferability to invaded ranges "
    "(mean AUC ≈ 0.7 in cross-continental evaluations). Suitability outside the training "
    "range is an exploratory screening signal, not a confident invasion forecast; interpret "
    "together with the MESS extrapolation layer and the exDet / MOP / Shape / AOA diagnostics."
)


@dataclass(frozen=True)
class TrainingConfig:
    feature_classes: str = "LQH"
    beta_multiplier: float = 1.0
    tune: bool = True
    tuning_grid: tuple[tuple[str, float], ...] = WALLACE_TUNING_GRID
    n_folds: int = 4
    n_background: int = 10_000
    background_buffer_km: float = 500.0
    # "target_group" (preferred: background mirrors the observer bias of the presences, drawn
    # from cells where GBIF holds records of the species' `target_group_rank`) or "buffer"
    # (uniform inside the buffer). Target-group falls back to buffer when the effort data
    # has fewer than `min_target_group_cells` occupied cells in the region.
    background_method: str = "target_group"
    target_group_rank: str = "order"
    # unique_cells | record_count | sqrt_record_count (features/background_sampler.py)
    target_group_weighting: str = "unique_cells"
    min_target_group_cells: int = 500
    thin_km: float = 10.0
    # Thinning (features/thinning.py): distance (fixed thin_km, default) | autocorrelation
    # (B3: variogram range, clamped to [thin_km, thin_max_km]) | environmental (B4: one record
    # per cell of an env_thin_bins grid on the first env_thin_axes principal axes).
    # thin_reps > 1 = maximal-retention thinning (spThin) for the geographic methods.
    thin_method: str = "distance"
    thin_reps: int = 1
    thin_max_km: float = 50.0
    env_thin_bins: int = 20
    env_thin_axes: int = 2
    # B1: restrict the target-group effort to the presences' basis of record / year window.
    target_group_match_source: bool = False
    target_group_match_years: bool = False
    min_presences: int = 15
    importance_repeats: int = 5
    # MOP: share (%) of the closest reference points averaged (mop R package default: 1).
    mop_percentage: float = 1.0
    # Presence observation weights: none | quality_tier | density | quality_tier_density.
    presence_weighting: str = "none"
    tier_weights: tuple[tuple[str, float], ...] = tuple(DEFAULT_TIER_WEIGHTS.items())
    density_neighbors: int = 5
    # Bias covariate: none | target_group_effort (then the background is drawn uniformly in
    # the buffer: observer bias is modelled explicitly instead of mirrored by the background).
    bias_covariate: str = "none"
    bias_condition: str = "max"  # max | p95 of the covariate over the training points
    # Static observer-bias rasters registered with the bioclim version (travel time, roads, …),
    # added as bias covariates and held at their most-effort end for prediction.
    bias_static_layers: tuple[str, ...] = ()
    # Threshold used for the binary zones: p10 | max_tss | equal_sens_spec |
    # min_training_presence | fixed (then `threshold_value`). All rules are logged.
    threshold_rule: str = "p10"
    threshold_value: float | None = None
    # maxent | ensemble | integrated (see the module docstring).
    algorithm: str = "maxent"
    ensemble_members: tuple[str, ...] = ALGORITHMS
    ensemble_min_auc: float = 0.7
    # Pseudo-absences for the presence–absence members: background | disk |
    # environmental_profile | completeness (inferred absences from well-surveyed cells).
    pseudo_absence_strategy: str = "disk"
    n_pseudo_absences: int = 5_000
    pa_min_distance_km: float = 50.0
    pa_repeats: int = 5
    completeness_min_records: float = 50.0
    # Integrated model: ridge penalty and minimum survey absences required.
    integrated_l2: float = 1.0
    min_survey_absences: int = 10
    # Point process (B8 / B9): offset | estimated | shared; ridge penalty.
    ppm_bias: str = "offset"
    ppm_l2: float = 1.0
    # Limiting-factor raster (Elith et al. 2010): one extra prediction per predictor.
    limiting_factor: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        for value, allowed, name in (
            (self.presence_weighting, PRESENCE_WEIGHTINGS, "presence_weighting"),
            (self.bias_covariate, BIAS_COVARIATES, "bias_covariate"),
            (self.bias_condition, BIAS_CONDITIONS, "bias_condition"),
            (self.threshold_rule, tuple(THRESHOLD_RULES), "threshold_rule"),
            (self.algorithm, ALGORITHM_MODES, "algorithm"),
            (self.pseudo_absence_strategy, PSEUDO_ABSENCE_STRATEGIES, "pseudo_absence_strategy"),
            (self.thin_method, THIN_METHODS, "thin_method"),
            (self.ppm_bias, PPM_BIAS_MODES, "ppm_bias"),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, not {value!r}")
        unknown = set(self.ensemble_members) - set(ALGORITHMS)
        if unknown or not self.ensemble_members:
            raise ValueError(f"ensemble_members must be a non-empty subset of {ALGORITHMS}")
        if self.threshold_rule == "fixed" and self.threshold_value is None:
            raise ValueError("threshold_rule 'fixed' needs threshold_value")
        if self.thin_reps < 1 or self.env_thin_bins < 2 or self.env_thin_axes < 1:
            raise ValueError("thin_reps ≥ 1, env_thin_bins ≥ 2 and env_thin_axes ≥ 1 required")

    @property
    def needs_pseudo_absences(self) -> bool:
        return self.algorithm == "ensemble" and any(m != "maxent" for m in self.ensemble_members)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "feature_classes": self.feature_classes,
            "beta_multiplier": self.beta_multiplier,
            "tune": self.tune,
            "tuning_grid": [list(g) for g in self.tuning_grid],
            "n_folds": self.n_folds,
            "n_background": self.n_background,
            "background_buffer_km": self.background_buffer_km,
            "background_method": self.background_method,
            "target_group_rank": self.target_group_rank,
            "target_group_weighting": self.target_group_weighting,
            "min_target_group_cells": self.min_target_group_cells,
            "thin_km": self.thin_km,
            "thin_method": self.thin_method,
            "thin_reps": self.thin_reps,
            "target_group_match_source": self.target_group_match_source,
            "target_group_match_years": self.target_group_match_years,
            "min_presences": self.min_presences,
            "importance_repeats": self.importance_repeats,
            "mop_percentage": self.mop_percentage,
            "presence_weighting": self.presence_weighting,
            "tier_weights": dict(self.tier_weights),
            "density_neighbors": self.density_neighbors,
            "bias_covariate": self.bias_covariate,
            "bias_condition": self.bias_condition,
            "bias_static_layers": list(self.bias_static_layers),
            "threshold_rule": self.threshold_rule,
            "threshold_value": self.threshold_value,
            "algorithm": self.algorithm,
            "limiting_factor": self.limiting_factor,
            "seed": self.seed,
        }
        if self.algorithm == "ensemble":
            out.update(
                ensemble_members=list(self.ensemble_members),
                ensemble_min_auc=self.ensemble_min_auc,
                pseudo_absence_strategy=self.pseudo_absence_strategy,
                n_pseudo_absences=self.n_pseudo_absences,
                pa_min_distance_km=self.pa_min_distance_km,
                pa_repeats=self.pa_repeats,
                completeness_min_records=self.completeness_min_records,
            )
        if self.algorithm == "integrated":
            out.update(
                integrated_l2=self.integrated_l2, min_survey_absences=self.min_survey_absences
            )
        if self.algorithm == "ppm":
            out.update(ppm_bias=self.ppm_bias, ppm_l2=self.ppm_l2)
        if self.thin_method == "autocorrelation":
            out.update(thin_max_km=self.thin_max_km)
        if self.thin_method == "environmental":
            out.update(env_thin_bins=self.env_thin_bins, env_thin_axes=self.env_thin_axes)
        return out


@dataclass
class TrainingData:
    predictors: list[str]
    presence_lon: np.ndarray
    presence_lat: np.ndarray
    presence_x: pd.DataFrame
    background_lon: np.ndarray
    background_lat: np.ndarray
    background_x: pd.DataFrame
    background_method: str
    # Observation weight per presence (None = unweighted).
    presence_weight: np.ndarray | None = None
    # Bias covariate column (in presence_x / background_x) → value it is held at for prediction.
    bias_covariates: dict[str, float] = field(default_factory=dict)
    # Pseudo-absences for the presence–absence ensemble members (model columns).
    absence_lon: np.ndarray | None = None
    absence_lat: np.ndarray | None = None
    absence_x: pd.DataFrame | None = None
    # Presence–absence survey records for the integrated model (climate predictors).
    survey_lon: np.ndarray | None = None
    survey_lat: np.ndarray | None = None
    survey_x: pd.DataFrame | None = None
    survey_y: np.ndarray | None = None
    # Point process: bias column → fixed coefficient (the offset; B8 / B9). Bias columns not
    # listed are estimated.
    offset_coefficients: dict[str, float] = field(default_factory=dict)

    @property
    def model_columns(self) -> list[str]:
        return [*self.predictors, *self.bias_covariates]

    @property
    def n_presence(self) -> int:
        return len(self.presence_x)

    @property
    def n_background(self) -> int:
        return len(self.background_x)


@dataclass
class TrainingResult:
    model: Any  # elapid.MaxentModel, ensemble or integrated model, maybe bias-conditioned
    model_type: ModelType
    predictors: list[str]
    feature_classes: str
    beta_multiplier: float
    folds: list[FoldMetrics]
    cv_summary: dict[str, float]
    tuning: list[dict[str, Any]]
    threshold: float
    train_auc: float
    importance: dict[str, float]
    mess_reference: MessReference
    presence_envelope: dict[str, tuple[float, float]]
    n_presence: int
    n_background: int
    background_method: str
    # exDet / MOP / Shape / AOA reference (None only for models trained before it existed).
    extrapolation_reference: ExtrapolationReference | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    algorithm: str = "maxent"
    threshold_rule: str = THRESHOLD_RULES["p10"]
    # Every threshold rule (value, training sensitivity, background specificity).
    thresholds: dict[str, dict[str, Any]] = field(default_factory=dict)
    response_curves: dict[str, Any] | None = None
    # Mean of each predictor over the training presences (limiting-factor reference).
    presence_mean: dict[str, float] = field(default_factory=dict)
    cv_method: str | None = None

    @property
    def confidence_label(self) -> str:
        return "standard" if self.model_type == "B" else "lower"


class BiasConditionedModel:
    """A model with bias covariates, predicting with each covariate held at its conditioning
    value. Callers pass the climate predictors only, exactly as for a plain model, so
    projections, scenarios, high-resolution regions and checks need no changes."""

    def __init__(self, model: Any, predictors: Sequence[str], bias: dict[str, float]) -> None:
        self.model = model
        self.predictors = list(predictors)
        self.bias = dict(bias)

    def _full(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        frame = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x, columns=self.predictors)
        full = frame[self.predictors].reset_index(drop=True).copy()
        for name, value in self.bias.items():
            full[name] = value
        return full

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self._full(x)))

    def __getattr__(self, name: str) -> Any:
        # Ensemble helpers (predict_members, member_agreement) pass through, conditioned.
        if name in ("predict_members", "member_agreement") and hasattr(self.model, name):
            inner = getattr(self.model, name)
            return lambda x, *args: inner(self._full(x), *args)
        raise AttributeError(name)


def select_model_type(effective_labels: pd.Series) -> ModelType:
    """Model B (combined native + invaded) whenever introduced-range records exist."""
    n_intro = int((effective_labels == "introduced").sum())
    return "B" if n_intro >= MIN_INTRODUCED_FOR_MODEL_B else "A"


def training_presence_mask(effective_labels: pd.Series, model_type: ModelType) -> np.ndarray:
    if model_type == "A":
        return (effective_labels == "native").to_numpy()
    return effective_labels.isin(["native", "introduced"]).to_numpy()


def _make_model(feature_classes: str, beta: float, seed: int) -> elapid.MaxentModel:
    return elapid.MaxentModel(
        feature_types=FEATURE_CLASSES[feature_classes],
        beta_multiplier=beta,
        random_state=seed,
        n_cpus=1,
    )


def _stack_xy(
    data: TrainingData,
) -> tuple[pd.DataFrame, np.ndarray, gpd.GeoSeries, np.ndarray | None]:
    cols = data.model_columns
    x = pd.concat([data.presence_x[cols], data.background_x[cols]], ignore_index=True)
    y = np.r_[np.ones(data.n_presence), np.zeros(data.n_background)]
    w = (
        None
        if data.presence_weight is None
        else np.r_[np.asarray(data.presence_weight, dtype="float64"), np.ones(data.n_background)]
    )
    pts = gpd.GeoSeries(
        gpd.points_from_xy(
            np.r_[data.presence_lon, data.background_lon],
            np.r_[data.presence_lat, data.background_lat],
        ),
        crs="EPSG:4326",
    )
    return x, y, pts, w


def _fit(model: elapid.MaxentModel, x: pd.DataFrame, y: np.ndarray, w: np.ndarray | None) -> None:
    # elapid scales sample_weight by its class weights in place: always pass a copy.
    model.fit(x, y, sample_weight=None if w is None else np.array(w, dtype="float64"))


def spatial_fold_ids(points: gpd.GeoSeries, n_folds: int, seed: int) -> np.ndarray:
    """Spatial CV fold of every row — the same deterministic split `spatial_block_cv` uses."""
    ids = np.zeros(len(points), dtype=int)
    splitter = elapid.GeographicKFold(n_splits=n_folds, random_state=seed)
    for i, (_, te) in enumerate(splitter.split(points)):
        ids[te] = i
    return ids


def spatial_block_cv(
    x: pd.DataFrame,
    y: np.ndarray,
    points: gpd.GeoSeries,
    feature_classes: str,
    beta: float,
    n_folds: int,
    seed: int,
    weights: np.ndarray | None = None,
) -> list[FoldMetrics]:
    """Spatial k-fold CV: folds are geographic clusters (elapid.GeographicKFold), never random.
    Training folds use the observation weights; test metrics are unweighted."""
    folds: list[FoldMetrics] = []
    splitter = elapid.GeographicKFold(n_splits=n_folds, random_state=seed)
    for i, (tr, te) in enumerate(splitter.split(points)):
        y_tr, y_te = y[tr], y[te]
        if y_tr.sum() < 5 or y_te.sum() < 1 or (1 - y_te).sum() < 1:
            log.debug("Skipping degenerate spatial fold %d", i)
            continue
        model = _make_model(feature_classes, beta, seed)
        _fit(model, x.iloc[tr], y_tr, None if weights is None else weights[tr])
        pred = np.asarray(model.predict(x.iloc[te]), dtype="float64").ravel()
        folds.append(evaluate_fold(i, int(y_tr.sum()), pred[y_te == 1], pred[y_te == 0]))
    return folds


def train_model(data: TrainingData, model_type: ModelType, cfg: TrainingConfig) -> TrainingResult:
    """Train with the configured algorithm (MaxEnt by default)."""
    if cfg.algorithm == "ensemble":
        return train_ensemble(data, model_type, cfg)
    if cfg.algorithm == "integrated":
        return train_integrated(data, model_type, cfg)
    if cfg.algorithm == "ppm":
        return train_ppm(data, model_type, cfg)
    return train_maxent(data, model_type, cfg)


def _check_presences(data: TrainingData, cfg: TrainingConfig) -> None:
    if data.n_presence < cfg.min_presences:
        raise ValueError(
            f"Only {data.n_presence} presences after thinning (minimum {cfg.min_presences})"
        )


def train_maxent(data: TrainingData, model_type: ModelType, cfg: TrainingConfig) -> TrainingResult:
    _check_presences(data, cfg)
    x, y, pts, w = _stack_xy(data)

    grid = cfg.tuning_grid if cfg.tune else ((cfg.feature_classes, cfg.beta_multiplier),)
    tuning: list[dict[str, Any]] = []
    best: tuple[float, float] = (-np.inf, -np.inf)
    best_cfg = grid[0]
    best_folds: list[FoldMetrics] = []
    for fc, rm in grid:
        folds = spatial_block_cv(x, y, pts, fc, rm, cfg.n_folds, cfg.seed, w)
        summary = summarize_folds(folds) if folds else {}
        tuning.append(
            {"feature_classes": fc, "beta_multiplier": rm, "n_folds": len(folds), **summary}
        )
        score = (
            np.nan_to_num(summary.get("cbi_mean", np.nan), nan=-np.inf),
            np.nan_to_num(summary.get("auc_mean", np.nan), nan=-np.inf),
        )
        if folds and score > best:
            best, best_cfg, best_folds = score, (fc, rm), folds
    if not best_folds:
        raise ValueError("Spatial cross-validation produced no usable folds")

    fc, rm = best_cfg
    fitted = _make_model(fc, rm, cfg.seed)
    _fit(fitted, x, y, w)
    full_importance = _permutation_importance(fitted, x, y, data.model_columns, cfg)
    return _finalize(
        data,
        model_type,
        cfg,
        fitted,
        full_importance,
        best_folds,
        tuning,
        fc,
        rm,
        algorithm="maxent",
        cv_method=f"spatial block CV (elapid.GeographicKFold, k={cfg.n_folds})",
        extra={},
    )


def _finalize(
    data: TrainingData,
    model_type: ModelType,
    cfg: TrainingConfig,
    fitted: Any,
    full_importance: dict[str, float],
    folds: list[FoldMetrics],
    tuning: list[dict[str, Any]],
    feature_classes: str,
    beta: float,
    algorithm: str,
    cv_method: str,
    extra: dict[str, Any],
    conditioned: bool = True,
) -> TrainingResult:
    """Shared tail of every algorithm: bias conditioning, thresholds (all rules), training
    AUC, response curves and the climate-only extrapolation references."""
    x, y, pts, w = _stack_xy(data)
    climate = x[data.predictors]
    model: Any = (
        BiasConditionedModel(fitted, data.predictors, data.bias_covariates)
        if data.bias_covariates and conditioned
        else fitted
    )
    # Thresholds and training AUC refer to the published (bias-conditioned) surface.
    train_pred = np.asarray(model.predict(climate), dtype="float64").ravel()
    pres_pred, bg_pred = train_pred[y == 1], train_pred[y == 0]
    thresholds = threshold_candidates(pres_pred, bg_pred, cfg.threshold_value)
    threshold, rule_label = choose_threshold(thresholds, cfg.threshold_rule)
    importance = {p: full_importance.get(p, 0.0) for p in data.predictors}
    envelope = {
        p: (
            float(np.nanpercentile(data.presence_x[p], 5)),
            float(np.nanpercentile(data.presence_x[p], 95)),
        )
        for p in data.predictors
    }
    return TrainingResult(
        model=model,
        model_type=model_type,
        predictors=list(data.predictors),
        feature_classes=feature_classes,
        beta_multiplier=beta,
        folds=folds,
        cv_summary=summarize_folds(folds),
        tuning=tuning,
        threshold=threshold,
        train_auc=auc(pres_pred, bg_pred),
        importance=importance,
        mess_reference=MessReference.from_array(data.predictors, climate.to_numpy()),
        extrapolation_reference=ExtrapolationReference.build(
            data.predictors,
            climate.to_numpy(),
            folds=spatial_fold_ids(pts, cfg.n_folds, cfg.seed),
            weights=[importance[p] for p in data.predictors],
            mop_percentage=cfg.mop_percentage,
        ),
        presence_envelope=envelope,
        n_presence=data.n_presence,
        n_background=data.n_background,
        background_method=data.background_method,
        extra={
            "bias_covariates": dict(data.bias_covariates),
            "bias_importance": {b: full_importance.get(b, 0.0) for b in data.bias_covariates},
            "weighted": w is not None,
            **extra,
        },
        algorithm=algorithm,
        threshold_rule=rule_label,
        thresholds=thresholds,
        response_curves=response_curves(
            model, data.presence_x, data.background_x, data.predictors, seed=cfg.seed
        ),
        presence_mean={p: float(np.nanmean(data.presence_x[p])) for p in data.predictors},
        cv_method=cv_method,
    )


def _permutation_importance(
    model: elapid.MaxentModel,
    x: pd.DataFrame,
    y: np.ndarray,
    predictors: Sequence[str],
    cfg: TrainingConfig,
) -> dict[str, float]:
    """Permutation importance (drop in training AUC), normalised to percent contribution."""
    np.random.seed(cfg.seed)  # elapid's permutation routine draws from the global RNG
    scores = np.asarray(
        model.permutation_importance_scores(x, y, n_repeats=cfg.importance_repeats, n_jobs=1)
    )
    mean = scores.mean(axis=1) if scores.ndim == 2 else scores
    mean = np.clip(mean, 0, None)
    total = mean.sum()
    pct = mean / total * 100 if total > 0 else np.zeros_like(mean)
    return {p: round(float(v), 2) for p, v in zip(predictors, pct, strict=True)}


# ---------------------------------------------------------------------------
# Ensemble (U3)
# ---------------------------------------------------------------------------
def _member_params(algorithm: str, fc: str, rm: float, cfg: TrainingConfig) -> dict[str, Any]:
    if algorithm == "maxent":
        return {"feature_classes": fc, "beta_multiplier": rm}
    if algorithm in ("rf", "brt"):
        return {"pseudo_absences_per_fit": "as many as presences", "repeats": cfg.pa_repeats}
    return {"weights": "total presence weight = total pseudo-absence weight"}


def train_ensemble(
    data: TrainingData, model_type: ModelType, cfg: TrainingConfig
) -> TrainingResult:
    _check_presences(data, cfg)
    members = list(cfg.ensemble_members)
    pa_members = [m for m in members if m != "maxent"]
    has_pa = data.absence_x is not None and len(data.absence_x) >= 10
    if pa_members and not has_pa:
        raise ValueError("The ensemble's presence–absence members need pseudo-absences")
    cols = data.model_columns
    x, y, _, _ = _stack_xy(data)
    pw = None if data.presence_weight is None else np.asarray(data.presence_weight, "float64")

    # The MaxEnt member keeps its usual Wallace tuning.
    tuned = train_maxent(data, model_type, cfg) if "maxent" in members else None
    fc = tuned.feature_classes if tuned else cfg.feature_classes
    rm = tuned.beta_multiplier if tuned else cfg.beta_multiplier

    plon, plat, k = data.presence_lon, data.presence_lat, cfg.n_folds
    pres_ids = spatial_block_ids(plon, plat, plon, plat, k, cfg.seed)
    bg_ids = spatial_block_ids(data.background_lon, data.background_lat, plon, plat, k, cfg.seed)
    pres_x = data.presence_x[cols].reset_index(drop=True)
    bg_x = data.background_x[cols].reset_index(drop=True)
    if has_pa:
        assert data.absence_x is not None
        assert data.absence_lon is not None and data.absence_lat is not None
        ab_x = data.absence_x[cols].reset_index(drop=True)
        ab_ids = spatial_block_ids(data.absence_lon, data.absence_lat, plon, plat, k, cfg.seed)
    else:  # MaxEnt-only ensemble: tested against the background
        ab_x, ab_ids = bg_x, bg_ids
    pa_x = pd.concat([pres_x, ab_x], ignore_index=True)
    pa_y = np.r_[np.ones(len(pres_x)), np.zeros(len(ab_x))]
    pa_w = None if pw is None else np.r_[pw, np.ones(len(ab_x))]

    def fit_member(
        algorithm: str, pres_mask: np.ndarray, bg_mask: np.ndarray, ab_mask: np.ndarray
    ) -> Any:
        if algorithm == "maxent":
            mx = pd.concat([pres_x[pres_mask], bg_x[bg_mask]], ignore_index=True)
            my = np.r_[np.ones(pres_mask.sum()), np.zeros(bg_mask.sum())]
            mw = None if pw is None else np.r_[pw[pres_mask], np.ones(bg_mask.sum())]
            model = _make_model(fc, rm, cfg.seed)
            _fit(model, mx, my, mw)
            return model
        mask = np.r_[pres_mask, ab_mask]
        return fit_pa_member(
            algorithm,
            pa_x[mask].reset_index(drop=True),
            pa_y[mask],
            cols,
            cfg.seed,
            cfg.pa_repeats,
            None if pa_w is None else pa_w[mask],
        )

    test_preds: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {m: {} for m in members}
    results: list[MemberResult] = []
    for algorithm in members:
        folds: list[FoldMetrics] = []
        for i in range(k):
            tr_p, te_p = pres_ids != i, pres_ids == i
            tr_a, te_a = ab_ids != i, ab_ids == i
            if tr_p.sum() < 5 or te_p.sum() < 1 or te_a.sum() < 1 or tr_a.sum() < 1:
                continue
            fold_model = fit_member(algorithm, tr_p, bg_ids != i, tr_a)
            p_te = np.asarray(fold_model.predict(pres_x[te_p]), "float64").ravel()
            a_te = np.asarray(fold_model.predict(ab_x[te_a]), "float64").ravel()
            test_preds[algorithm][i] = (p_te, a_te)
            folds.append(evaluate_fold(i, int(tr_p.sum()), p_te, a_te))
        if algorithm == "maxent" and tuned is not None:
            inner = tuned.model
            model = inner.model if isinstance(inner, BiasConditionedModel) else inner
        else:
            model = fit_member(
                algorithm,
                np.ones(len(pres_x), bool),
                np.ones(len(bg_x), bool),
                np.ones(len(ab_x), bool),
            )
        own = np.asarray(model.predict(pres_x), "float64").ravel()
        results.append(
            MemberResult(
                algorithm=algorithm,
                model=model,
                folds=folds,
                cv_summary=summarize_folds(folds) if folds else {},
                threshold=p10_threshold(own),
                params=_member_params(algorithm, fc, rm, cfg),
            )
        )
    weights, rule = ensemble_weights(results, cfg.ensemble_min_auc)
    total = sum(weights.values())
    for m in results:
        m.included = m.algorithm in weights
        m.weight = weights.get(m.algorithm, 0.0) / total if total else 0.0
        if not m.included:
            m.note = "left out: spatial-CV AUC below the ensemble minimum"
    ensemble = WeightedEnsemble(
        {m.algorithm: m.model for m in results if m.included},
        weights,
        {m.algorithm: m.threshold for m in results},
        cols,
    )
    # Ensemble CV: the final weights applied to each member's held-out predictions.
    ens_folds: list[FoldMetrics] = []
    for i in range(k):
        if not all(i in test_preds[a] for a in ensemble.weights):
            continue
        p_te = sum(ensemble.weights[a] * test_preds[a][i][0] for a in ensemble.weights)
        a_te = sum(ensemble.weights[a] * test_preds[a][i][1] for a in ensemble.weights)
        ens_folds.append(
            evaluate_fold(i, int((pres_ids != i).sum()), np.asarray(p_te), np.asarray(a_te))
        )
    if not ens_folds:
        raise ValueError("Spatial cross-validation produced no usable ensemble folds")
    full_importance = permutation_importance(
        ensemble.predict, x[cols], y, cols, cfg.importance_repeats, cfg.seed
    )
    test_set = "pseudo-absences" if has_pa else "background"
    return _finalize(
        data,
        model_type,
        cfg,
        ensemble,
        full_importance,
        ens_folds,
        tuned.tuning if tuned else [],
        fc,
        rm,
        algorithm="ensemble",
        cv_method=(
            f"spatial block CV (k-means blocks of the presences, k={k}); every member and the "
            f"ensemble tested on held-out presences vs held-out {test_set}"
        ),
        extra={
            "ensemble": {
                "members": [m.describe() for m in results],
                "weighting": rule,
                "cv_test_set": test_set,
                "reference": "Araújo & New 2007; Barbet-Massin et al. 2012; Valavi et al. 2021",
            }
        },
    )


# ---------------------------------------------------------------------------
# Integrated model (§4.4)
# ---------------------------------------------------------------------------
def train_integrated(
    data: TrainingData, model_type: ModelType, cfg: TrainingConfig
) -> TrainingResult:
    _check_presences(data, cfg)
    if (
        data.survey_x is None
        or data.survey_y is None
        or data.survey_lon is None
        or data.survey_lat is None
    ):
        raise ValueError("The integrated model needs presence–absence survey records")
    sy = np.asarray(data.survey_y, dtype="float64")
    n_abs, n_pres = int((sy == 0).sum()), int((sy == 1).sum())
    if n_abs < cfg.min_survey_absences or n_pres < 5:
        raise ValueError(
            f"The integrated model needs ≥ {cfg.min_survey_absences} survey absences and ≥ 5 "
            f"survey presences (have {n_abs} absences, {n_pres} presences)"
        )
    preds = data.predictors
    bias_cols = list(data.bias_covariates)
    cols = data.model_columns
    po_x = data.presence_x[cols].reset_index(drop=True)
    q_x = data.background_x[cols].reset_index(drop=True)
    sv_x = data.survey_x[preds].reset_index(drop=True)
    plon, plat, k = data.presence_lon, data.presence_lat, cfg.n_folds
    po_ids = spatial_block_ids(plon, plat, plon, plat, k, cfg.seed)
    q_ids = spatial_block_ids(data.background_lon, data.background_lat, plon, plat, k, cfg.seed)
    sv_ids = spatial_block_ids(data.survey_lon, data.survey_lat, plon, plat, k, cfg.seed)

    def fit(po_m: np.ndarray, q_m: np.ndarray, sv_m: np.ndarray) -> dict[str, Any]:
        return fit_integrated(
            IntegratedData(preds, po_x[po_m], q_x[q_m], sv_x[sv_m], sy[sv_m], bias_cols),
            l2=cfg.integrated_l2,
        )

    folds: list[FoldMetrics] = []
    survey_auc: list[float | None] = []
    for i in range(k):
        po_tr, q_tr, sv_tr = po_ids != i, q_ids != i, sv_ids != i
        if po_tr.sum() < 5 or (po_ids == i).sum() < 1 or (q_ids == i).sum() < 1:
            continue
        model = fit(po_tr, q_tr, sv_tr)["model"]
        p_te = model.predict(po_x[po_ids == i])
        b_te = model.predict(q_x[q_ids == i])
        folds.append(evaluate_fold(i, int(po_tr.sum()), p_te, b_te))
        te_y = sy[sv_ids == i]
        if (te_y == 1).any() and (te_y == 0).any():
            s_pred = model.predict(sv_x[sv_ids == i])
            survey_auc.append(round(auc(s_pred[te_y == 1], s_pred[te_y == 0]), 4))
        else:
            survey_auc.append(None)
    if not folds:
        raise ValueError("Spatial cross-validation produced no usable folds")
    fitted = fit(np.ones(len(po_x), bool), np.ones(len(q_x), bool), np.ones(len(sv_x), bool))
    model = fitted["model"]
    x, y, _, _ = _stack_xy(data)
    full_importance = permutation_importance(
        model.predict, x[preds], y, preds, cfg.importance_repeats, cfg.seed
    )
    valid_auc = [a for a in survey_auc if a is not None]
    return _finalize(
        data,
        model_type,
        cfg,
        model,
        full_importance,
        folds,
        [],
        "LQ (integrated)",
        cfg.integrated_l2,
        algorithm="integrated",
        cv_method=(
            f"spatial block CV (k-means blocks of the presences, k={k}): presence-only test "
            "presences vs background; survey AUC on held-out survey records"
        ),
        extra={
            "integrated": {
                **fitted["report"],
                "cv_survey_auc": survey_auc,
                "cv_survey_auc_mean": round(float(np.mean(valid_auc)), 4) if valid_auc else None,
                "presence_weights_used": False,
            }
        },
        conditioned=False,  # the integrated model drops its presence-only bias terms itself
    )


# ---------------------------------------------------------------------------
# Point process with an effort offset (B8) / shared bias (B9)
# ---------------------------------------------------------------------------
def train_ppm(data: TrainingData, model_type: ModelType, cfg: TrainingConfig) -> TrainingResult:
    _check_presences(data, cfg)
    preds = data.predictors
    fixed = {c: v for c, v in data.offset_coefficients.items() if c in data.bias_covariates}
    free = [c for c in data.bias_covariates if c not in fixed]
    cols = data.model_columns
    po_x = data.presence_x[cols].reset_index(drop=True)
    q_x = data.background_x[cols].reset_index(drop=True)
    plon, plat, k = data.presence_lon, data.presence_lat, cfg.n_folds
    po_ids = spatial_block_ids(plon, plat, plon, plat, k, cfg.seed)
    q_ids = spatial_block_ids(data.background_lon, data.background_lat, plon, plat, k, cfg.seed)

    def fit(po_m: np.ndarray, q_m: np.ndarray) -> dict[str, Any]:
        return fit_ppm(PpmData(preds, po_x[po_m], q_x[q_m], fixed, free), l2=cfg.ppm_l2)

    folds: list[FoldMetrics] = []
    for i in range(k):
        po_tr, q_tr = po_ids != i, q_ids != i
        if po_tr.sum() < 5 or (po_ids == i).sum() < 1 or (q_ids == i).sum() < 1:
            continue
        model = fit(po_tr, q_tr)["model"]
        folds.append(
            evaluate_fold(
                i,
                int(po_tr.sum()),
                model.predict(po_x[po_ids == i]),
                model.predict(q_x[q_ids == i]),
            )
        )
    if not folds:
        raise ValueError("Spatial cross-validation produced no usable folds")
    everything = np.ones(len(po_x), bool), np.ones(len(q_x), bool)
    fitted = fit(*everything)
    model = fitted["model"]
    x, y, _, _ = _stack_xy(data)
    full_importance = permutation_importance(
        model.predict, x[cols], y, cols, cfg.importance_repeats, cfg.seed
    )
    test = (
        offset_test(PpmData(preds, po_x, q_x, fixed, free), next(iter(fixed)), cfg.ppm_l2)
        if fixed
        else None
    )
    return _finalize(
        data,
        model_type,
        cfg,
        model,
        full_importance,
        folds,
        [],
        "LQ (point process)",
        cfg.ppm_l2,
        algorithm="ppm",
        cv_method=(
            f"spatial block CV (k-means blocks of the presences, k={k}): test presences vs "
            "test quadrature points, offset dropped"
        ),
        extra={
            "point_process": {
                **fitted["report"],
                "method": "Poisson point process, uniform quadrature (Berman & Turner 1992)",
                "reference": "Renner et al. 2015; Fithian et al. 2015 (shared bias)",
                "features": "linear + quadratic terms of standardised predictors",
                "bias_mode": cfg.ppm_bias,
                "l2_penalty": cfg.ppm_l2,
                "offset": (" + ".join(f"{v:g}·{c}" for c, v in fixed.items()) if fixed else "none"),
                "offset_test": test,
                "prediction": "1 − exp(−λ/λ̄) with the offset dropped (constant effort) and "
                "free bias covariates held at their conditioning value",
                "presence_weights_used": False,
            }
        },
    )
