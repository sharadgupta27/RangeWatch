"""MaxEnt training via `elapid.MaxentModel` with spatial block cross-validation.

Modeling defaults are reused from Wallace / ENMeval rather than re-derived: the candidate
feature classes L, LQ, H, LQH with regularization multipliers 1 and 2, selected by spatially
cross-validated performance. Model selection uses mean test CBI (presence-only appropriate),
with mean test AUC as tie-breaker. Evaluation metrics come from `evaluation.py`.
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
from src.modeling.evaluation import (
    FoldMetrics,
    MessReference,
    auc,
    evaluate_fold,
    p10_threshold,
    summarize_folds,
)

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

TRANSFERABILITY_CAVEAT = (
    "Native-range-only MaxEnt models show only moderate transferability to invaded ranges "
    "(mean AUC ≈ 0.7 in cross-continental evaluations). Suitability outside the training "
    "range is an exploratory screening signal, not a confident invasion forecast; interpret "
    "together with the MESS extrapolation layer."
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
    thin_km: float = 10.0
    min_presences: int = 15
    importance_repeats: int = 5
    seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_classes": self.feature_classes,
            "beta_multiplier": self.beta_multiplier,
            "tune": self.tune,
            "tuning_grid": [list(g) for g in self.tuning_grid],
            "n_folds": self.n_folds,
            "n_background": self.n_background,
            "background_buffer_km": self.background_buffer_km,
            "thin_km": self.thin_km,
            "min_presences": self.min_presences,
            "importance_repeats": self.importance_repeats,
            "seed": self.seed,
        }


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

    @property
    def n_presence(self) -> int:
        return len(self.presence_x)

    @property
    def n_background(self) -> int:
        return len(self.background_x)


@dataclass
class TrainingResult:
    model: elapid.MaxentModel
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
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence_label(self) -> str:
        return "standard" if self.model_type == "B" else "lower"


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


def _stack_xy(data: TrainingData) -> tuple[pd.DataFrame, np.ndarray, gpd.GeoSeries]:
    x = pd.concat(
        [data.presence_x[data.predictors], data.background_x[data.predictors]], ignore_index=True
    )
    y = np.r_[np.ones(data.n_presence), np.zeros(data.n_background)]
    pts = gpd.GeoSeries(
        gpd.points_from_xy(
            np.r_[data.presence_lon, data.background_lon],
            np.r_[data.presence_lat, data.background_lat],
        ),
        crs="EPSG:4326",
    )
    return x, y, pts


def spatial_block_cv(
    x: pd.DataFrame,
    y: np.ndarray,
    points: gpd.GeoSeries,
    feature_classes: str,
    beta: float,
    n_folds: int,
    seed: int,
) -> list[FoldMetrics]:
    """Spatial k-fold CV: folds are geographic clusters (elapid.GeographicKFold), never random."""
    folds: list[FoldMetrics] = []
    splitter = elapid.GeographicKFold(n_splits=n_folds, random_state=seed)
    for i, (tr, te) in enumerate(splitter.split(points)):
        y_tr, y_te = y[tr], y[te]
        if y_tr.sum() < 5 or y_te.sum() < 1 or (1 - y_te).sum() < 1:
            log.debug("Skipping degenerate spatial fold %d", i)
            continue
        model = _make_model(feature_classes, beta, seed)
        model.fit(x.iloc[tr], y_tr)
        pred = np.asarray(model.predict(x.iloc[te]), dtype="float64").ravel()
        folds.append(evaluate_fold(i, int(y_tr.sum()), pred[y_te == 1], pred[y_te == 0]))
    return folds


def train_maxent(data: TrainingData, model_type: ModelType, cfg: TrainingConfig) -> TrainingResult:
    if data.n_presence < cfg.min_presences:
        raise ValueError(
            f"Only {data.n_presence} presences after thinning (minimum {cfg.min_presences})"
        )
    x, y, pts = _stack_xy(data)

    grid = cfg.tuning_grid if cfg.tune else ((cfg.feature_classes, cfg.beta_multiplier),)
    tuning: list[dict[str, Any]] = []
    best: tuple[float, float] = (-np.inf, -np.inf)
    best_cfg = grid[0]
    best_folds: list[FoldMetrics] = []
    for fc, rm in grid:
        folds = spatial_block_cv(x, y, pts, fc, rm, cfg.n_folds, cfg.seed)
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
    model = _make_model(fc, rm, cfg.seed)
    model.fit(x, y)
    train_pred = np.asarray(model.predict(x), dtype="float64").ravel()
    pres_pred, bg_pred = train_pred[y == 1], train_pred[y == 0]

    importance = _permutation_importance(model, x, y, data.predictors, cfg)
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
        feature_classes=fc,
        beta_multiplier=rm,
        folds=best_folds,
        cv_summary=summarize_folds(best_folds),
        tuning=tuning,
        threshold=p10_threshold(pres_pred),
        train_auc=auc(pres_pred, bg_pred),
        importance=importance,
        mess_reference=MessReference.from_array(data.predictors, x.to_numpy()),
        presence_envelope=envelope,
        n_presence=data.n_presence,
        n_background=data.n_background,
        background_method=data.background_method,
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
