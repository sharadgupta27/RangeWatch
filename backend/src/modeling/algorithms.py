"""Presence–(pseudo-)absence algorithms and the CV-weighted ensemble (U3).

elapid provides MaxEnt, a niche-envelope model and an *unweighted* `EnsembleModel` (mean /
median / mode of member predictions); it has no GLM, GAM, random forest or boosted trees. The
members below are therefore scikit-learn estimators (already a dependency of elapid, version
logged per model version), each wrapped so it exposes the same `predict(x) -> 0..1` as an
elapid model. No new modelling library is introduced.

Pseudo-absence numbers and weights follow Barbet-Massin et al. (2012):
  * GLM / GAM: many pseudo-absences, weighted so total presence weight = total absence weight;
  * RF / BRT: as many pseudo-absences as presences, repeated `pa_repeats` times, averaged
    (down-sampled / balanced RF, Valavi et al. 2021).

Every member is evaluated on the same spatial blocks (`spatial_block_ids`: k-means clusters of
the presences, every other point assigned to the nearest cluster), and the ensemble is the
mean of the members weighted by (CV AUC − 0.5), members below `min_auc` left out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

# Algorithms a training run can combine. "maxent" is the presence–background elapid model.
ALGORITHMS = ("maxent", "glm", "gam", "rf", "brt")
PA_ALGORITHMS = ("glm", "gam", "rf", "brt")
ALGORITHM_LABELS = {
    "maxent": "MaxEnt (elapid, presence–background)",
    "glm": "GLM (logistic, linear + quadratic terms)",
    "gam": "GAM (penalised cubic-spline logistic regression)",
    "rf": "Random forest (down-sampled, balanced)",
    "brt": "Boosted regression trees (histogram gradient boosting, balanced)",
}


class ProbabilityModel:
    """Adapter: a fitted scikit-learn classifier with `predict(x) -> P(presence)`."""

    def __init__(self, estimator: Any, columns: Sequence[str]) -> None:
        self.estimator = estimator
        self.columns = list(columns)

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x, columns=self.columns)
        return np.asarray(self.estimator.predict_proba(frame[self.columns].to_numpy())[:, 1])


class AveragedModel:
    """Mean prediction of models fitted on repeated balanced draws (RF / BRT)."""

    def __init__(self, models: Sequence[Any]) -> None:
        self.models = list(models)

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.mean([np.asarray(m.predict(x), dtype="float64") for m in self.models], axis=0)


class WeightedEnsemble:
    """Weighted mean of member predictions. `predict_members` returns one column per member
    (used for the ensemble-agreement uncertainty layer)."""

    def __init__(
        self,
        members: dict[str, Any],
        weights: dict[str, float],
        thresholds: dict[str, float],
        columns: Sequence[str],
    ) -> None:
        self.members = dict(members)
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
        self.thresholds = dict(thresholds)
        self.columns = list(columns)

    def predict_members(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x, columns=self.columns)
        return np.column_stack(
            [
                np.asarray(self.members[k].predict(frame), dtype="float64").ravel()
                for k in self.weights
            ]
        )

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        preds = self.predict_members(x)
        return preds @ np.array(list(self.weights.values()))

    def member_agreement(self, x: pd.DataFrame | np.ndarray, threshold: float) -> np.ndarray:
        """Share of members (weighted) whose own binary call agrees with the ensemble's."""
        preds = self.predict_members(x)
        w = np.array(list(self.weights.values()))
        member_bin = preds >= np.array([self.thresholds[k] for k in self.weights])
        ens_bin = (preds @ w) >= threshold
        return (member_bin == ens_bin[:, None]).astype("float64") @ w


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    """Total presence weight = total absence weight (= n / 2)."""
    n1, n0 = max(int(y.sum()), 1), max(int((1 - y).sum()), 1)
    return np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * n0))


def make_pa_estimator(algorithm: str, seed: int) -> Any:
    if algorithm == "glm":
        return make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            LogisticRegression(C=1.0, max_iter=5000),
        )
    if algorithm == "gam":
        return make_pipeline(
            StandardScaler(),
            SplineTransformer(n_knots=5, degree=3, extrapolation="linear"),
            LogisticRegression(C=0.5, max_iter=5000),
        )
    if algorithm == "rf":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=1
        )
    if algorithm == "brt":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=8,
            l2_regularization=1.0,
            random_state=seed,
        )
    raise ValueError(f"Unknown presence–absence algorithm {algorithm!r}")


def fit_pa_member(
    algorithm: str,
    x: pd.DataFrame,
    y: np.ndarray,
    columns: Sequence[str],
    seed: int,
    repeats: int,
    presence_weight: np.ndarray | None = None,
) -> Any:
    """Fit one presence–absence member following Barbet-Massin et al. (2012)."""
    xv = x[list(columns)].to_numpy(dtype="float64")
    obs = np.ones(len(y)) if presence_weight is None else presence_weight
    if algorithm in ("glm", "gam"):
        est = make_pa_estimator(algorithm, seed)
        w = _balanced_weights(y) * obs
        est.fit(xv, y, **{f"{est.steps[-1][0]}__sample_weight": w})
        return ProbabilityModel(est, columns)
    rng = np.random.default_rng(seed)
    pres, absn = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    models = []
    for r in range(max(1, repeats)):
        take = rng.choice(absn, size=min(len(pres), len(absn)), replace=False)
        idx = np.r_[pres, take]
        est = make_pa_estimator(algorithm, seed + r)
        est.fit(xv[idx], y[idx], sample_weight=obs[idx])
        models.append(ProbabilityModel(est, columns))
    return AveragedModel(models)


def spatial_block_ids(
    lon: np.ndarray,
    lat: np.ndarray,
    anchor_lon: np.ndarray,
    anchor_lat: np.ndarray,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Spatial blocks shared by every member: k-means clusters of the anchor points (the
    presences, on the unit sphere so the dateline is no edge); each point joins the nearest
    cluster. Same idea as elapid.GeographicKFold, but stable across data sets."""

    def xyz(lo: np.ndarray, la: np.ndarray) -> np.ndarray:
        lo_r, la_r = np.deg2rad(np.asarray(lo, "float64")), np.deg2rad(np.asarray(la, "float64"))
        return np.column_stack(
            [np.cos(la_r) * np.cos(lo_r), np.cos(la_r) * np.sin(lo_r), np.sin(la_r)]
        )

    anchors = xyz(anchor_lon, anchor_lat)
    k = max(2, min(n_folds, len(anchors)))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(anchors)
    return km.predict(xyz(lon, lat))


@dataclass
class MemberResult:
    algorithm: str
    model: Any
    folds: list[Any]
    cv_summary: dict[str, float]
    threshold: float
    included: bool = True
    weight: float = 0.0
    note: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        def finite(v: Any) -> Any:
            return None if isinstance(v, float) and not np.isfinite(v) else v

        return {
            "algorithm": self.algorithm,
            "label": ALGORITHM_LABELS[self.algorithm],
            "included": self.included,
            "weight": round(self.weight, 4),
            "threshold": round(self.threshold, 6),
            **{k: finite(round(v, 4)) for k, v in self.cv_summary.items()},
            "n_cv_folds": len(self.folds),
            "params": self.params,
            **({"note": self.note} if self.note else {}),
        }


def ensemble_weights(
    members: Sequence[MemberResult], min_auc: float
) -> tuple[dict[str, float], str]:
    """Weights ∝ (CV AUC − 0.5) of members with AUC ≥ `min_auc`; equal weights for all
    members if none qualifies (stated in the returned rule)."""
    scores = {m.algorithm: m.cv_summary.get("auc_mean", float("nan")) for m in members}
    ok = {k: v - 0.5 for k, v in scores.items() if np.isfinite(v) and v >= min_auc}
    if ok:
        return ok, f"weight ∝ (spatial-CV AUC − 0.5); members with AUC < {min_auc:g} left out"
    return (
        {m.algorithm: 1.0 for m in members},
        f"no member reached spatial-CV AUC {min_auc:g}: equal weights for all members",
    )
