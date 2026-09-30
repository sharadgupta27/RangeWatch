"""Model evaluation metrics: AUC, TSS, Continuous Boyce Index (CBI) and MESS.

Every evaluation metric lives here (CLAUDE.md) — maxent_trainer only calls into this module.
Presence-background data has no true absences, so AUC/TSS treat background as pseudo-absence;
CBI is the presence-only-appropriate metric and must always be reported alongside them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve


def auc(presence_pred: np.ndarray, background_pred: np.ndarray) -> float:
    y = np.r_[np.ones(len(presence_pred)), np.zeros(len(background_pred))]
    return float(roc_auc_score(y, np.r_[presence_pred, background_pred]))


def tss(presence_pred: np.ndarray, background_pred: np.ndarray) -> tuple[float, float]:
    """Maximum True Skill Statistic (sensitivity + specificity - 1) and its threshold."""
    y = np.r_[np.ones(len(presence_pred)), np.zeros(len(background_pred))]
    fpr, tpr, thr = roc_curve(y, np.r_[presence_pred, background_pred])
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(j[i]), float(min(thr[i], 1.0))


def continuous_boyce_index(
    presence_pred: np.ndarray,
    background_pred: np.ndarray,
    n_windows: int = 101,
    window_width: float = 0.1,
) -> float:
    """Continuous Boyce Index (Hirzel et al. 2006), moving-window version as in ecospat.boyce.

    For each window over the suitability range: F = P/E where P is the share of presence
    predictions and E the share of background predictions falling in the window. CBI is the
    Spearman correlation between F and the window centre. Ranges -1..1; > 0 means the model
    predicts presences better than random.
    """
    pres = np.asarray(presence_pred, dtype="float64")
    bg = np.asarray(background_pred, dtype="float64")
    lo = min(pres.min(), bg.min())
    hi = max(pres.max(), bg.max())
    width = (hi - lo) * window_width
    if width <= 0:
        return float("nan")
    starts = np.linspace(lo, hi - width, n_windows)
    f_ratio: list[float] = []
    centres: list[float] = []
    for s in starts:
        end = s + width
        p = np.mean((pres >= s) & (pres <= end))
        exp = np.mean((bg >= s) & (bg <= end))
        if exp > 0:
            f_ratio.append(p / exp)
            centres.append(s + width / 2)
    f_arr = np.asarray(f_ratio)
    c_arr = np.asarray(centres)
    # ecospat removes successive duplicated F values (flat plateaus inflate the rank corr).
    if f_arr.size > 1:
        keep = np.r_[True, np.diff(f_arr) != 0]
        f_arr, c_arr = f_arr[keep], c_arr[keep]
    if f_arr.size < 3:
        return float("nan")
    rho = spearmanr(f_arr, c_arr).statistic
    return float(rho)


def p10_threshold(train_presence_pred: np.ndarray) -> float:
    """10th-percentile training-presence threshold (standard for binarising Maxent output)."""
    return float(np.nanpercentile(train_presence_pred, 10))


@dataclass(frozen=True)
class FoldMetrics:
    fold: int
    n_train_presence: int
    n_test_presence: int
    n_test_background: int
    auc: float
    tss: float
    cbi: float


def evaluate_fold(
    fold: int,
    n_train_presence: int,
    test_presence_pred: np.ndarray,
    test_background_pred: np.ndarray,
) -> FoldMetrics:
    t, _ = tss(test_presence_pred, test_background_pred)
    return FoldMetrics(
        fold=fold,
        n_train_presence=n_train_presence,
        n_test_presence=len(test_presence_pred),
        n_test_background=len(test_background_pred),
        auc=auc(test_presence_pred, test_background_pred),
        tss=t,
        cbi=continuous_boyce_index(test_presence_pred, test_background_pred),
    )


def summarize_folds(folds: Sequence[FoldMetrics]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ("auc", "tss", "cbi"):
        vals = np.array([getattr(f, name) for f in folds], dtype="float64")
        out[f"{name}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        out[f"{name}_std"] = float(np.nanstd(vals)) if np.isfinite(vals).any() else float("nan")
    return out


# ---------------------------------------------------------------------------
# MESS — Multivariate Environmental Similarity Surface (Elith et al. 2010)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MessReference:
    """Sorted per-variable reference distributions (training presence + background)."""

    variables: tuple[str, ...]
    sorted_values: tuple[np.ndarray, ...]

    @classmethod
    def from_array(cls, variables: Sequence[str], reference: np.ndarray) -> MessReference:
        ref = np.asarray(reference, dtype="float64")
        ref = ref[~np.isnan(ref).any(axis=1)]
        return cls(tuple(variables), tuple(np.sort(ref[:, j]) for j in range(ref.shape[1])))


def mess(reference: MessReference, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MESS value and most-dissimilar-variable index for each target row.

    Negative MESS = at least one variable outside the training range (extrapolation).
    Rows with NaN inputs return NaN / -1.
    """
    tgt = np.asarray(target, dtype="float64")
    n, p = tgt.shape
    sims = np.empty((n, p), dtype="float64")
    for j, ref in enumerate(reference.sorted_values):
        t = tgt[:, j]
        mn, mx = ref[0], ref[-1]
        rng = (mx - mn) or 1e-12
        f = np.searchsorted(ref, t, side="left") / ref.size * 100.0
        sim = np.where(
            f == 0,
            (t - mn) / rng * 100.0,
            np.where(
                f <= 50,
                2.0 * f,
                np.where(f < 100, 2.0 * (100.0 - f), (mx - t) / rng * 100.0),
            ),
        )
        sims[:, j] = sim
    nan_rows = np.isnan(tgt).any(axis=1)
    sims[nan_rows] = np.nan
    out = np.full(n, np.nan)
    mod = np.full(n, -1, dtype="int16")
    ok = ~nan_rows
    if ok.any():
        out[ok] = sims[ok].min(axis=1)
        mod[ok] = sims[ok].argmin(axis=1)
    return out.astype("float32"), mod
