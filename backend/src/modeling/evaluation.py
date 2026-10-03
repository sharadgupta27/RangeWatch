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


# ---------------------------------------------------------------------------
# Climate-data cross-check (same model fitted on two bioclim sources)
# ---------------------------------------------------------------------------
def predictor_agreement(primary: np.ndarray, alternative: np.ndarray) -> dict[str, float]:
    """Agreement of one predictor between two climate datasets at the same locations.

    `scale_ratio` (ratio of median absolute values) far from 1 flags a units mismatch rather
    than a genuine climatological difference."""
    a = np.asarray(primary, dtype="float64")
    b = np.asarray(alternative, dtype="float64")
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        nan = float("nan")
        return {"pearson_r": nan, "mean_diff": nan, "mean_abs_diff": nan, "scale_ratio": nan}
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    med_a = float(np.median(np.abs(a)))
    return {
        "pearson_r": r,
        "mean_diff": float(np.mean(b - a)),
        "mean_abs_diff": float(np.mean(np.abs(b - a))),
        "scale_ratio": float(np.median(np.abs(b)) / med_a) if med_a > 0 else float("nan"),
    }


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation of two suitability surfaces sampled at the same points."""
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    rho = spearmanr(a[ok], b[ok]).statistic
    return float(rho)


def binary_agreement(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Overall agreement and Cohen's kappa of two suitable/unsuitable classifications."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.size == 0:
        return {"agreement": float("nan"), "kappa": float("nan")}
    po = float(np.mean(a == b))
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"agreement": po, "kappa": float(kappa)}
