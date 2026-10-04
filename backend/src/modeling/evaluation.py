"""Model evaluation metrics: AUC, TSS, Continuous Boyce Index (CBI) and extrapolation
diagnostics (MESS, exDet, MOP, Shape, AOA).

Every evaluation metric lives here (CLAUDE.md) — maxent_trainer only calls into this module.
Presence-background data has no true absences, so AUC/TSS treat background as pseudo-absence;
CBI is the presence-only-appropriate metric and must always be reported alongside them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
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
# Advanced extrapolation diagnostics: exDet, MOP, Shape, AOA
# ---------------------------------------------------------------------------
# MESS asks only "is some predictor outside its training range?". These diagnostics add what
# it misses: novel *combinations* of in-range values (exDet NT2), distance to the closest
# analogous training conditions (MOP, Shape) and distance in the importance-weighted space the
# model actually relies on (AOA). All share the MESS reference set (training presences +
# background), so their verdicts are comparable cell by cell.
EXTRAPOLATION_DIAGNOSTICS: tuple[str, ...] = ("mess", "exdet", "mop", "shape", "aoa")
DIAGNOSTIC_INFO: dict[str, dict[str, str]] = {
    "mess": {
        "label": "MESS",
        "name": "Multivariate Environmental Similarity Surface",
        "reference": "Elith et al. 2010",
    },
    "exdet": {
        "label": "exDet",
        "name": "Extrapolation Detection (NT1 univariate / NT2 combinatorial novelty)",
        "reference": "Mesgaran et al. 2014",
    },
    "mop": {
        "label": "MOP",
        "name": "Mobility-Oriented Parity",
        "reference": "Owens et al. 2013; Cobos et al. 2024",
    },
    "shape": {
        "label": "Shape",
        "name": "Shape extrapolation degree",
        "reference": "Velazco et al. 2024",
    },
    "aoa": {
        "label": "AOA",
        "name": "Area of Applicability (dissimilarity index)",
        "reference": "Meyer & Pebesma 2021",
    },
}
CV_THRESHOLD_RULE = (
    "outlier-trimmed maximum (min(Q75 + 1.5·IQR, max)) of the training points' values "
    "against the other spatial CV folds"
)
_KNN_CHUNK = 50_000


def outlier_trimmed_max(values: np.ndarray) -> float:
    """AOA threshold rule (CAST `.di_threshold`): Q75 + 1.5·IQR of the training values, capped
    at their maximum — i.e. the largest training value that is not an outlier."""
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    q25, q75 = np.percentile(v, [25, 75])
    return float(min(q75 + 1.5 * (q75 - q25), v.max()))


def _whitener(cov: np.ndarray) -> np.ndarray:
    """W such that ||(x - y) @ W|| is the Mahalanobis distance under `cov` (pseudo-inverse, so
    a constant or perfectly collinear predictor cannot make it singular)."""
    vals, vecs = np.linalg.eigh(np.atleast_2d(cov))
    keep = vals > max(float(vals.max()), 0.0) * 1e-10
    return vecs[:, keep] / np.sqrt(vals[keep])


def _cross_fold_knn(points: np.ndarray, folds: np.ndarray, k: int) -> np.ndarray:
    """Distances from each point to its k nearest neighbours in the *other* CV folds (CAST's
    cross-validated training DI). With a single fold the point itself is excluded instead."""
    n = len(points)
    out = np.full((n, k), np.nan)
    groups = np.unique(folds)
    if groups.size < 2:
        kk = min(k + 1, n)
        d, _ = cKDTree(points).query(points, k=kk)
        d = np.asarray(d).reshape(n, kk)[:, 1:]
        out[:, : d.shape[1]] = d
        return out
    for g in groups:
        test = folds == g
        other = points[~test]
        kk = min(k, len(other))
        d, _ = cKDTree(other).query(points[test], k=kk)
        out[test, :kk] = np.asarray(d).reshape(int(test.sum()), kk)
    return out


def _mean_pairwise_distance(points: np.ndarray, chunk: int = 512) -> float:
    """Mean Euclidean distance over all pairs of distinct points (CAST's trainDist_avrgmean)."""
    n = len(points)
    if n < 2:
        return float("nan")
    total = sum(float(cdist(points[s : s + chunk], points).sum()) for s in range(0, n, chunk))
    return total / (n * (n - 1))


def _knn(tree: cKDTree, x: np.ndarray, k: int) -> np.ndarray:
    """(n, k) nearest-neighbour distances, queried in chunks to bound memory."""
    k = min(k, tree.n)
    out = np.empty((len(x), k))
    for s in range(0, len(x), _KNN_CHUNK):
        d, _ = tree.query(x[s : s + _KNN_CHUNK], k=k, workers=-1)
        out[s : s + _KNN_CHUNK] = np.asarray(d).reshape(-1, k)
    return out


@dataclass(eq=False)
class ExtrapolationReference:
    """What exDet, MOP, Shape and AOA need from the training data, precomputed once per model
    version and pickled into model.pkl next to the MESS reference.

    exDet keeps its published cut-offs (NT1 < 0, NT2 > 1). MOP and Shape publish no fixed
    threshold, so both use the AOA rule like the DI does: every training point is scored
    against the training points of the *other spatial CV folds*, and the threshold is the
    outlier-trimmed maximum of those scores. A cell is flagged when it lies farther from the
    training data than one spatial block's training data lie from the rest.
    """

    variables: tuple[str, ...]
    n_reference: int
    n_folds: int
    # exDet (its Mahalanobis geometry is shared with Shape)
    mins: np.ndarray
    maxs: np.ndarray
    centroid: np.ndarray
    whitener: np.ndarray
    drop_whiteners: tuple[np.ndarray, ...]
    nt2_max: float
    # z-score scale shared by MOP and AOA
    scale_std: np.ndarray
    # MOP
    mop_percentage: float
    mop_k: int
    mop_reference: np.ndarray
    mop_threshold: float
    # Shape
    shape_reference: np.ndarray
    shape_base: float
    shape_threshold: float
    # AOA
    aoa_weights: np.ndarray
    aoa_reference: np.ndarray
    aoa_mean_distance: float
    aoa_threshold: float
    _trees: dict[str, cKDTree] = field(default_factory=dict, repr=False)

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_trees": {}}  # KD-trees are rebuilt lazily after loading

    @classmethod
    def build(
        cls,
        variables: Sequence[str],
        reference: np.ndarray,
        folds: np.ndarray | None = None,
        weights: Sequence[float] | None = None,
        mop_percentage: float = 1.0,
    ) -> ExtrapolationReference:
        """`reference`: training presences + background (rows) × predictors; `folds`: spatial
        CV fold of each row; `weights`: predictor importance for the AOA distance."""
        ref = np.asarray(reference, dtype="float64")
        ok = np.isfinite(ref).all(axis=1)
        ref = ref[ok]
        n, p = ref.shape
        if n < 3:
            raise ValueError("The extrapolation reference needs at least 3 complete rows")
        fold_ids = np.zeros(n, dtype=int) if folds is None else np.asarray(folds)[ok]

        # exDet (Mesgaran et al. 2014; dsmextra): NT2 = squared Mahalanobis distance to the
        # reference centroid relative to the largest one within the reference.
        centroid = ref.mean(axis=0)
        cov = np.atleast_2d(np.cov(ref, rowvar=False))
        whitener = _whitener(cov)
        d_centroid = np.sqrt((((ref - centroid) @ whitener) ** 2).sum(axis=1))
        drop = (
            tuple(_whitener(np.delete(np.delete(cov, j, 0), j, 1)) for j in range(p))
            if p > 1
            else ()
        )
        std = ref.std(axis=0, ddof=1)
        std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        z = (ref - centroid) / std

        # MOP (Owens et al. 2013; mop R package): mean Euclidean distance (z-scored, as the
        # package recommends) to the closest `mop_percentage`% of reference points.
        k = max(1, int(round(n * mop_percentage / 100)))
        mop_train = np.nanmean(_cross_fold_knn(z, fold_ids, k), axis=1)

        # Shape (Velazco et al. 2024; flexsdm::extra_eval): Mahalanobis distance to the nearest
        # training point / mean Mahalanobis distance of training points to their centroid × 100.
        shape_ref = ref @ whitener
        shape_base = float(d_centroid.mean()) or float("nan")
        shape_train = _cross_fold_knn(shape_ref, fold_ids, 1)[:, 0] / shape_base * 100

        # AOA (Meyer & Pebesma 2021; CAST::trainDI): importance-weighted z-scores; DI = distance
        # to the nearest training point / mean pairwise distance between training points.
        w = (
            np.ones(p)
            if weights is None
            else np.clip(np.asarray(weights, dtype="float64"), 0, None)
        )
        if w.shape != (p,) or not np.isfinite(w).all() or w.max() <= 0:
            w = np.ones(p)
        w = w / w.max()
        aoa_ref = z * w
        mean_dist = _mean_pairwise_distance(aoa_ref) or float("nan")
        train_di = _cross_fold_knn(aoa_ref, fold_ids, 1)[:, 0] / mean_dist

        return cls(
            variables=tuple(variables),
            n_reference=n,
            n_folds=int(np.unique(fold_ids).size),
            mins=ref.min(axis=0),
            maxs=ref.max(axis=0),
            centroid=centroid,
            whitener=whitener,
            drop_whiteners=drop,
            nt2_max=float((d_centroid**2).max()) or float("nan"),
            scale_std=std,
            mop_percentage=float(mop_percentage),
            mop_k=k,
            mop_reference=z,
            mop_threshold=outlier_trimmed_max(mop_train),
            shape_reference=shape_ref,
            shape_base=shape_base,
            shape_threshold=outlier_trimmed_max(shape_train),
            aoa_weights=w,
            aoa_reference=aoa_ref,
            aoa_mean_distance=mean_dist,
            aoa_threshold=outlier_trimmed_max(train_di),
        )

    def _tree(self, name: str) -> cKDTree:
        if name not in self._trees:
            self._trees[name] = cKDTree(getattr(self, f"{name}_reference"))
        return self._trees[name]

    # ----------------------------------------------------------- per-cell metrics
    def nt1(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """exDet NT1 (≤ 0; < 0 = univariate novelty) and the most-deviating predictor index."""
        rng = np.where(self.maxs > self.mins, self.maxs - self.mins, 1e-12)
        ud = np.minimum(np.minimum(x - self.mins, self.maxs - x), 0.0) / rng
        return ud.sum(axis=1), ud.argmin(axis=1)

    def nt2(self, x: np.ndarray) -> np.ndarray:
        """exDet NT2 (≥ 0; > 1 = combination of values never seen together in training)."""
        return (((x - self.centroid) @ self.whitener) ** 2).sum(axis=1) / self.nt2_max

    def nt2_mic(self, x: np.ndarray) -> np.ndarray:
        """Most influential covariate for NT2: the predictor whose removal reduces the
        Mahalanobis distance most (as in the ExDet tool / dsmextra)."""
        if not self.drop_whiteners:
            return np.zeros(len(x), dtype="int64")
        xc = x - self.centroid
        d_full = ((xc @ self.whitener) ** 2).sum(axis=1)
        reduction = np.column_stack(
            [
                d_full - ((np.delete(xc, j, axis=1) @ w) ** 2).sum(axis=1)
                for j, w in enumerate(self.drop_whiteners)
            ]
        )
        return reduction.argmax(axis=1)

    def mop(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.centroid) / self.scale_std
        return _knn(self._tree("mop"), z, self.mop_k).mean(axis=1)

    def shape(self, x: np.ndarray) -> np.ndarray:
        return _knn(self._tree("shape"), x @ self.whitener, 1)[:, 0] / self.shape_base * 100

    def aoa_di(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.centroid) / self.scale_std * self.aoa_weights
        return _knn(self._tree("aoa"), z, 1)[:, 0] / self.aoa_mean_distance

    def describe(self) -> dict[str, Any]:
        """Configuration and thresholds, logged with the model version and shown in the UI."""
        thresholds = {
            "mess": (0.0, "MESS < 0", "a predictor lies outside its training range"),
            "exdet": (
                1.0,
                "NT1 < 0 or NT2 > 1",
                "published cut-offs: NT1 < 0 univariate, NT2 > 1 combinatorial novelty",
            ),
            "mop": (self.mop_threshold, "MOP distance > threshold", CV_THRESHOLD_RULE),
            "shape": (self.shape_threshold, "Shape > threshold", CV_THRESHOLD_RULE),
            "aoa": (
                self.aoa_threshold,
                "DI > threshold (outside the AOA)",
                CV_THRESHOLD_RULE + " (Meyer & Pebesma 2021)",
            ),
        }
        return {
            "reference_set": "training presences + background (as for MESS)",
            "n_reference": self.n_reference,
            "n_cv_folds": self.n_folds,
            "mop_percentage": self.mop_percentage,
            "mop_k": self.mop_k,
            "aoa_weights": {
                v: round(float(w), 4) for v, w in zip(self.variables, self.aoa_weights, strict=True)
            },
            "thresholds": {
                key: {
                    "value": _finite_or_none(val),
                    "flag_rule": flag,
                    "rule": rule,
                    **DIAGNOSTIC_INFO[key],
                }
                for key, (val, flag, rule) in thresholds.items()
            },
        }


def _finite_or_none(v: float) -> float | None:
    return round(float(v), 6) if np.isfinite(v) else None


@dataclass
class ExtrapolationDiagnostics:
    """Per-cell diagnostics for a batch of complete target rows."""

    exdet: np.ndarray  # NT1 where < 0 (univariate novelty), otherwise NT2 (> 1 = combinatorial)
    exdet_mic: np.ndarray  # most influential covariate index; -1 where exDet finds no novelty
    mop: np.ndarray
    shape: np.ndarray
    aoa_di: np.ndarray
    flags: dict[str, np.ndarray]  # per diagnostic: True = extrapolation by that method's rule

    @property
    def univariate(self) -> np.ndarray:
        return self.exdet < 0

    @property
    def combinatorial(self) -> np.ndarray:
        return self.exdet > 1

    @property
    def consensus(self) -> np.ndarray:
        """Number of the five diagnostics that flag each cell as extrapolation (0–5)."""
        return np.sum([self.flags[k] for k in EXTRAPOLATION_DIAGNOSTICS], axis=0).astype("uint8")


def extrapolation_diagnostics(
    reference: ExtrapolationReference, target: np.ndarray, mess_values: np.ndarray
) -> ExtrapolationDiagnostics:
    """exDet, MOP, Shape and AOA for complete target rows, plus every method's verdict (with
    the MESS values that `mess` computed for the same rows)."""
    x = np.asarray(target, dtype="float64")
    nt1, mic1 = reference.nt1(x)
    nt2 = reference.nt2(x)
    univariate = nt1 < 0
    combinatorial = ~univariate & (nt2 > 1)
    mic = np.full(len(x), -1, dtype="int16")
    mic[univariate] = mic1[univariate]
    if combinatorial.any():
        mic[combinatorial] = reference.nt2_mic(x[combinatorial])
    mop_d = reference.mop(x)
    shape_v = reference.shape(x)
    di = reference.aoa_di(x)
    return ExtrapolationDiagnostics(
        exdet=np.where(univariate, nt1, nt2).astype("float32"),
        exdet_mic=mic,
        mop=mop_d.astype("float32"),
        shape=shape_v.astype("float32"),
        aoa_di=di.astype("float32"),
        flags={
            "mess": np.asarray(mess_values) < 0,
            "exdet": univariate | combinatorial,
            "mop": mop_d > reference.mop_threshold,
            "shape": shape_v > reference.shape_threshold,
            "aoa": di > reference.aoa_threshold,
        },
    )


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
