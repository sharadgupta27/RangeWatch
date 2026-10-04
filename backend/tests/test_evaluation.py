"""Evaluation metric tests (AUC, TSS, CBI, MESS, exDet, MOP, Shape, AOA)."""

from __future__ import annotations

import pickle

import numpy as np
from scipy.spatial.distance import cdist

from src.modeling.evaluation import (
    EXTRAPOLATION_DIAGNOSTICS,
    ExtrapolationReference,
    MessReference,
    auc,
    continuous_boyce_index,
    evaluate_fold,
    extrapolation_diagnostics,
    mess,
    outlier_trimmed_max,
    p10_threshold,
    summarize_folds,
    tss,
)


def test_auc_and_tss_perfect_separation():
    pres, bg = np.array([0.8, 0.9, 0.95]), np.array([0.1, 0.2, 0.3])
    assert auc(pres, bg) == 1.0
    t, thr = tss(pres, bg)
    assert t == 1.0 and 0.3 < thr <= 0.8


def test_cbi_positive_for_informative_model_and_negative_for_inverted():
    rng = np.random.default_rng(0)
    bg = rng.uniform(0, 1, 5000)
    pres = rng.beta(5, 1.5, 500)  # presences concentrated at high suitability
    assert continuous_boyce_index(pres, bg) > 0.8
    assert continuous_boyce_index(1 - pres, bg) < -0.5


def test_cbi_degenerate_returns_nan():
    assert np.isnan(continuous_boyce_index(np.ones(5), np.ones(5)))


def test_p10_threshold():
    assert p10_threshold(np.arange(1, 101) / 100) == np.percentile(np.arange(1, 101) / 100, 10)


def test_mess_inside_positive_outside_negative_and_mod():
    ref = MessReference.from_array(
        ["t", "p"], np.c_[np.linspace(0, 10, 101), np.linspace(0, 100, 101)]
    )
    target = np.array(
        [
            [5.0, 50.0],  # centre of both distributions → ~100
            [12.0, 50.0],  # t above the max → negative, MoD = t
            [5.0, -20.0],  # p below the min → negative, MoD = p
            [np.nan, 1.0],
        ]
    )
    m, mod = mess(ref, target)
    assert m[0] > 90
    assert m[1] < 0 and mod[1] == 0
    assert m[2] < 0 and mod[2] == 1
    assert np.isnan(m[3]) and mod[3] == -1
    # Elith et al. 2010: (max - p) / (max - min) * 100 for values above the range
    np.testing.assert_allclose(m[1], (10 - 12) / 10 * 100, rtol=1e-6)


def test_summarize_folds():
    f1 = evaluate_fold(0, 10, np.array([0.9, 0.8]), np.array([0.1, 0.2, 0.3]))
    f2 = evaluate_fold(1, 10, np.array([0.6, 0.7]), np.array([0.5, 0.65, 0.2]))
    s = summarize_folds([f1, f2])
    assert s["auc_mean"] < 1.0 and s["auc_std"] > 0
    assert set(s) == {"auc_mean", "auc_std", "tss_mean", "tss_std", "cbi_mean", "cbi_std"}


# ---------------------------------------------------------------------------
# Advanced extrapolation diagnostics
# ---------------------------------------------------------------------------
def _correlated_reference(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two strongly correlated predictors + one independent one, in four spatial 'folds'."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    ref = np.c_[a, a + 0.1 * rng.normal(size=n), rng.normal(size=n)]
    folds = (ref[:, 2] > 0).astype(int) + 2 * (ref[:, 0] > 0).astype(int)
    return ref, folds


def _diagnose(ref_arr, folds, target, **kw):
    ref = ExtrapolationReference.build(["a", "b", "c"], ref_arr, folds, **kw)
    m, _ = mess(MessReference.from_array(["a", "b", "c"], ref_arr), target)
    return ref, m, extrapolation_diagnostics(ref, target, m)


def test_outlier_trimmed_max_is_cast_rule():
    v = np.r_[np.arange(1.0, 11.0), 100.0]
    q25, q75 = np.percentile(v, [25, 75])
    assert outlier_trimmed_max(v) == q75 + 1.5 * (q75 - q25)  # the outlier 100 is trimmed
    assert outlier_trimmed_max(np.array([1.0, 2.0, 3.0])) == 3.0  # capped at the max
    assert np.isnan(outlier_trimmed_max(np.array([np.nan])))


def test_exdet_nt1_nt2_against_formulas():
    ref_arr, folds = _correlated_reference()
    target = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 0.0]])
    ref, _, dg = _diagnose(ref_arr, folds, target)
    # NT1 (Mesgaran et al. 2014): sum of min(t - min, max - t, 0) / range
    lo, hi = ref_arr.min(axis=0), ref_arr.max(axis=0)
    ud = np.minimum(np.minimum(target[1] - lo, hi - target[1]), 0) / (hi - lo)
    np.testing.assert_allclose(dg.exdet[1], ud.sum(), rtol=1e-5)
    # NT2: squared Mahalanobis to the centroid / max squared Mahalanobis within the reference
    cov_inv = np.linalg.inv(np.cov(ref_arr, rowvar=False))
    c = ref_arr.mean(axis=0)
    d2 = np.einsum("ij,jk,ik->i", ref_arr - c, cov_inv, ref_arr - c)
    t0 = target[0] - c
    np.testing.assert_allclose(dg.exdet[0], t0 @ cov_inv @ t0 / d2.max(), rtol=1e-4)
    assert dg.exdet_mic[0] == -1 and dg.exdet_mic[1] in (0, 1)


def test_novel_combination_missed_by_mess_is_caught_by_the_other_diagnostics():
    ref_arr, folds = _correlated_reference(n=2000)
    # a and b are each within range but anti-correlated, against the training structure.
    target = np.array([[0.2, 0.2, 0.0], [1.5, -1.5, 0.0], [10.0, 10.0, 0.0]])
    _, m, dg = _diagnose(ref_arr, folds, target)
    assert m[1] > 0  # MESS: "similar"
    assert dg.exdet[1] > 1 and dg.combinatorial[1] and not dg.univariate[1]
    for name in ("exdet", "mop", "shape", "aoa"):
        assert dg.flags[name][1], name
        assert dg.flags[name][2] and not dg.flags[name][0], name
    assert not dg.flags["mess"][1] and dg.flags["mess"][2]
    assert dg.consensus.tolist() == [0, 4, 5]
    assert dg.univariate[2] and dg.exdet[2] < 0


def test_mop_shape_aoa_match_brute_force():
    ref_arr, folds = _correlated_reference(n=300)
    target = np.array([[0.3, 0.1, -0.5], [2.0, -1.0, 1.0]])
    ref, _, dg = _diagnose(ref_arr, folds, target, weights=[3.0, 1.0, 2.0], mop_percentage=5)
    mean, std = ref_arr.mean(axis=0), ref_arr.std(axis=0, ddof=1)
    # MOP: mean distance to the closest 5% of the (z-scored) reference
    k = round(len(ref_arr) * 0.05)
    d = np.sort(cdist((target - mean) / std, (ref_arr - mean) / std), axis=1)
    np.testing.assert_allclose(dg.mop, d[:, :k].mean(axis=1), rtol=1e-5)
    # Shape: Mahalanobis to nearest training point / mean Mahalanobis to the centroid × 100
    vi = np.linalg.inv(np.cov(ref_arr, rowvar=False))
    nearest = cdist(target, ref_arr, metric="mahalanobis", VI=vi).min(axis=1)
    base = cdist(ref_arr, mean[None, :], metric="mahalanobis", VI=vi).mean()
    np.testing.assert_allclose(dg.shape, nearest / base * 100, rtol=1e-4)
    # AOA: weighted z-scores, nearest distance / mean pairwise training distance
    w = np.array([3.0, 1.0, 2.0]) / 3.0
    zr, zt = (ref_arr - mean) / std * w, (target - mean) / std * w
    pair = cdist(zr, zr)
    mean_pair = pair.sum() / (len(zr) * (len(zr) - 1))
    np.testing.assert_allclose(dg.aoa_di, cdist(zt, zr).min(axis=1) / mean_pair, rtol=1e-5)
    # AOA threshold from the cross-validated training DI (nearest point in another fold)
    train_di = np.array([pair[i, folds != folds[i]].min() for i in range(len(zr))]) / mean_pair
    np.testing.assert_allclose(ref.aoa_threshold, outlier_trimmed_max(train_di), rtol=1e-6)


def test_aoa_ignores_zero_weight_predictors():
    ref_arr, folds = _correlated_reference()
    far_on_c = np.array([[0.0, 0.0, 50.0]])
    _, _, weighted = _diagnose(ref_arr, folds, far_on_c, weights=[1.0, 1.0, 0.0])
    _, _, equal = _diagnose(ref_arr, folds, far_on_c)
    assert not weighted.flags["aoa"][0] and equal.flags["aoa"][0]


def test_single_fold_reference_excludes_self_and_survives_pickle():
    ref_arr, _ = _correlated_reference(n=200)
    ref = ExtrapolationReference.build(["a", "b", "c"], ref_arr)
    assert ref.n_folds == 1 and ref.aoa_threshold > 0 and ref.shape_threshold > 0
    target = ref_arr[:5] + 0.01
    first = ref.aoa_di(target)
    clone = pickle.loads(pickle.dumps(ref))
    assert clone._trees == {}  # KD-trees are not pickled
    np.testing.assert_allclose(clone.aoa_di(target), first)
    desc = clone.describe()
    assert list(desc["thresholds"]) == list(EXTRAPOLATION_DIAGNOSTICS)
    assert desc["thresholds"]["exdet"]["value"] == 1.0
    assert desc["thresholds"]["aoa"]["value"] == round(ref.aoa_threshold, 6)


def test_reference_drops_incomplete_rows():
    ref_arr, folds = _correlated_reference(n=100)
    ref_arr[0, 1] = np.nan
    ref = ExtrapolationReference.build(["a", "b", "c"], ref_arr, folds)
    assert ref.n_reference == 99 and np.isfinite(ref.mins).all()
