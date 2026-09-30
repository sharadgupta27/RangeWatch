"""Evaluation metric tests (AUC, TSS, CBI, MESS)."""

from __future__ import annotations

import numpy as np

from src.modeling.evaluation import (
    MessReference,
    auc,
    continuous_boyce_index,
    evaluate_fold,
    mess,
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
