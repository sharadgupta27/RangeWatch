"""Unit tests of the priority-2/3 modelling pieces: thresholds, response curves, ensemble
members, the integrated model, pseudo-absence samplers and the annotation rules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.domain import RecordAnnotation
from src.features.annotations import (
    apply_exclusions,
    apply_range_labels,
    effective_annotations,
    validate_annotation,
)
from src.features.background_sampler import (
    distance_to_nearest_km,
    sample_completeness_absences,
    sample_disk_pseudo_absences,
    sample_profile_pseudo_absences,
)
from src.features.raster_sampler import BioclimStack
from src.modeling.algorithms import (
    WeightedEnsemble,
    ensemble_weights,
    fit_pa_member,
    spatial_block_ids,
)
from src.modeling.evaluation import (
    auc,
    choose_threshold,
    permutation_importance,
    schoeners_d,
    threshold_candidates,
)
from src.modeling.integrated import IntegratedData, fit_integrated
from src.modeling.response import response_curves
from tests.conftest import BIOCLIM_VERSION, TAXON_KEY


def separable(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.3).astype(float)
    x = pd.DataFrame(
        {"a": rng.normal(y * 2.0, 1.0), "b": rng.normal(0, 1, n), "c": rng.normal(-y, 1.0)}
    )
    return x, y


# ------------------------------------------------------------------ thresholds
def test_threshold_candidates_cover_every_rule_with_rates():
    rng = np.random.default_rng(1)
    pres, bg = rng.beta(5, 2, 300), rng.beta(2, 5, 3000)
    cands = threshold_candidates(pres, bg, fixed_value=0.4)
    assert set(cands) == {"p10", "max_tss", "equal_sens_spec", "min_training_presence", "fixed"}
    assert cands["p10"]["sensitivity"] == pytest.approx(0.9, abs=0.01)
    assert cands["min_training_presence"]["sensitivity"] == 1.0
    eq = cands["equal_sens_spec"]
    assert abs(eq["sensitivity"] - eq["specificity_background"]) < 0.02
    # max TSS really maximises sensitivity + specificity
    best = cands["max_tss"]["sensitivity"] + cands["max_tss"]["specificity_background"]
    for rule in ("p10", "equal_sens_spec", "fixed"):
        assert best >= cands[rule]["sensitivity"] + cands[rule]["specificity_background"] - 1e-6
    assert choose_threshold(cands, "fixed") == (0.4, "user-defined value")
    with pytest.raises(ValueError):
        choose_threshold(cands, "nonsense")


def test_permutation_importance_finds_the_informative_column():
    x, y = separable()
    imp = permutation_importance(lambda f: f["a"].to_numpy(), x, y, ["a", "b", "c"], 2, 0)
    assert imp["a"] == pytest.approx(100.0) and imp["b"] == 0.0


def test_schoeners_d_is_one_for_identical_and_lower_for_shifted_surfaces():
    a = np.linspace(0, 1, 100)
    assert schoeners_d(a, a) == pytest.approx(1.0)
    assert schoeners_d(a, a[::-1]) < 0.7


# ------------------------------------------------------------------ members / ensemble
@pytest.mark.parametrize("algorithm", ["glm", "gam", "rf", "brt"])
def test_presence_absence_members_learn_a_separable_signal(algorithm):
    x, y = separable()
    model = fit_pa_member(algorithm, x, y, ["a", "b", "c"], seed=0, repeats=3)
    pred = model.predict(x)
    assert pred.min() >= 0 and pred.max() <= 1
    assert auc(pred[y == 1], pred[y == 0]) > 0.8


def test_spatial_blocks_are_shared_and_ensemble_weights_follow_cv():
    rng = np.random.default_rng(3)
    plon, plat = rng.uniform(0, 40, 200), rng.uniform(30, 60, 200)
    ids = spatial_block_ids(plon, plat, plon, plat, 4, seed=1)
    again = spatial_block_ids(plon[:50], plat[:50], plon, plat, 4, seed=1)
    assert set(ids) == {0, 1, 2, 3} and (again == ids[:50]).all()

    class M:
        def __init__(self, algorithm, auc_mean):
            self.algorithm, self.cv_summary = algorithm, {"auc_mean": auc_mean}

    w, rule = ensemble_weights([M("maxent", 0.9), M("glm", 0.8), M("rf", 0.6)], 0.7)
    assert w == pytest.approx({"maxent": 0.4, "glm": 0.3}) and "left out" in rule
    w, rule = ensemble_weights([M("glm", 0.55)], 0.7)
    assert w == {"glm": 1.0} and "equal weights" in rule


def test_weighted_ensemble_prediction_and_agreement():
    class Const:
        def __init__(self, v):
            self.v = v

        def predict(self, x):
            return np.full(len(x), self.v)

    ens = WeightedEnsemble(
        {"a": Const(0.8), "b": Const(0.2)}, {"a": 3.0, "b": 1.0}, {"a": 0.5, "b": 0.5}, ["x"]
    )
    x = pd.DataFrame({"x": [1.0, 2.0]})
    assert ens.predict(x) == pytest.approx([0.65, 0.65])
    # ensemble says suitable (0.65 ≥ 0.5); only member a (weight 0.75) agrees
    assert ens.member_agreement(x, 0.5) == pytest.approx([0.75, 0.75])


# ------------------------------------------------------------------ integrated model
def test_integrated_model_estimates_the_bias_and_shares_the_response():
    rng = np.random.default_rng(5)
    nq = 4000
    env = rng.normal(0, 1, nq)
    effort = rng.normal(0, 1, nq)
    true_eta = 1.0 * env - 0.6 * env**2
    # presence-only: thinned by observer effort (bias coefficient 1.2)
    lam = np.exp(true_eta + 1.2 * effort)
    po = rng.choice(nq, size=600, p=lam / lam.sum())
    # survey: unbiased presence–absence at random sites
    sites = rng.choice(nq, size=400, replace=False)
    p_true = 1 - np.exp(-np.exp(-0.5 + true_eta[sites]))
    y = (rng.uniform(size=400) < p_true).astype(float)
    frame = pd.DataFrame({"env": env, "effort": effort})
    data = IntegratedData(["env"], frame.iloc[po], frame, frame.iloc[sites][["env"]], y, ["effort"])
    out = fit_integrated(data, l2=0.1)
    rep, model = out["report"], out["model"]
    assert rep["converged"]
    assert rep["bias_coefficients"]["effort"] > 0.5  # bias estimated, not assumed
    assert rep["coefficients"]["env"] > 0 and rep["coefficients"]["env²"] < 0
    pred = model.predict(pd.DataFrame({"env": [-2.0, 0.8, 3.0]}))
    assert pred[1] > pred[0] and pred[1] > pred[2] and ((pred >= 0) & (pred <= 1)).all()
    assert (
        auc(
            model.predict(frame.iloc[sites][["env"]])[y == 1],
            model.predict(frame.iloc[sites][["env"]])[y == 0],
        )
        > 0.6
    )


def test_response_curves_follow_the_model():
    class Model:
        def predict(self, x):
            return 1 / (1 + np.exp(-x["a"].to_numpy()))

    x, y = separable()
    rc = response_curves(Model(), x[y == 1], x[y == 0], ["a", "b"], n_points=10, n_rows=50)
    a, b = rc["curves"]
    assert a["predictor"] == "a" and len(a["x"]) == len(a["y"]) == 10
    assert np.all(np.diff(a["y"]) > 0)  # increasing in a
    assert np.allclose(b["y"], b["y"][0], atol=0.02)  # flat in b
    assert a["presence_range"][0] < a["presence_range"][1]


# ------------------------------------------------------------------ pseudo-absences
@pytest.fixture
def stack(data_root):
    return BioclimStack.open_version(data_root / "bioclim", BIOCLIM_VERSION)


def test_disk_and_profile_pseudo_absences(stack):
    rng = np.random.default_rng(2)
    lon, lat = rng.uniform(5, 30, 80), rng.uniform(40, 52, 80)
    preds = ["bio1", "bio4", "bio12"]
    disk = sample_disk_pseudo_absences(lon, lat, stack, 300, 800, 150, 1, preds)
    assert len(disk.lon) == 300 and not disk.features.isna().any().any()
    assert distance_to_nearest_km(disk.lon, disk.lat, lon, lat).min() >= 150
    pres_x = stack.sample(lon, lat, preds)
    prof = sample_profile_pseudo_absences(lon, lat, pres_x, stack, 200, 1500, 1, preds)
    lo, hi = pres_x.quantile(0.025), pres_x.quantile(0.975)
    inside = ((prof.features >= lo) & (prof.features <= hi)).all(axis=1)
    assert not inside.any()  # nothing inside the presences' envelope


def test_completeness_absences_need_well_sampled_cells_without_the_species(stack):
    elon, elat = np.meshgrid(np.arange(0.25, 30, 0.5), np.arange(40.25, 55, 0.5))
    elon, elat = elon.ravel(), elat.ravel()
    counts = np.where(elon < 15, 200, 5)  # west well sampled, east barely
    plon, plat = np.array([5.0]), np.array([45.0])
    sample, info = sample_completeness_absences(
        elon,
        elat,
        counts,
        plon,
        plat,
        stack,
        500,
        1,
        ["bio1"],
        min_records=50,
        min_distance_km=100,
    )
    assert info["n_well_sampled"] == int((counts >= 50).sum())
    assert (sample.lon < 15).all()
    assert distance_to_nearest_km(sample.lon, sample.lat, plon, plat).min() >= 100


# ------------------------------------------------------------------ annotations
def ann(record_id, kind, value, minutes):
    return RecordAnnotation(
        TAXON_KEY,
        "GBIF",
        record_id,
        kind,
        value,
        created_ts=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes),
    )


def test_latest_annotation_per_kind_wins_and_rules_apply():
    eff = effective_annotations(
        [
            ann(1, "exclude", "true", 0),
            ann(1, "exclude", "false", 5),  # withdrawn
            ann(2, "cultivated", "true", 1),
            ann(3, "verified", "true", 2),
            ann(4, "range_label", "introduced", 3),
        ]
    )
    assert len(eff) == 4 and eff.set_index("record_id").loc[1, "exclude"] == False  # noqa: E712
    occ = pd.DataFrame({"source": ["GBIF"] * 5, "record_id": [1, 2, 3, 4, 5]})
    quality_excluded = np.array([False, False, True, False, True])
    excluded, info = apply_exclusions(occ, quality_excluded, eff)
    assert excluded.tolist() == [False, True, False, False, True]
    assert info["n_kept_verified"] == 1 and info["n_excluded_cultivated"] == 1
    labels, n = apply_range_labels(occ, pd.Series(["native"] * 5), eff)
    assert labels.tolist() == ["native", "native", "native", "introduced", "native"] and n == 1
    assert validate_annotation("range_label", " Native ") == "native"
    with pytest.raises(ValueError):
        validate_annotation("exclude", "maybe")
