"""Sampling-bias handling (features/sampling_bias.py) and its use in the MaxEnt trainer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.sampling_bias import (
    conditioning_value,
    density_factor,
    effort_values,
    presence_weights,
)
from src.modeling.maxent_trainer import (
    BiasConditionedModel,
    TrainingConfig,
    TrainingData,
    train_maxent,
)


def test_density_factor_downweights_clusters():
    rng = np.random.default_rng(0)
    cluster_lon, cluster_lat = rng.normal(10, 0.05, 20), rng.normal(45, 0.05, 20)
    lon = np.r_[cluster_lon, [20.0, 30.0, -5.0]]
    lat = np.r_[cluster_lat, [50.0, 40.0, 38.0]]
    f = density_factor(lon, lat, k=3)
    assert np.median(f) == pytest.approx(1.0)
    assert f[:20].max() < f[20:].min() and f[20:].min() > 1.0
    assert density_factor([1.0], [2.0], 5).tolist() == [1.0]


def test_presence_weights_by_tier_are_rescaled_to_mean_one():
    lon, lat = [0.0, 1.0, 2.0, 3.0], [40.0, 41.0, 42.0, 43.0]
    assert presence_weights(lon, lat, None, "none") == (None, {"method": "none"})
    w, info = presence_weights(lon, lat, ["A", "A", "C", "C"], "quality_tier")
    assert w.mean() == pytest.approx(1.0)
    assert w[0] / w[2] == pytest.approx(2.0)  # A = 1.0, C = 0.5
    assert info["n_by_tier"] == {"A": 2, "C": 2} and info["tier_weights"]["B"] == 0.75
    with pytest.raises(ValueError, match="quality tiers"):
        presence_weights(lon, lat, None, "quality_tier")
    with pytest.raises(ValueError, match="Unknown presence weighting"):
        presence_weights(lon, lat, None, "kernel")


def test_effort_values_and_conditioning():
    # two 0.5° effort pixels (centres), one point in each, one point with no effort
    v = effort_values(
        [10.25, 11.25], [45.25, 45.25], [9, 99], 0.5, [10.1, 11.4, 30.0], [45.4, 45.1, 0.0]
    )
    assert v.tolist() == pytest.approx([np.log(10), np.log(100), 0.0])
    assert conditioning_value(v, "max") == pytest.approx(np.log(100))
    assert conditioning_value(np.arange(101.0), "p95") == pytest.approx(95.0)
    with pytest.raises(ValueError):
        conditioning_value(v, "mean")


def test_training_config_rejects_unknown_options():
    with pytest.raises(ValueError, match="presence_weighting"):
        TrainingConfig(presence_weighting="kde")
    with pytest.raises(ValueError, match="bias_covariate"):
        TrainingConfig(bias_covariate="roads")


def _synthetic(seed: int = 0) -> tuple[TrainingData, np.ndarray, np.ndarray]:
    """Presences prefer warm cells AND are recorded where effort is high."""
    rng = np.random.default_rng(seed)
    n_bg = 800
    bg_t = rng.uniform(0, 30, n_bg)
    bg_eff = rng.uniform(0, 5, n_bg)
    p = (bg_t > 15) & (bg_eff > 2.5)
    idx = np.flatnonzero(p)[:120]
    lon_bg, lat_bg = rng.uniform(-10, 30, n_bg), rng.uniform(35, 60, n_bg)
    data = TrainingData(
        predictors=["bio1"],
        presence_lon=lon_bg[idx],
        presence_lat=lat_bg[idx],
        presence_x=pd.DataFrame({"bio1": bg_t[idx], "bias_effort": bg_eff[idx]}),
        background_lon=lon_bg,
        background_lat=lat_bg,
        background_x=pd.DataFrame({"bio1": bg_t, "bias_effort": bg_eff}),
        background_method="test",
    )
    return data, bg_t, bg_eff


CFG = TrainingConfig(
    tune=False, feature_classes="LQ", n_folds=3, importance_repeats=1, min_presences=10
)


def test_bias_conditioned_model_holds_the_covariate_constant():
    data, bg_t, _ = _synthetic()
    data.bias_covariates = {"bias_effort": 5.0}
    result = train_maxent(data, "A", CFG)
    assert isinstance(result.model, BiasConditionedModel)
    assert result.predictors == ["bio1"]
    # callers pass climate only; the covariate is the conditioning constant
    x = pd.DataFrame({"bio1": [5.0, 25.0]})
    direct = np.asarray(
        result.model.model.predict(pd.DataFrame({"bio1": [5.0, 25.0], "bias_effort": 5.0}))
    )
    assert np.allclose(result.model.predict(x), direct)
    assert result.model.predict(x)[1] > result.model.predict(x)[0]
    # MESS / extrapolation reference are climate-only; covariate importance is reported apart
    assert result.mess_reference.variables == ("bio1",)
    assert set(result.importance) == {"bio1"}
    assert set(result.extra["bias_importance"]) == {"bias_effort"}


def test_presence_weights_reach_elapid_without_mutating_the_input():
    data, _, _ = _synthetic(1)
    data.predictors = ["bio1"]
    plain = train_maxent(data, "A", CFG)
    weights = np.linspace(0.2, 1.8, data.n_presence)
    data.presence_weight = weights.copy()
    weighted = train_maxent(data, "A", CFG)
    assert np.array_equal(data.presence_weight, weights)  # elapid scales a copy
    assert weighted.extra["weighted"] and not plain.extra["weighted"]
    grid = pd.DataFrame({"bio1": np.linspace(0, 30, 7)})
    assert not np.allclose(plain.model.predict(grid), weighted.model.predict(grid))
