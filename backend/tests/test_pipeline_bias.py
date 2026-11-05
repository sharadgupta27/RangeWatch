"""Pipeline integration of the sampling-bias options (Priority 2: B6 bias covariate, B7
observation weights): logged per version, stored in the training snapshot, never blocking."""

from __future__ import annotations

from dataclasses import replace

import elapid
import numpy as np
import pandas as pd
import pytest

from src.features.sampling_bias import EFFORT_COVARIATE
from src.modeling.maxent_trainer import BiasConditionedModel
from src.orchestration.pipeline import Decision
from tests.conftest import TAXON_KEY, FakeGbif
from tests.test_pipeline import confirm_native, initial_records, make_pipeline


def trained_with(settings, gbif=None, **cfg):
    p = make_pipeline(settings, gbif or FakeGbif(initial=initial_records()))
    p.training_config = replace(p.training_config, **cfg)
    p.run(TAXON_KEY)
    confirm_native(p)
    assert p.run(TAXON_KEY).decision == Decision.TRAINED
    return p, p.repo.get_model_version(TAXON_KEY, 1)


def test_defaults_train_unweighted_without_bias_covariates(settings):
    _, mv = trained_with(settings)
    repro = mv.metrics["reproducibility"]
    assert repro["presence_weights"] == {"method": "none"}
    assert repro["bias_covariates"] == {"method": "none"}
    assert repro["training_config"]["presence_weighting"] == "none"


def test_presence_weights_are_logged_and_snapshotted(settings):
    p, mv = trained_with(settings, presence_weighting="quality_tier_density")
    info = mv.metrics["reproducibility"]["presence_weights"]
    assert info["method"] == "quality_tier_density" and info["density_neighbors"] == 5
    assert info["min"] <= info["median"] <= info["max"]
    training = pd.read_parquet(p.artifacts.local_path(mv.artifacts["training_data"]))
    pres = training[training["role"] == "presence"]
    assert len(pres) == info["n_presences"] == mv.metrics["n_presence"]
    assert pres["sample_weight"].mean() == pytest.approx(1.0)
    assert (training.loc[training["role"] == "background", "sample_weight"] == 1.0).all()


def test_effort_bias_covariate_is_conditioned_for_prediction(settings):
    p, mv = trained_with(settings, bias_covariate="target_group_effort")
    repro = mv.metrics["reproducibility"]
    bias = repro["bias_covariates"]
    assert bias["status"] == "used" and bias["covariate"] == EFFORT_COVARIATE
    assert bias["conditioning_value"] == round(np.log1p(3), 4)  # fake effort: 3 per pixel
    assert EFFORT_COVARIATE in bias["importance"]
    # observer bias is modelled explicitly, so the background is uniform in the buffer
    assert repro["background"]["method"].startswith("buffer")
    assert repro["background"]["bias_covariate_override"] == "target_group_effort"

    training = pd.read_parquet(p.artifacts.local_path(mv.artifacts["training_data"]))
    assert EFFORT_COVARIATE in training and training[EFFORT_COVARIATE].notna().all()
    saved = elapid.load_object(str(p.artifacts.local_path(mv.artifacts["model"])))
    model = saved["model"]
    assert isinstance(model, BiasConditionedModel)
    assert saved["predictors"] == mv.metrics["predictors"]
    assert EFFORT_COVARIATE not in mv.metrics["predictors"]
    assert mv.artifacts["suitability"]  # global projection ran with climate predictors only


def test_unavailable_effort_never_blocks_training(settings):
    class NoEffort(FakeGbif):
        def fetch_effort_density(self, taxon_key, bbox, max_pixel_deg):
            raise TimeoutError("map API stalled")

    _, mv = trained_with(
        settings, NoEffort(initial=initial_records()), bias_covariate="target_group_effort"
    )
    bias = mv.metrics["reproducibility"]["bias_covariates"]
    assert bias["status"] == "unavailable" and "map API stalled" in bias["reason"]
    assert "importance" not in bias
