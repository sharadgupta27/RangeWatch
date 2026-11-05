"""Virtual-species bias benchmark (§4.6) on the synthetic stack: every variant is scored
against the known truth, and the run is stored as a report the API can serve."""

from __future__ import annotations

import json

from src.api.schemas import BenchmarkReport
from src.features.raster_sampler import BioclimStack
from src.modeling.virtual_species import run_benchmark
from src.orchestration.benchmark import run_and_store
from tests.conftest import BIOCLIM_VERSION, FakeGbif
from tests.test_pipeline import initial_records, make_pipeline


def test_benchmark_scores_every_variant_against_the_truth(data_root):
    stack = BioclimStack.open_version(data_root / "bioclim", BIOCLIM_VERSION)
    out = run_benchmark(stack, {"n_presences": 150, "bias_strength": 3.0, "seed": 1})
    by = {v["variant"]: v for v in out["variants"]}
    assert set(by) == {"no_correction", "target_group", "bias_covariate", "density_weights"}
    for v in by.values():
        assert v.get("error") is None, v
        assert -1 <= v["spearman_vs_truth"] <= 1 and 0 <= v["schoeners_d"] <= 1
    assert out["best_variant"] in by and out["interpretation"]
    assert out["virtual_species"]["predictors"] == ["bio1", "bio4"]


def test_benchmark_report_is_stored_and_matches_the_api_schema(settings):
    p = make_pipeline(settings, FakeGbif(initial=initial_records()))
    request = {
        "n_presences": 100,
        "bias": "gbif_effort",
        "target_group_key": 408,
        "variants": ["no_correction", "target_group"],
        "seed": 3,
    }
    report = run_and_store(p, request)
    path = p.artifacts.local_path("benchmarks/latest.json")
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["benchmark_id"] == report["benchmark_id"]
    assert stored["bias"]["source"] == "gbif_effort" and stored["bias"]["query"] == "fake://density"
    BenchmarkReport.model_validate(stored)
