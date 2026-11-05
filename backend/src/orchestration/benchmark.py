"""Run and store the virtual-species bias benchmark (§4.6) as a job."""

from __future__ import annotations

import json
import uuid
from typing import Any

from src.domain import utcnow
from src.features.raster_sampler import BioclimStack
from src.modeling.virtual_species import run_benchmark
from src.orchestration.pipeline import SpeciesPipeline

BENCHMARK_DIR = "benchmarks"


def run_and_store(pipeline: SpeciesPipeline, request: dict[str, Any]) -> dict[str, Any]:
    stack = BioclimStack.open_version(
        pipeline.settings.bioclim_root, pipeline.settings.bioclim_version
    )

    def fetch(key: int, bbox: tuple[float, float, float, float], pixel: float) -> Any:
        return pipeline.gbif.fetch_effort_density(key, bbox, max_pixel_deg=pixel)

    result = run_benchmark(
        stack,
        request,
        fetch_effort=fetch,
        progress=lambda f, m: pipeline.progress("benchmark", f, m),
    )
    report = {
        "benchmark_id": uuid.uuid4().hex[:12],
        "created_ts": utcnow().isoformat(),
        "request": request,
        "bioclim_version": stack.version,
        **result,
    }
    out = pipeline.artifacts.local_path(BENCHMARK_DIR)
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, default=str)
    (out / f"{report['benchmark_id']}.json").write_text(text, encoding="utf-8")
    (out / "latest.json").write_text(text, encoding="utf-8")
    pipeline.progress("done", 1.0, f"Benchmark: {result['interpretation']}")
    return report
