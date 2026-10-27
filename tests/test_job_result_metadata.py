from __future__ import annotations

import time

from src.data.schemas import PipelineSelection, SampleModel
from src.runtime.job_manager import JobManager, TERMINAL_STATUSES


def test_cpu_generation_persists_unambiguous_model_and_sample_metadata(tmp_path):
    manager = JobManager(tmp_path / "gen")
    try:
        job = manager.submit(
            benchmark_name="users",
            samples=[SampleModel(id="case-a", prompt="a tree", width=64, height=64, num_inference_steps=1)],
            selections=[PipelineSelection(pipeline="demo", config_name="default", gpu_ids=["cpu"])],
            configs={("demo", "default"): {"step_delay": 0}},
        )
        deadline = time.monotonic() + 20
        while job.status not in TERMINAL_STATUSES and time.monotonic() < deadline:
            time.sleep(0.05)
        snapshot = job.snapshot()
        assert snapshot["status"] == "completed", snapshot
        result = snapshot["results"][0]
        assert result["pipeline"] == "demo"
        assert result["config_name"] == "default"
        assert result["benchmark"] == "users"
        assert result["sample_id"] == "case-a"
        assert len(result["images"]) == 1
    finally:
        manager.shutdown()
