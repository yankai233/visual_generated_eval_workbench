from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.data.sample_identity import prepare_sample
from src.data.schemas import PipelineSelection, SampleModel
from src.runtime.job_manager import JobManager, TERMINAL_STATUSES
from src.web.server import create_app


@pytest.fixture
def benchmark_client(tmp_path, monkeypatch):
    benchmarks = tmp_path / "benchmarks"
    for name in ("first", "second"):
        directory = benchmarks / name
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps([
            {"id": "1", "prompt": {"zh": f"{name} 中文", "en": f"{name} English"}},
            {"id": "2", "prompt": {"zh": "第二个样本", "en": "Second sample"}},
        ]), encoding="utf-8")
    monkeypatch.setattr("src.web.server.BENCHMARK_ROOT", benchmarks)
    monkeypatch.setattr("src.web.server.GEN_OUTPUT_ROOT", tmp_path / "gen")
    app = create_app()
    submitted = {}

    def capture(**kwargs):
        submitted.clear()
        submitted.update(kwargs)
        return SimpleNamespace(snapshot=lambda: {"id": "multi", "results": []})

    app.state.job_manager.submit = capture
    with TestClient(app) as client:
        yield client, submitted


def request(benchmarks):
    return {"input_mode": "benchmark", "benchmarks": benchmarks, "pipelines": [{"pipeline": "demo", "config_name": "default", "gpu_ids": ["cpu"]}]}


def test_multi_benchmark_selection_keeps_identity_language_and_duplicate_ids(benchmark_client) -> None:
    client, submitted = benchmark_client
    response = client.post("/api/jobs", json=request([
        {"name": "first", "language": "en", "sample_ids": ["1"]},
        {"name": "second", "language": "zh", "sample_ids": ["1", "2"]},
    ]))
    assert response.status_code == 201, response.text
    samples = submitted["samples"]
    assert [(sample.benchmark_name, sample.id) for sample in samples] == [("first", "1"), ("second", "1"), ("second", "2")]
    assert samples[0].prompt == "first English"
    assert samples[1].prompt == "second 中文"
    assert samples[0].comparison["case_id"] == "benchmark:first:1"
    assert samples[1].comparison["case_id"] == "benchmark:second:1"


def test_empty_explicit_selection_is_not_interpreted_as_all_and_legacy_works(benchmark_client) -> None:
    client, submitted = benchmark_client
    response = client.post("/api/jobs", json=request([{"name": "first", "sample_ids": []}, {"name": "second"}]))
    assert response.status_code == 201
    assert [(sample.benchmark_name, sample.id) for sample in submitted["samples"]] == [("second", "1"), ("second", "2")]
    assert client.post("/api/jobs", json=request([{"name": "first", "sample_ids": []}])).status_code == 400
    payload = request([])
    payload.update(benchmark_name="first", benchmark_language="en", sample_ids=[])
    assert client.post("/api/jobs", json=payload).status_code == 201
    assert len(submitted["samples"]) == 2
    assert all(sample.benchmark_name == "first" for sample in submitted["samples"])


@pytest.mark.parametrize("selections", [[], [{"name": "first"}, {"name": "first"}], [{"name": "missing"}], [{"name": "first", "sample_ids": ["missing"]}]])
def test_invalid_multi_benchmark_requests_do_not_submit(benchmark_client, selections) -> None:
    client, submitted = benchmark_client
    response = client.post("/api/jobs", json=request(selections))
    assert response.status_code == 400
    assert not submitted


def test_custom_input_cannot_override_benchmark_output_folder(benchmark_client) -> None:
    client, submitted = benchmark_client
    payload = request([])
    payload.update(input_mode="custom", custom_samples=[{"id": "1", "prompt": "custom", "benchmark_name": "spoofed"}])
    assert client.post("/api/jobs", json=payload).status_code == 201
    assert submitted["samples"][0].benchmark_name is None
    assert submitted["samples"][0].comparison["source"] == "users"


def test_cpu_job_saves_same_ids_to_separate_benchmark_directories(tmp_path: Path) -> None:
    root = tmp_path / "gen"
    samples = [prepare_sample(SampleModel(id="1", prompt=name, benchmark_name=name, width=256, height=256, num_inference_steps=1), f"benchmark:{name}", root) for name in ("first", "second")]
    manager = JobManager(root)
    try:
        job = manager.submit(benchmark_name="multiple", samples=samples, selections=[PipelineSelection(pipeline="demo", config_name="test", gpu_ids=["cpu"])], configs={("demo", "test"): {"step_delay": 0}})
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            snapshot = job.snapshot()
            if snapshot["status"] in TERMINAL_STATUSES:
                break
            time.sleep(0.1)
        assert snapshot["status"] == "completed", snapshot
        assert snapshot["total"] == snapshot["completed"] == 2
        for name in ("first", "second"):
            directory = root / "demo" / name / f"1-{job.batch_started_at}"
            assert (directory / "generated.png").is_file()
            result = json.loads((directory / "result.json").read_text())
            assert result["benchmark"] == name
            assert result["comparison"]["source"] == f"benchmark:{name}"
        assert not (root / "demo" / "multiple").exists()
    finally:
        manager.shutdown()
