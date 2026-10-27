from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.data.result_catalog import ResultCatalog
from src.data.sample_identity import prepare_sample
from src.data.schemas import PipelineSelection, SampleModel
from src.pipelines.base import GenerationCancelled
from src.runtime.job_manager import JobManager, TERMINAL_STATUSES
from src.runtime.preview_store import PreviewStore, PreviewUnavailable
from src.web.server import _public_snapshot, create_app


def png(color="green"):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
    return buffer.getvalue()


def context(root, name="sample"):
    sample = SampleModel(id=name, prompt="a tree", width=64, height=64).model_dump()
    return {"output_dir": str(root / "demo" / "users" / f"{name}-batch"), "unit_id": f"demo:{name}:1",
            "request": sample, "config": {"step_delay": 0},
            "result": {"schema_version": 2, "status": "completed", "job_id": "preview-job", "pipeline": "demo",
                       "config_name": "default", "sample_id": name, "benchmark": "users", "batch_started_at": "batch", "images": []}}


def test_preview_is_ram_only_saves_selected_image_and_is_idempotent(tmp_path):
    root = tmp_path / "gen"
    store = PreviewStore(root)
    try:
        key = store.add([png(), png("blue")], context(root))
        assert not root.exists()
        assert store.image(key, 1) == (png("blue"), "generated_1.png")
        saved = store.save(key, 1)
        directory = Path(saved["output_dir"])
        assert (directory / "generated_1.png").read_bytes() == png("blue")
        assert not (directory / "generated_0.png").exists()
        assert store.save(key, 1) == saved
        assert json.loads((directory / "result.json").read_text())["images"] == saved["images"]
        assert (directory / "request.json").is_file()
        assert (directory / "config.yaml").is_file()
        catalog = ResultCatalog(root, tmp_path / "benchmarks")
        assert catalog.catalog()["record_count"] == 1
        assert len(catalog.records()[0]["image_urls"]) == 1
        saved2 = store.save(key, 0)
        assert len(saved2["images"]) == 2
        assert len(catalog.records()[0]["image_urls"]) == 2
    finally:
        store.close()


def test_preview_expiry_capacity_invalid_index_and_destination_guard(tmp_path):
    clock = [0.0]
    store = PreviewStore(tmp_path / "gen", max_bytes=len(png()) + 1, ttl_seconds=10, clock=lambda: clock[0])
    try:
        first = store.add([png()], context(tmp_path / "gen", "first"))
        second = store.add([png()], context(tmp_path / "gen", "second"))
        with pytest.raises(PreviewUnavailable):
            store.image(first, 0)
        with pytest.raises(PreviewUnavailable):
            store.image(second, -1)
        with pytest.raises(PreviewUnavailable):
            store.image(second, 1)
        clock[0] = 10
        with pytest.raises(PreviewUnavailable):
            store.save(second, 0)
        with pytest.raises(ValueError):
            store.add([png(), png()], context(tmp_path / "gen"))
        key = store.add([png()], context(tmp_path / "outside"))
        with pytest.raises(ValueError, match="非法"):
            store.save(key, 0)
        assert not (tmp_path / "outside").exists()
    finally:
        store.close()


def test_no_save_input_identity_does_not_archive_reference_images(tmp_path):
    reference = tmp_path / "ref.png"
    reference.write_bytes(png())
    sample = SampleModel(id="my-case", prompt="edit", task_type="TI2I", input_images=[str(reference)])
    root = tmp_path / "gen"
    prepared = prepare_sample(sample, "users", root, archive=False)
    assert not root.exists()
    assert prepared.input_images == sample.input_images
    assert "input_manifest" not in prepared.comparison
    archived = prepare_sample(sample, "users", root)
    assert prepared.id == archived.id
    assert prepared.comparison["case_id"] == archived.comparison["case_id"]


def test_cpu_generation_no_save_preview_promotion_and_restart(tmp_path, monkeypatch):
    root = tmp_path / "gen"
    monkeypatch.setattr("src.web.server.OUTPUT_ROOT", tmp_path)
    manager = JobManager(root)
    try:
        job = manager.submit(benchmark_name="users", samples=[SampleModel(id="test", prompt="test", width=64, height=64, num_inference_steps=1)],
                             selections=[PipelineSelection(pipeline="demo", config_name="default", gpu_ids=["cpu"])],
                             configs={("demo", "default"): {"step_delay": 0}}, save_images=False)
        deadline = time.monotonic() + 20
        while job.status not in TERMINAL_STATUSES and time.monotonic() < deadline:
            time.sleep(0.05)
        snapshot = job.snapshot()
        assert snapshot["status"] == "completed", snapshot
        assert not snapshot["save_images"]
        assert not (root / "demo").exists()
        assert not list(root.rglob("*.png"))
        result = snapshot["results"][0]
        assert result["images"] == []
        assert result["preview_count"] == 1
        preview_id = result["preview_id"]
        assert _public_snapshot(snapshot)["results"][0]["image_urls"] == [f"/api/previews/{preview_id}/0"]
        assert manager.previews.image(preview_id, 0)[0].startswith(b"\x89PNG")
        saved = manager.save_preview(preview_id, 0)
        assert Path(saved["path"]).is_file()
        assert ResultCatalog(root, tmp_path / "benchmarks").catalog()["record_count"] == 1
        assert _public_snapshot(job.snapshot())["results"][0]["image_urls"][0].endswith("generated.png")
        restarted = JobManager(root)
        try:
            snapshot = restarted.get_snapshot(job.id)
            assert snapshot["results"][0]["saved_images"]["0"] == saved["path"]
            with pytest.raises(PreviewUnavailable):
                restarted.previews.image(preview_id, 0)
        finally:
            restarted.shutdown()
    finally:
        manager.shutdown()


def test_preview_web_read_download_save_and_expired_response(tmp_path, monkeypatch):
    root = tmp_path / "gen"
    monkeypatch.setattr("src.web.server.GEN_OUTPUT_ROOT", root)
    monkeypatch.setattr("src.web.server.OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr("src.web.server.UPLOAD_ROOT", tmp_path / "uploads")
    app = create_app()
    store = app.state.job_manager.previews
    with TestClient(app) as client:
        key = store.add([png()], context(root))
        response = client.get(f"/api/previews/{key}/0")
        assert response.content == png()
        assert response.headers["cache-control"] == "no-store"
        response = client.get(f"/api/previews/{key}/0/download")
        assert response.content == png()
        assert "attachment" in response.headers["content-disposition"]
        assert not (root / "demo").exists()
        response = client.post(f"/api/previews/{key}/0/save")
        assert response.status_code == 200, response.text
        assert client.get(response.json()["url"]).content == png()
        assert client.post(f"/api/previews/{key}/0/save").json() == response.json()
        assert client.get(f"/api/previews/{key}/-1").status_code == 410
        assert client.get("/api/previews/missing/0").status_code == 410
        assert client.post("/api/previews/missing/0/save").status_code == 410


def test_preview_saving_never_overwrites_existing_image(tmp_path):
    root = tmp_path / "gen"
    store = PreviewStore(root)
    try:
        details = context(root)
        key = store.add([png()], details)
        directory = Path(details["output_dir"])
        directory.mkdir(parents=True)
        destination = directory / "generated.png"
        destination.write_bytes(b"existing-user-image")
        with pytest.raises(ValueError, match="未覆盖"):
            store.save(key, 0)
        assert destination.read_bytes() == b"existing-user-image"
    finally:
        store.close()


def test_no_save_api_forwards_option_and_skips_input_archive(tmp_path, monkeypatch):
    from types import SimpleNamespace
    root = tmp_path / "gen"
    monkeypatch.setattr("src.web.server.GEN_OUTPUT_ROOT", root)
    app = create_app()
    captured = {}
    def submit(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(snapshot=lambda: {"id": "test", "results": []})
    app.state.job_manager.submit = submit
    with TestClient(app) as client:
        response = client.post("/api/jobs", json={"input_mode": "custom", "save_images": False,
            "custom_samples": [{"id": "my-test", "prompt": "test"}],
            "pipelines": [{"pipeline": "demo", "config_name": "default", "gpu_ids": ["cpu"]}]})
        assert response.status_code == 201, response.text
        assert captured["save_images"] is False
        assert not (root / ".cases").exists()


@pytest.mark.parametrize("exception", [RuntimeError("test failure"), GenerationCancelled("cancelled")])
def test_no_save_worker_failures_and_cancellation_do_not_create_directories(tmp_path, monkeypatch, exception):
    import queue
    import threading
    import src.runtime.gpu_worker as worker
    class FailingPipeline:
        @classmethod
        def from_pretrained(cls, config, device):
            return cls()
        def __call__(self, *args, **kwargs):
            raise exception
    monkeypatch.setattr(worker, "get_pipeline", lambda name: FailingPipeline)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "never-created"
    messages = queue.Queue()
    worker.run_gpu_worker("test", "cpu", [{"unit_id": "test", "pipeline": "test", "config_name": "default", "config": {},
        "sample": SampleModel(id="test", prompt="test").model_dump(), "save_images": False, "output_dir": str(output)}],
        threading.Lock(), threading.Event(), threading.Event(), messages)
    assert not output.exists()
    types = []
    while not messages.empty():
        types.append(messages.get()["type"])
    assert "worker_done" in types
    assert "unit_failed" in types or "unit_cancelled" in types
