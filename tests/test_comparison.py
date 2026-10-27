from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from src.data.result_catalog import ResultCatalog
from src.data.sample_identity import atomic_json, input_identity, prepare_sample
from src.data.schemas import SampleModel
from src.web.comparison import comparison_router
from src.runtime.gpu_worker import run_gpu_worker


def test_identity_uses_ordered_image_content_and_prompt_not_generation_parameters(tmp_path: Path) -> None:
    red = tmp_path / "a.png"
    red_copy = tmp_path / "different-upload.png"
    blue = tmp_path / "b.png"
    Image.new("RGB", (8, 8), "red").save(red)
    Image.new("RGBA", (8, 8), "red").save(red_copy)
    Image.new("RGB", (8, 8), "blue").save(blue)
    sample = {"task_type": "MultiReference", "prompt": "test", "input_images": [str(red), str(blue)]}
    digest = input_identity(sample)["fingerprint"]
    assert input_identity({**sample, "input_images": [str(red_copy), str(blue)], "seed": 900, "num_inference_steps": 1})["fingerprint"] == digest
    assert input_identity({**sample, "prompt": "different"})["fingerprint"] != digest
    assert input_identity({**sample, "input_images": [str(blue), str(red)]})["fingerprint"] != digest


def test_archive_survives_upload_deletion_and_preserves_repeated_samples(tmp_path: Path) -> None:
    original = tmp_path / "upload.png"
    Image.new("RGB", (8, 8), "red").save(original)
    root = tmp_path / "gen"
    sample = SampleModel(id="one", task_type="TI2I", prompt="edit", input_images=[str(original)])
    first = prepare_sample(sample, "users", root)
    second = prepare_sample(sample.model_copy(update={"id": "two", "seed": 2}), "users", root)
    assert first.comparison["case_id"] == second.comparison["case_id"]
    assert first.id != second.id  # one batch can repeat an input without overwriting outputs
    original.unlink()
    assert Path(first.input_images[0]).is_file()
    assert input_identity(first.model_dump())["fingerprint"] == first.comparison["input_fingerprint"]


def add_result(root: Path, pipeline: str, run: str, sample: SampleModel, source: str = "users", config: str = "default", legacy: bool = False, image_count: int = 1, generation_duration_seconds: float | None = 2.5) -> Path:
    prepared = sample if legacy else prepare_sample(sample, source, root)
    directory = root / pipeline / ("users" if source == "users" else source.removeprefix("benchmark:")) / f"{prepared.id}-{run}"
    directory.mkdir(parents=True)
    images = []
    for index in range(image_count):
        path = directory / f"generated_{index}.png"
        Image.new("RGB", (8, 8), "green").save(path)
        images.append(str(path))
    atomic_json(directory / "request.json", prepared.model_dump())
    atomic_json(directory / "result.json", {
        "status": "completed", "pipeline": pipeline, "sample_id": prepared.id,
        "job_id": run, "batch_started_at": run, "config_name": config,
        "images": images, "comparison": prepared.comparison,
        "generation_duration_seconds": generation_duration_seconds,
    })
    return directory


def test_comparison_alignment_variants_filters_missing_and_pagination(tmp_path: Path) -> None:
    root = tmp_path / "gen"
    a = SampleModel(id="a", prompt="same")
    b = SampleModel(id="b", prompt="different")
    add_result(root, "method_a", "001", a)
    add_result(root, "method_a", "003", a, config="cfg2", image_count=2)
    add_result(root, "method_b", "002", a.model_copy(update={"id": "independent-upload", "seed": 99}))
    add_result(root, "method_a", "004", b)
    catalog = ResultCatalog(root, tmp_path / "benchmarks")
    columns = [{"pipeline": "method_a"}, {"pipeline": "method_b"}]
    view = catalog.view({"source": "users", "columns": columns})
    assert view["total"] == 2
    paired = next(row for row in view["rows"] if row["prompt"] == "same")
    assert [len(cell) for cell in paired["cells"]] == [2, 1]
    assert paired["cells"][0][0]["run_id"] == "003"
    assert len(paired["cells"][0][0]["image_urls"]) == 2
    assert not paired["different_inputs"]
    assert next(row for row in view["rows"] if row["prompt"] == "different")["cells"][1] == []
    assert catalog.view({"source": "users", "columns": columns, "only_common": True})["total"] == 1
    assert catalog.view({"source": "users", "columns": columns, "query": "different"})["total"] == 1
    assert len(catalog.view({"source": "users", "columns": columns, "page_size": 1})["rows"]) == 1
    filtered = catalog.view({"source": "users", "columns": [{"pipeline": "method_a", "config_name": "default", "run_id": "001"}]})
    assert len(filtered["rows"][0]["cells"][0]) == 1
    assert filtered["rows"][0]["cells"][0][0]["generation_duration_seconds"] == 2.5


def test_timing_summary_uses_latest_timed_result_per_sample(tmp_path: Path) -> None:
    root = tmp_path / "gen"
    first = SampleModel(id="1", prompt="first")
    second = SampleModel(id="2", prompt="second")
    add_result(root, "method_a", "001", first, "benchmark:test", generation_duration_seconds=2.0)
    add_result(root, "method_a", "003", first, "benchmark:test", config="cfg2", generation_duration_seconds=6.0)
    add_result(root, "method_a", "002", second, "benchmark:test", generation_duration_seconds=4.0)
    add_result(root, "method_b", "004", first, "benchmark:test", generation_duration_seconds=None)
    summary = ResultCatalog(root, tmp_path / "benchmarks").catalog()["timing_summary"]
    assert summary == [{
        "source": "benchmark:test",
        "benchmark": "test",
        "pipeline": "method_a",
        "sample_count": 2,
        "average_generation_seconds": 5.0,
    }]


def test_benchmark_ids_align_but_different_input_versions_are_flagged(tmp_path: Path) -> None:
    root = tmp_path / "gen"
    add_result(root, "a", "001", SampleModel(id="7", prompt="English"), "benchmark:test")
    add_result(root, "b", "002", SampleModel(id="7", prompt="中文"), "benchmark:test")
    view = ResultCatalog(root, tmp_path / "benchmarks").view({"source": "benchmark:test", "columns": [{"pipeline": "a"}, {"pipeline": "b"}]})
    assert view["total"] == 1
    assert view["rows"][0]["different_inputs"]


def test_legacy_inputs_missing_are_not_guessed_and_groups_do_not_rewrite_results(tmp_path: Path) -> None:
    root = tmp_path / "gen"
    sample = SampleModel(id="user-0001", task_type="TI2I", prompt="edit", input_images=[str(tmp_path / "missing.png")])
    a = add_result(root, "a", "001", sample, legacy=True)
    b = add_result(root, "b", "002", sample, legacy=True)
    catalog = ResultCatalog(root, tmp_path / "benchmarks")
    records = catalog.records("users")
    assert len({record["case_id"] for record in records}) == 2
    assert all(record["match_kind"] == "unresolved" for record in records)
    originals = [(path, path.read_bytes()) for directory in (a, b) for path in (directory / "result.json", directory / "request.json")]
    ids = [record["result_id"] for record in records]
    catalog.set_group(ids, "my sample")
    reloaded = ResultCatalog(root, tmp_path / "benchmarks")
    assert len(reloaded.samples("users")) == 1
    assert reloaded.samples("users")[0]["label"] == "my sample"
    assert all(path.read_bytes() == original for path, original in originals)
    reloaded.set_group(ids, None)
    assert len(reloaded.samples("users")) == 2
    with pytest.raises(ValueError):
        catalog.set_group(["../../outside"], "bad")


def test_comparison_routes_and_empty_catalog(tmp_path: Path) -> None:
    catalog = ResultCatalog(tmp_path / "gen", tmp_path / "benchmarks")
    app = FastAPI()
    app.include_router(comparison_router(catalog))
    with TestClient(app) as client:
        assert client.get("/compare").status_code == 200
        assert client.get("/api/comparison/catalog").json()["sources"] == []
        assert client.post("/api/comparison/view", json={"source": "users", "columns": []}).status_code == 422
        assert client.post("/api/comparison/group", json={"result_ids": ["bad"], "name": "x"}).status_code == 400


def test_worker_persists_comparison_metadata_and_archived_references(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    root = tmp_path / "gen"
    original = tmp_path / "upload.png"
    Image.new("RGB", (8, 8), "red").save(original)
    sample = prepare_sample(SampleModel(
        id="one", task_type="TI2I", prompt="an archived input", input_images=[str(original)],
        width=256, height=256, num_inference_steps=1,
    ), "users", root)
    original.unlink()
    output = root / "demo" / "users" / f"{sample.id}-20261019T140000"
    messages = queue.Queue()
    run_gpu_worker("demo-run", "cpu", [{
        "unit_id": "unit", "pipeline": "demo", "config_name": "test",
        "config": {"step_delay": 0}, "sample": sample.model_dump(),
        "output_dir": str(output), "benchmark": "users", "batch_started_at": "20261019T140000",
    }], threading.Lock(), threading.Event(), threading.Event(), messages)
    events = []
    while not messages.empty():
        events.append(messages.get()["type"])
    assert "unit_completed" in events
    assert "unit_failed" not in events
    result = json.loads((output / "result.json").read_text())
    assert result["schema_version"] == 2
    assert result["comparison"] == sample.comparison
    assert result["generation_duration_seconds"] >= 0
    assert (output / "request.json").is_file()
    assert (output / "config.yaml").is_file()
    records = ResultCatalog(root, tmp_path / "benchmarks").records("users")
    assert len(records) == 1
    assert records[0]["case_id"] == sample.comparison["case_id"]
    assert records[0]["reference_urls"][0].startswith("/files/output/gen/.cases/")
