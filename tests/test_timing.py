from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from src.data.schemas import PipelineSelection, SampleModel
from src.pipelines.base import GenerationCancelled, PipelineControl
from src.runtime.job_manager import JobManager, TERMINAL_STATUSES
from src.runtime.timing import JobTiming, StepTiming, finite_seconds
from src.runtime.state_store import JsonJobStateStore


def unit(name, pipeline="demo", weight=10):
    return {"unit_id": name, "pipeline": pipeline, "config_name": "default", "sample": {"width": 1, "height": 1, "num_inference_steps": weight}}


def send(tracker, gpu, kind, name=None, **kwargs):
    tracker.handle({"gpu_id": gpu, "type": kind, "unit_id": name, **kwargs})


def start(tracker, gpu, name, elapsed, eta):
    send(tracker, gpu, "unit_started", name)
    send(tracker, gpu, "unit_generation_started", name)
    send(tracker, gpu, "denoise_progress", name, step=4, total=10, generation_elapsed_seconds=elapsed, eta_seconds=eta)


def test_step_eta_waits_for_intervals_and_ignores_initial_setup() -> None:
    timer = StepTiming()
    assert timer.update(1, 10, 100) is None
    assert timer.update(2, 10, 102) == 16
    assert timer.update(3, 10, 104) == 14
    assert timer.update(10, 10, 118) == 0
    assert timer.update(1, 20, 200) is None  # next image / callback sequence
    assert timer.update(1, 20, 201) is None  # repeated steps do not divide by zero
    assert timer.update(2, 20, 202) == 18
    assert len(timer.points) <= 7


def test_parallel_eta_is_slowest_gpu_and_pause_does_not_advance() -> None:
    clock = [100.0]
    tracker = JobTiming({"0": [unit("a"), unit("b")], "1": [unit("c"), unit("d"), unit("e")]}, clock=lambda: clock[0])
    assert tracker.snapshot("running")["eta_seconds"] is None
    start(tracker, "0", "a", elapsed=4, eta=6)
    assert tracker.snapshot("running")["eta_seconds"] is None  # other GPU is still waiting
    start(tracker, "1", "c", elapsed=8, eta=12)
    snapshot = tracker.snapshot("running")
    assert snapshot["eta_seconds"] == 52  # max(6+10, 12+20+20), not the sum
    assert snapshot["active"]["0"]["eta_seconds"] == 6
    clock[0] += 5
    assert tracker.snapshot("running")["eta_seconds"] == 47
    tracker.pause()
    clock[0] += 600
    assert tracker.snapshot("paused")["eta_seconds"] == 47
    assert tracker.snapshot("paused")["active"]["1"]["eta_seconds"] == 7
    tracker.resume()
    clock[0] += 2
    assert tracker.snapshot("running")["eta_seconds"] == 45


def test_completed_rates_exclude_model_loading_and_scale_pending_work() -> None:
    tracker = JobTiming({"0": [unit("a"), unit("b", weight=20)]}, clock=lambda: 0)
    send(tracker, "0", "unit_started", "a")
    assert tracker.snapshot("running")["eta_phase"] == "loading"
    start(tracker, "0", "a", elapsed=4, eta=6)
    assert tracker.snapshot("running")["eta_seconds"] == 26
    send(tracker, "0", "unit_completed", "a", duration_seconds=1010, generation_duration_seconds=10)
    assert tracker.snapshot("running")["eta_seconds"] == 20
    assert tracker.snapshot("running")["average_sample_seconds"] == 10
    send(tracker, "0", "unit_started", "b")
    assert tracker.snapshot("running")["eta_seconds"] is None
    send(tracker, "0", "unit_generation_started", "b")
    assert tracker.snapshot("running")["eta_seconds"] == 20
    send(tracker, "0", "unit_completed", "b", duration_seconds=30, generation_duration_seconds=30)
    assert tracker.snapshot("completed")["average_sample_seconds"] == 20


def test_unknown_pipelines_and_failed_units_are_not_assumed_fast() -> None:
    tracker = JobTiming({"0": [unit("a"), unit("b", pipeline="other")]}, clock=lambda: 0)
    start(tracker, "0", "a", elapsed=4, eta=6)
    assert tracker.snapshot("running")["eta_seconds"] is None
    send(tracker, "0", "unit_failed", "a")
    assert tracker.snapshot("running")["eta_seconds"] is None
    assert tracker.snapshot("completed")["eta_seconds"] == 0
    assert tracker.snapshot("completed_with_errors")["eta_seconds"] == 0
    for status in ("cancelled", "failed", "interrupted", "cancelling"):
        assert tracker.snapshot(status)["eta_seconds"] is None
        assert tracker.snapshot(status)["active"] == {}
    assert finite_seconds(float("nan")) is None
    assert finite_seconds(-1) is None
    assert finite_seconds(float("inf")) is None


def test_pipeline_active_clock_excludes_actual_pause_and_handles_cancel(monkeypatch) -> None:
    clock = [10.0]
    pause = threading.Event()
    pause.set()
    monkeypatch.setattr("src.pipelines.base.time", SimpleNamespace(monotonic=lambda: clock[0]))

    class Cancel:
        def is_set(self):
            return False

        def wait(self, timeout):
            clock[0] += 100
            pause.clear()
            return False

    control = PipelineControl(pause, Cancel())
    control.checkpoint()
    assert control.active_time() == 10
    clock[0] += 5
    assert control.active_time() == 15
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(GenerationCancelled):
        PipelineControl(pause, cancelled).checkpoint()


def test_cpu_job_emits_both_eta_fields_and_finishes_with_zero(tmp_path) -> None:
    manager = JobManager(tmp_path / "gen")
    samples = [SampleModel(id=str(index), prompt="timing test", width=256, height=256, num_inference_steps=20) for index in range(3)]
    seen_eta = False
    try:
        job = manager.submit(benchmark_name="users", samples=samples, selections=[PipelineSelection(pipeline="demo", config_name="default", gpu_ids=["cpu"])], configs={("demo", "default"): {"step_delay": 0.03}})
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            snapshot = job.snapshot()
            if snapshot["eta_seconds"] is not None and any(item.get("eta_seconds") is not None for item in snapshot["active"].values()):
                seen_eta = True
            if snapshot["status"] in TERMINAL_STATUSES:
                break
            time.sleep(0.025)
        assert snapshot["status"] == "completed", snapshot
        assert snapshot["eta_seconds"] == 0
        assert snapshot["average_sample_seconds"] is not None
        assert seen_eta
    finally:
        manager.shutdown()


def test_interrupted_history_does_not_keep_an_old_eta(tmp_path) -> None:
    JsonJobStateStore(tmp_path / ".jobs").save({"id": "old", "status": "running", "eta_seconds": 123, "eta_phase": "running", "active": {"0": {"eta_seconds": 20}}})
    manager = JobManager(tmp_path)
    try:
        snapshot = manager.get_snapshot("old")
        assert snapshot["status"] == "interrupted"
        assert snapshot["eta_seconds"] is None
        assert snapshot["active"] == {}
    finally:
        manager.shutdown()
