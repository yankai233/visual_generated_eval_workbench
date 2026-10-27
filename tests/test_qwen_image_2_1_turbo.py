from __future__ import annotations

import sys
from types import SimpleNamespace

from src.pipelines.base import GenerationRequest
from src.pipelines.qwen_image_2_1_turbo.pipeline import (
    QwenImage21TurboWorkbenchPipeline,
)


class FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device

    def manual_seed(self, seed: int):
        self.seed = seed
        return self


def test_turbo_metadata_uses_eight_default_steps() -> None:
    metadata = QwenImage21TurboWorkbenchPipeline.metadata()
    assert metadata["default_num_inference_steps"] == 8
    assert metadata["supported_tasks"] == ["T2I", "TI2I", "MultiReference"]


def test_turbo_progress_uses_checkpoint_sigma_count(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(seed=lambda: 7, Generator=FakeGenerator),
    )
    pipeline = SimpleNamespace(config=SimpleNamespace(sample_sigmas=[1.0] * 8))
    adapter = QwenImage21TurboWorkbenchPipeline(
        pipeline,
        {"true_cfg_scale": 1.0, "use_kv_cache": True},
        "cuda:0",
    )
    reported = []
    request = GenerationRequest(
        id="turbo",
        task_type="T2I",
        prompt="test",
        seed=42,
        num_inference_steps=28,
    )
    kwargs = adapter._build_kwargs(
        request,
        lambda step, total: reported.append((step, total)),
        None,
    )
    kwargs["callback_on_step_end"](pipeline, 0, None, {})
    kwargs["callback_on_step_end"](pipeline, 7, None, {})
    assert reported == [(1, 8), (8, 8)]
    assert kwargs["num_inference_steps"] == 28
    assert kwargs["use_kv_cache"] is True
