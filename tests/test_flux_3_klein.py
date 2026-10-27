from __future__ import annotations

import sys
from types import SimpleNamespace

from src.pipelines.base import GenerationRequest
from src.pipelines.flux_3_klein.pipeline import Flux3KleinWorkbenchPipeline


class FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device

    def manual_seed(self, seed: int):
        self.seed = seed
        return self


def test_flux3_klein_metadata_uses_distilled_defaults() -> None:
    metadata = Flux3KleinWorkbenchPipeline.metadata()
    assert metadata["group"] == "FLUX"
    assert metadata["default_num_inference_steps"] == 16
    assert metadata["supported_tasks"] == ["T2I", "TI2I", "MultiReference"]
    assert metadata["max_reference_images"] == 10
    assert Flux3KleinWorkbenchPipeline.modular_component_load_kwargs({}) == {
        "fix_mistral_regex": {"tokenizer": True}
    }


def test_flux3_klein_does_not_send_cfg_inputs(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(seed=lambda: 7, Generator=FakeGenerator),
    )
    adapter = Flux3KleinWorkbenchPipeline(
        SimpleNamespace(config={}),
        {"guidance_scale": 9.0},
        "cuda:0",
    )
    kwargs = adapter._build_kwargs(
        GenerationRequest(
            id="distilled",
            task_type="T2I",
            prompt="test",
            negative_prompt="must not be forwarded",
            seed=42,
            num_inference_steps=16,
        ),
        None,
        None,
    )
    assert "guidance_scale" not in kwargs
    assert "negative_prompt" not in kwargs
    assert kwargs["num_inference_steps"] == 16
