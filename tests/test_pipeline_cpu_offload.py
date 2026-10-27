from __future__ import annotations

import sys
from types import SimpleNamespace

from src.pipelines.base import BaseGenerationPipeline


class FakePipeline:
    def __init__(self) -> None:
        self.device = None

    def to(self, device: str) -> None:
        self.device = device


class FakeModernDiffusersPipeline(FakePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.offload_device = None

    def enable_model_cpu_offload(self, device=None) -> None:
        self.offload_device = device


class FakeLegacyDiffusersPipeline(FakePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.offload_gpu_id = None

    def enable_model_cpu_offload(self, gpu_id=0) -> None:
        self.offload_gpu_id = gpu_id


class FakeComponentsManager:
    def __init__(self) -> None:
        self.offload_device = None

    def enable_auto_cpu_offload(self, device=None) -> None:
        self.offload_device = device


def test_parent_moves_pipeline_when_cpu_offload_is_disabled() -> None:
    pipeline = FakePipeline()
    BaseGenerationPipeline.configure_model_device(
        pipeline, {"cpu_offload": False}, "cuda:2"
    )
    assert pipeline.device == "cuda:2"


def test_parent_supports_modern_and_legacy_diffusers_offload() -> None:
    modern = FakeModernDiffusersPipeline()
    BaseGenerationPipeline.configure_model_device(
        modern, {"cpu_offload": True}, "cuda:3"
    )
    assert modern.offload_device == "cuda:3"

    legacy = FakeLegacyDiffusersPipeline()
    BaseGenerationPipeline.configure_model_device(
        legacy, {"cpu_offload": True}, "cuda:5"
    )
    assert legacy.offload_gpu_id == 5


def test_parent_supports_modular_components_manager() -> None:
    manager = FakeComponentsManager()
    pipeline = SimpleNamespace(_components_manager=manager)
    BaseGenerationPipeline.configure_model_device(
        pipeline, {"cpu_offload": True}, "cuda:4"
    )
    assert manager.offload_device == "cuda:4"


def test_parent_supports_raw_torch_modules(monkeypatch) -> None:
    calls = []

    class Module:
        pass

    module = Module()
    fake_torch = SimpleNamespace(
        nn=SimpleNamespace(Module=Module),
        device=lambda value: f"device:{value}",
    )
    fake_accelerate = SimpleNamespace(
        cpu_offload=lambda model, execution_device: calls.append(
            (model, execution_device)
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)

    BaseGenerationPipeline.configure_model_device(
        module, {"cpu_offload": True}, "cuda:6"
    )
    assert calls == [(module, "device:cuda:6")]


def test_pipeline_metadata_advertises_cpu_offload() -> None:
    assert BaseGenerationPipeline.metadata()["supports_cpu_offload"] is True
