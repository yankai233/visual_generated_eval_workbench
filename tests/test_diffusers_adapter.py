from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.pipelines.diffusers_adapter import DiffusersPipelineAdapter


class FakeModularPipeline:
    last_instance = None

    def __init__(self) -> None:
        class FakeBlocks:
            def __init__(self) -> None:
                self.sub_blocks = {
                    "prompt_upsampler": object(), "text_encoder": object(), "denoise": object()
                }

            @property
            def expected_components(self):
                names = ["text_encoder", "tokenizer", "transformer", "vae"]
                if "prompt_upsampler" in self.sub_blocks:
                    names += ["prompt_upsampler", "prompt_upsampler_processor"]
                return [SimpleNamespace(name=name) for name in names]

        self._blocks = FakeBlocks()
        self.prompt_upsampler = None
        self.prompt_upsampler_processor = object()
        self.text_encoder = object()
        self.tokenizer = object()
        self.transformer = object()
        self.vae = object()
        self.blocks_seen_during_load = None
        self.components_seen_during_load = None
        self.device = None
        FakeModularPipeline.last_instance = self

    @classmethod
    def from_pretrained(cls, *_args, **kwargs):
        instance = cls()
        instance._components_manager = kwargs.get("components_manager")
        return instance

    def load_components(self, names=None, **_kwargs) -> None:
        self.blocks_seen_during_load = tuple(self._blocks.sub_blocks)
        self.components_seen_during_load = tuple(names or ())

    def to(self, device: str) -> None:
        self.device = device


class FakeFlux3Adapter(DiffusersPipelineAdapter):
    display_name = "Fake FLUX.3"
    diffusers_class_name = "ModularPipeline"
    modular_pipeline = True
    modular_feature_blocks = {"prompt_upsampling": "prompt_upsampler"}
    modular_required_components = ("text_encoder", "tokenizer", "transformer", "vae")
    modular_feature_components = {
        "prompt_upsampling": ("prompt_upsampler", "prompt_upsampler_processor")
    }


class FakeComponentsManager:
    def __init__(self) -> None:
        self.offload_device = None

    def enable_auto_cpu_offload(self, device: str) -> None:
        self.offload_device = device


@pytest.fixture
def fake_dependencies(monkeypatch):
    fake_torch = SimpleNamespace(bfloat16=object(), float16=object(), float32=object())
    fake_diffusers = SimpleNamespace(
        ModularPipeline=FakeModularPipeline,
        ComponentsManager=FakeComponentsManager,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setattr("src.pipelines.diffusers_adapter.resolve_pretrained_path", lambda value: value)


def test_disabled_modular_feature_is_removed_before_components_load(fake_dependencies) -> None:
    adapter = FakeFlux3Adapter.from_pretrained(
        {"model_path": "/model", "prompt_upsampling": False}, "cuda:0"
    )
    pipe = adapter.pipeline
    assert "prompt_upsampler" not in pipe.blocks_seen_during_load
    assert "prompt_upsampler" not in pipe.components_seen_during_load
    assert pipe.device == "cuda:0"


def test_enabled_modular_feature_requires_loaded_component(fake_dependencies) -> None:
    with pytest.raises(RuntimeError, match="prompt_upsampler"):
        FakeFlux3Adapter.from_pretrained(
            {"model_path": "/model", "prompt_upsampling": True}, "cuda:0"
        )
    assert "prompt_upsampler" in FakeModularPipeline.last_instance.blocks_seen_during_load
    assert "prompt_upsampler" in FakeModularPipeline.last_instance.components_seen_during_load


def test_modular_cpu_offload_uses_components_manager(fake_dependencies) -> None:
    adapter = FakeFlux3Adapter.from_pretrained(
        {
            "model_path": "/model",
            "prompt_upsampling": False,
            "cpu_offload": True,
        },
        "cuda:3",
    )
    pipe = adapter.pipeline
    assert pipe.device is None
    assert isinstance(pipe._components_manager, FakeComponentsManager)
    assert pipe._components_manager.offload_device == "cuda:3"


def test_modular_denoise_loop_reports_each_progress_step() -> None:
    class FakeLoop:
        sub_blocks = {}

        def loop_step(self):
            pass

        def run(self) -> None:
            with self.progress_bar(total=4) as progress:
                for _ in range(4):
                    progress.update()

    loop = FakeLoop()
    root = SimpleNamespace(sub_blocks={"nested": SimpleNamespace(sub_blocks={"denoise": loop})})
    pipeline = SimpleNamespace(_blocks=root)
    adapter = FakeFlux3Adapter(pipeline, {}, "cuda:0")
    reported = []

    with adapter._modular_progress_bridge(
        lambda step, total: reported.append((step, total)), None
    ):
        loop.run()

    assert reported == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert "progress_bar" not in loop.__dict__
