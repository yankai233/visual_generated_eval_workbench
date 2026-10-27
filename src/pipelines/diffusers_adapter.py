from __future__ import annotations

from contextlib import contextmanager
import inspect
from pathlib import Path
from types import MethodType
from typing import Any, Mapping

from PIL import Image

from src.utils.model_paths import inspect_model_config, resolve_pretrained_path

from .base import (
    BaseGenerationPipeline,
    GenerationRequest,
    GenerationResult,
    PipelineControl,
    ProgressCallback,
)


class DiffusersPipelineAdapter(BaseGenerationPipeline):
    """Lazy Diffusers adapter shared by the real model plugins."""

    diffusers_class_name = ""
    modular_pipeline = False
    modular_feature_blocks: Mapping[str, str] = {}
    modular_required_components: tuple[str, ...] = ()
    modular_feature_components: Mapping[str, tuple[str, ...]] = {}

    def __init__(self, pipeline: Any, config: Mapping[str, Any], device: str) -> None:
        self.pipeline = pipeline
        self.config = dict(config)
        self.device = device

    @classmethod
    def inspect_config(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        return inspect_model_config(config, cls.dependency_requirements)

    @staticmethod
    def _torch_dtype(torch: Any, value: str) -> Any:
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        try:
            return mapping[value]
        except KeyError as exc:
            raise ValueError(f"不支持的 torch_dtype: {value}") from exc

    @classmethod
    def _configure_modular_blocks(cls, pipe: Any, config: Mapping[str, Any]) -> None:
        """Remove optional blocks before Diffusers resolves their component specs."""
        feature_blocks = cls.modular_feature_blocks
        if not feature_blocks:
            return
        blocks = getattr(getattr(pipe, "_blocks", None), "sub_blocks", None)
        if blocks is None:
            raise RuntimeError(f"{cls.display_name} 的 ModularPipeline 不包含可配置 blocks")
        for config_name, block_name in feature_blocks.items():
            if bool(config.get(config_name, False)):
                continue
            blocks.pop(block_name, None)

    @classmethod
    def _validate_modular_components(cls, pipe: Any, config: Mapping[str, Any]) -> None:
        required = list(cls.modular_required_components)
        for config_name, component_names in cls.modular_feature_components.items():
            if bool(config.get(config_name, False)):
                required.extend(component_names)
        missing = [name for name in required if getattr(pipe, name, None) is None]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                f"{cls.display_name} 组件加载失败：{joined}。请检查对应模型路径和权重是否完整"
            )

    @staticmethod
    def _active_modular_component_names(pipe: Any) -> list[str] | None:
        expected = getattr(getattr(pipe, "_blocks", None), "expected_components", None)
        if expected is None:
            return None
        return list(dict.fromkeys(spec.name for spec in expected))

    @classmethod
    def modular_component_load_kwargs(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        """Return model-specific keyword arguments for modular components."""
        return {}

    @classmethod
    def from_pretrained(
        cls, config: Mapping[str, Any], device: str
    ) -> "DiffusersPipelineAdapter":
        if not device.startswith("cuda"):
            raise RuntimeError(f"{cls.display_name} 当前只允许在 CUDA 设备上运行")
        import diffusers
        import torch

        pipeline_class = getattr(diffusers, cls.diffusers_class_name, None)
        if pipeline_class is None:
            raise RuntimeError(
                f"当前 diffusers 不包含 {cls.diffusers_class_name}，请按模型卡升级依赖"
            )
        model_path = resolve_pretrained_path(str(config["model_path"]))
        dtype = cls._torch_dtype(torch, str(config.get("torch_dtype", "bfloat16")))
        if cls.modular_pipeline:
            components_manager = cls.create_cpu_offload_manager(
                config, diffusers.ComponentsManager
            )
            pipe = pipeline_class.from_pretrained(
                str(model_path),
                local_files_only=bool(config.get("local_files_only", True)),
                components_manager=components_manager,
            )
            cls._configure_modular_blocks(pipe, config)
            pipe.load_components(
                names=cls._active_modular_component_names(pipe),
                dtype=dtype,
                **cls.modular_component_load_kwargs(config),
            )
            cls._validate_modular_components(pipe, config)
        else:
            pipe = pipeline_class.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                local_files_only=bool(config.get("local_files_only", True)),
            )
        cls.configure_model_device(pipe, config, device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        return cls(pipe, config, device)

    def _prepare_images(self, paths: list[Path]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        return images

    @staticmethod
    def _walk_modular_blocks(block: Any):
        yield block
        for child in getattr(block, "sub_blocks", {}).values():
            yield from DiffusersPipelineAdapter._walk_modular_blocks(child)

    @contextmanager
    def _modular_progress_bridge(
        self,
        progress_callback: ProgressCallback | None,
        control: PipelineControl | None,
    ):
        """Bridge ModularPipeline loop progress bars to workbench step events."""
        if not self.modular_pipeline or (progress_callback is None and control is None):
            yield
            return

        class WorkbenchProgress:
            def __init__(self, iterable=None, total=None) -> None:
                self.iterable = iterable
                self.total = max(
                    1,
                    int(total if total is not None else len(iterable)),
                )
                self.n = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                self.close()

            def __iter__(self):
                for value in self.iterable:
                    yield value
                    self.update()

            def update(self, amount=1) -> None:
                self.n = min(self.total, self.n + int(amount))
                if control is not None:
                    control.checkpoint()
                if progress_callback is not None:
                    progress_callback(self.n, self.total)

            def close(self) -> None:
                return None

        patched: list[tuple[Any, bool, Any]] = []

        def progress_bar(_block, iterable=None, total=None):
            return WorkbenchProgress(iterable=iterable, total=total)

        root = getattr(self.pipeline, "_blocks", None)
        if root is not None:
            for block in self._walk_modular_blocks(root):
                # LoopSequentialPipelineBlocks owns loop_step and creates the
                # denoising progress bar. Other blocks must keep their behavior.
                if not callable(getattr(block, "loop_step", None)):
                    continue
                had_override = "progress_bar" in getattr(block, "__dict__", {})
                previous = block.__dict__.get("progress_bar") if had_override else None
                patched.append((block, had_override, previous))
                block.progress_bar = MethodType(progress_bar, block)
        try:
            yield
        finally:
            for block, had_override, previous in patched:
                if had_override:
                    block.progress_bar = previous
                else:
                    delattr(block, "progress_bar")

    def _build_kwargs(
        self,
        request: GenerationRequest,
        progress_callback: ProgressCallback | None,
        control: PipelineControl | None,
    ) -> dict[str, Any]:
        import torch

        requested_steps = max(1, request.num_inference_steps)
        pipeline_config = getattr(self.pipeline, "config", None)
        if isinstance(pipeline_config, Mapping):
            sample_sigmas = pipeline_config.get("sample_sigmas")
        else:
            sample_sigmas = getattr(pipeline_config, "sample_sigmas", None)
        # Distilled checkpoints such as Qwen-Image-2.1-Turbo ship a fixed
        # sampling grid. Diffusers ignores num_inference_steps in that case, so
        # progress and ETA must follow the actual sigma count.
        total_steps = len(sample_sigmas) if sample_sigmas else requested_steps

        def callback_on_step_end(
            _pipeline: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]
        ) -> dict[str, Any]:
            if control is not None:
                control.checkpoint()
            if progress_callback is not None:
                progress_callback(step + 1, total_steps)
            return callback_kwargs

        seed = request.seed if request.seed is not None else torch.seed()
        generator_device = self.device if self.device.startswith("cuda") else "cpu"
        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or None,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": requested_steps,
            "guidance_scale": float(self.config.get("guidance_scale", 3.5)),
            "num_images_per_prompt": int(self.config.get("num_images_per_prompt", 1)),
            "generator": torch.Generator(device=generator_device).manual_seed(int(seed)),
        }
        if not self.modular_pipeline:
            kwargs["callback_on_step_end"] = callback_on_step_end
        if "true_cfg_scale" in self.config:
            kwargs["true_cfg_scale"] = float(self.config["true_cfg_scale"])
        if "max_sequence_length" in self.config:
            kwargs["max_sequence_length"] = int(self.config["max_sequence_length"])
        if "use_kv_cache" in self.config:
            kwargs["use_kv_cache"] = bool(self.config["use_kv_cache"])
        if self.modular_pipeline:
            kwargs["output"] = "images"
        if request.input_images:
            images = self._prepare_images(request.input_images)
            kwargs["image"] = images[0] if request.task_type == "TI2I" else images
        return kwargs

    def __call__(
        self,
        request: GenerationRequest,
        *,
        progress_callback: ProgressCallback | None = None,
        control: PipelineControl | None = None,
    ) -> GenerationResult:
        if request.task_type not in self.supported_tasks:
            raise ValueError(f"{self.display_name} 不支持 {request.task_type}")
        if (
            self.max_reference_images is not None
            and len(request.input_images) > self.max_reference_images
        ):
            raise ValueError(
                f"{self.display_name} 最多支持 {self.max_reference_images} 张参考图"
            )
        if control is not None:
            control.checkpoint()
        kwargs = self._build_kwargs(request, progress_callback, control)
        signature = inspect.signature(self.pipeline.__call__)
        if not any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        with self._modular_progress_bridge(progress_callback, control):
            output = self.pipeline(**kwargs)
        images = getattr(output, "images", output)
        if isinstance(images, Image.Image):
            images = [images]
        return GenerationResult(
            images=list(images),
            metadata={"seed": request.seed, "model_path": self.config.get("model_path")},
        )
