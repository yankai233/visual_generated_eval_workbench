from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ProgressCallback = Callable[[int, int], None]


class GenerationCancelled(RuntimeError):
    """Raised by a pipeline when a generation is cancelled cooperatively."""


@dataclass(slots=True)
class GenerationRequest:
    id: str
    task_type: str
    prompt: str
    negative_prompt: str = ""
    input_images: list[Path] = field(default_factory=list)
    seed: int | None = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 28
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    images: Sequence[Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineControl:
    """Process-safe cooperative pause/cancel control passed to pipelines."""

    def __init__(self, pause_event: Any, cancel_event: Any) -> None:
        self._pause_event = pause_event
        self._cancel_event = cancel_event
        self._paused_seconds = 0.0
        self._pause_started: float | None = None

    def active_time(self) -> float:
        """Monotonic worker clock excluding time spent at cooperative pause points."""
        now = time.monotonic()
        return now - self._paused_seconds - (max(0.0, now - self._pause_started) if self._pause_started is not None else 0.0)

    @property
    def cancelled(self) -> bool:
        return bool(self._cancel_event.is_set())

    @property
    def paused(self) -> bool:
        return bool(self._pause_event.is_set())

    def checkpoint(self) -> None:
        if self.cancelled:
            raise GenerationCancelled("generation cancelled")
        if self._pause_event.is_set():
            self._pause_started = time.monotonic()
            try:
                while self._pause_event.is_set():
                    if self._cancel_event.wait(timeout=0.2):
                        raise GenerationCancelled("generation cancelled")
            finally:
                self._paused_seconds += max(0.0, time.monotonic() - self._pause_started)
                self._pause_started = None


class BaseGenerationPipeline(ABC):
    """All workbench pipelines must implement this interface."""

    pipeline_name = ""
    display_name = ""
    description = ""
    pipeline_group = "Other"
    supported_tasks: tuple[str, ...] = ("T2I",)
    parameter_size = "未说明"
    max_reference_images: int | None = 0
    reference_limit_note = ""
    supported_devices: tuple[str, ...] = ("cuda",)
    dependency_requirements: Mapping[str, str] = {}
    supports_cpu_offload = True
    default_num_inference_steps = 28

    @classmethod
    def metadata(cls) -> dict[str, Any]:
        return {
            "name": cls.pipeline_name,
            "display_name": cls.display_name or cls.pipeline_name,
            "description": cls.description,
            "group": cls.pipeline_group,
            "supported_tasks": list(cls.supported_tasks),
            "parameter_size": cls.parameter_size,
            "max_reference_images": cls.max_reference_images,
            "reference_limit_note": cls.reference_limit_note,
            "supported_devices": list(cls.supported_devices),
            "supports_cpu_offload": cls.supports_cpu_offload,
            "default_num_inference_steps": cls.default_num_inference_steps,
        }

    @staticmethod
    def cpu_offload_enabled(config: Mapping[str, Any]) -> bool:
        """Return the shared CPU-offload switch used by every pipeline type."""
        return bool(config.get("cpu_offload", False))

    @classmethod
    def create_cpu_offload_manager(
        cls, config: Mapping[str, Any], manager_factory: Callable[[], Any]
    ) -> Any | None:
        """Create a manager only when a modular pipeline requests CPU offload."""
        if not cls.cpu_offload_enabled(config):
            return None
        return manager_factory()

    @staticmethod
    def _call_cpu_offload(enable_offload: Callable[..., Any], device: str) -> None:
        """Call old and new Diffusers offload APIs without losing the GPU index."""
        try:
            parameters = inspect.signature(enable_offload).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "device" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            enable_offload(device=device)
            return
        if "gpu_id" in parameters:
            device_index = device.partition(":")[2]
            enable_offload(gpu_id=int(device_index) if device_index else 0)
            return
        enable_offload()

    @classmethod
    def configure_model_device(
        cls,
        model: Any,
        config: Mapping[str, Any],
        device: str,
        *,
        move_when_disabled: bool = True,
    ) -> Any:
        """Apply one device/offload policy to Diffusers, modular, or torch models."""
        if not cls.cpu_offload_enabled(config):
            if move_when_disabled:
                move = getattr(model, "to", None)
                if not callable(move):
                    raise RuntimeError(f"{cls.display_name} 的模型不支持移动到 {device}")
                move(device)
            return model

        components_manager = getattr(model, "_components_manager", None)
        enable_auto_offload = getattr(components_manager, "enable_auto_cpu_offload", None)
        if callable(enable_auto_offload):
            enable_auto_offload(device=device)
            return model

        enable_model_offload = getattr(model, "enable_model_cpu_offload", None)
        if callable(enable_model_offload):
            cls._call_cpu_offload(enable_model_offload, device)
            return model

        try:
            import torch
            from accelerate import cpu_offload
        except ImportError as exc:
            raise RuntimeError(
                f"{cls.display_name} 启用 CPU Offload 需要安装 torch 和 accelerate"
            ) from exc
        if isinstance(model, torch.nn.Module):
            cpu_offload(model, execution_device=torch.device(device))
            return model
        raise RuntimeError(
            f"{cls.display_name} 无法启用 CPU Offload：模型没有受支持的卸载接口"
        )

    @classmethod
    def configure_modules_device(
        cls,
        modules: Sequence[Any],
        config: Mapping[str, Any],
        device: str,
        *,
        move_when_disabled: bool = False,
    ) -> None:
        """Apply the shared policy to each module of a custom multi-model pipeline."""
        for module in modules:
            if module is not None:
                cls.configure_model_device(
                    module,
                    config,
                    device,
                    move_when_disabled=move_when_disabled,
                )

    @classmethod
    def inspect_config(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        """Return a cheap, side-effect-free readiness report for one YAML config."""
        return {
            "available": True,
            "issues": [],
            "configured_model_path": config.get("model_path"),
            "resolved_model_path": config.get("model_path"),
        }

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, config: Mapping[str, Any], device: str
    ) -> "BaseGenerationPipeline":
        """Load the model on ``device`` from a validated YAML mapping."""

    @abstractmethod
    def __call__(
        self,
        request: GenerationRequest,
        *,
        progress_callback: ProgressCallback | None = None,
        control: PipelineControl | None = None,
    ) -> GenerationResult:
        """Generate images for one normalized request."""

