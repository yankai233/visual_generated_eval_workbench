from __future__ import annotations

from typing import Any, Mapping

from src.pipelines.base import GenerationRequest, PipelineControl, ProgressCallback
from src.pipelines.diffusers_adapter import DiffusersPipelineAdapter


class Flux3KleinWorkbenchPipeline(DiffusersPipelineAdapter):
    """Workbench adapter for the step-distilled FLUX.3 Image Klein model."""

    pipeline_name = "flux_3_klein"
    pipeline_group = "FLUX"
    display_name = "FLUX.3 Image Klein"
    description = "FLUX.3 的 16 步蒸馏版，支持文生图、单图编辑和多参考图生成。"
    supported_tasks = ("T2I", "TI2I", "MultiReference")
    parameter_size = "5.6B 图像 Transformer"
    max_reference_images = 10
    reference_limit_note = "最多 10 张参考图；默认直接使用 Prompt，不加载独立 Prompt Upsampler"
    supported_devices = ("cuda",)
    default_num_inference_steps = 16
    diffusers_class_name = "ModularPipeline"
    modular_pipeline = True
    modular_feature_blocks = {"prompt_upsampling": "prompt_upsampler"}
    modular_required_components = ("text_encoder", "tokenizer", "transformer", "vae")
    modular_feature_components = {
        "prompt_upsampling": ("prompt_upsampler", "prompt_upsampler_processor")
    }
    dependency_requirements = {
        "torch": "2.4",
        "diffusers": "0.41.0.dev0",
        "transformers": "5.17",
    }

    @classmethod
    def modular_component_load_kwargs(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        # Transformers detects this tokenizer's legacy Mistral regex and asks
        # callers to opt into the corrected expression at construction time.
        return {"fix_mistral_regex": {"tokenizer": True}}

    def _build_kwargs(
        self,
        request: GenerationRequest,
        progress_callback: ProgressCallback | None,
        control: PipelineControl | None,
    ) -> dict[str, Any]:
        kwargs = super()._build_kwargs(request, progress_callback, control)
        # The distilled checkpoint is trained without classifier-free guidance.
        # Passing CFG-only inputs to its modular blocks is both misleading and,
        # depending on the installed Diffusers revision, may be rejected.
        kwargs.pop("guidance_scale", None)
        kwargs.pop("negative_prompt", None)
        return kwargs
