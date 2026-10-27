from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from src.pipelines.base import (
    GenerationRequest,
    GenerationResult,
    PipelineControl,
    ProgressCallback,
)
from src.pipelines.outpainting_adapter import ExternalOutpaintingPipeline


class UNOWorkbenchPipeline(ExternalOutpaintingPipeline):
    """UNO adapter executed in the method's isolated Conda environment."""

    pipeline_name = "uno"
    pipeline_group = "Other"
    display_name = "UNO"
    description = "基于 FLUX.1-dev 的主体驱动生成，支持文生图、单参考图和多参考图。"
    supported_tasks = ("T2I", "TI2I", "MultiReference")
    parameter_size = "FLUX.1-dev 主干 + rank-512 LoRA"
    max_reference_images = 4
    reference_limit_note = "支持 0–4 张参考图"
    supported_devices = ("cuda",)
    dependency_requirements = {
        "torch": "2.4",
        "transformers": "4.43",
        "safetensors": "0.5",
    }

    method = "uno"
    bridge_script = Path(__file__).resolve().parents[1] / "uno_bridge.py"
    bridge_stderr = subprocess.DEVNULL
    required_files = ("flux_checkpoint", "ae_checkpoint", "lora_checkpoint")
    required_directories = ("repo_path", "t5_path", "clip_path")

    @classmethod
    def inspect_config(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        status = super().inspect_config(config)
        status["configured_model_path"] = config.get("lora_checkpoint")
        status["resolved_model_path"] = config.get("lora_checkpoint")
        return status

    @staticmethod
    def _validate_request(request: GenerationRequest) -> None:
        reference_count = len(request.input_images)
        if request.task_type == "T2I" and reference_count != 0:
            raise ValueError("UNO 文生图任务不接受参考图")
        if request.task_type == "TI2I" and reference_count != 1:
            raise ValueError("UNO 单参考图任务需要且只能输入 1 张图")
        if request.task_type == "MultiReference" and not 1 <= reference_count <= 4:
            raise ValueError("UNO 多参考图任务需要输入 1–4 张图")
        if request.task_type not in UNOWorkbenchPipeline.supported_tasks:
            raise ValueError(f"UNO 不支持任务类型 {request.task_type}")

    def __call__(
        self,
        request: GenerationRequest,
        *,
        progress_callback: ProgressCallback | None = None,
        control: PipelineControl | None = None,
    ) -> GenerationResult:
        self._validate_request(request)
        temporary = Path(tempfile.mkdtemp(prefix="workbench-uno-"))
        output = temporary / "generated.png"
        try:
            self._send(
                {
                    "type": "generate",
                    "request": {
                        "id": request.id,
                        "task_type": request.task_type,
                        "prompt": request.prompt,
                        "input_images": [str(path) for path in request.input_images],
                        "seed": request.seed if request.seed is not None else 42,
                        "width": request.width,
                        "height": request.height,
                        "num_inference_steps": request.num_inference_steps,
                        "extra": request.extra,
                        "output": str(output),
                    },
                }
            )
            message = self._receive(
                "result",
                timeout=float(self.config.get("generation_timeout", 3600)),
                control=control,
                progress_callback=progress_callback,
            )
            with Image.open(message["output"]) as opened:
                image = opened.convert("RGB").copy()
            return GenerationResult(images=[image], metadata=message.get("metadata", {}))
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

