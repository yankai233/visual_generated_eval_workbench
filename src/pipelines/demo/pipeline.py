from __future__ import annotations

import random
import time
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageFont

from ..base import (
    BaseGenerationPipeline,
    GenerationRequest,
    GenerationResult,
    PipelineControl,
    ProgressCallback,
)


class DemoPipeline(BaseGenerationPipeline):
    """A lightweight pipeline used to verify the complete workbench flow."""

    pipeline_name = "demo"
    pipeline_group = "Test"
    display_name = "Demo Pipeline"
    description = "无需模型权重，用于验证页面、调度、暂停和输出目录。"
    supported_tasks = ("T2I", "TI2I", "MultiReference")
    parameter_size = "无模型权重"
    max_reference_images = None
    reference_limit_note = "不限（仅用于工作台测试）"
    supported_devices = ("cpu", "cuda")

    def __init__(self, config: Mapping[str, Any], device: str) -> None:
        self.config = dict(config)
        self.device = device
        self.step_delay = max(0.0, float(self.config.get("step_delay", 0.04)))

    @classmethod
    def from_pretrained(
        cls, config: Mapping[str, Any], device: str
    ) -> "DemoPipeline":
        return cls(config, device)

    def __call__(
        self,
        request: GenerationRequest,
        *,
        progress_callback: ProgressCallback | None = None,
        control: PipelineControl | None = None,
    ) -> GenerationResult:
        steps = max(1, request.num_inference_steps)
        for step in range(steps):
            if control is not None:
                control.checkpoint()
            if self.step_delay:
                time.sleep(self.step_delay)
            if progress_callback is not None:
                progress_callback(step + 1, steps)

        seed = request.seed if request.seed is not None else random.randrange(2**31)
        rng = random.Random(seed)
        width = max(256, min(request.width, 1536))
        height = max(256, min(request.height, 1536))
        color_a = tuple(rng.randint(35, 155) for _ in range(3))
        color_b = tuple(min(255, value + rng.randint(50, 100)) for value in color_a)
        image = Image.new("RGB", (width, height), color_a)
        draw = ImageDraw.Draw(image)
        for y in range(height):
            ratio = y / max(1, height - 1)
            color = tuple(
                int(color_a[i] * (1 - ratio) + color_b[i] * ratio) for i in range(3)
            )
            draw.line((0, y, width, y), fill=color)

        font = ImageFont.load_default(size=20)
        lines = [
            "Visual Generation Eval Workbench",
            f"Task: {request.task_type}",
            f"Device: {self.device}",
            f"Seed: {seed}",
            f"Prompt: {request.prompt[:100] or '(empty)'}",
            f"Reference images: {len(request.input_images)}",
        ]
        box = (32, 32, width - 32, min(height - 32, 240))
        draw.rounded_rectangle(box, radius=18, fill=(255, 255, 255, 220))
        draw.multiline_text((54, 52), "\n".join(lines), fill=(20, 31, 26), font=font, spacing=10)
        return GenerationResult(images=[image], metadata={"seed": seed, "demo": True})

