from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# The bridge runs with another Python executable, so expose the workbench root
# before importing the shared parent-class runtime helpers.
WORKBENCH_ROOT = Path(__file__).resolve().parents[2]
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from src.pipelines.base import BaseGenerationPipeline


def emit(kind: str, **payload: Any) -> None:
    print("WORKBENCH_JSON " + json.dumps({"type": kind, **payload}, ensure_ascii=False), flush=True)


def input_box(request: dict[str, Any], image_size: tuple[int, int]) -> list[int]:
    box = request.get("extra", {}).get("input_box")
    if isinstance(box, list) and len(box) == 4:
        values = [int(value) for value in box]
    else:
        iw, ih = image_size
        left = (int(request["width"]) - iw) // 2
        top = (int(request["height"]) - ih) // 2
        values = [left, top, left + iw, top + ih]
    left, top, right, bottom = values
    if left < 0 or top < 0 or right > int(request["width"]) or bottom > int(request["height"]):
        raise ValueError("输入图像不能完整放入目标画布，请增大目标宽高")
    if (right - left, bottom - top) != image_size:
        raise ValueError("input_box 尺寸与输入图像尺寸不一致")
    return values


class ProOutBackend:
    def __init__(self, config: dict[str, Any], device: str) -> None:
        import torch

        repo = Path(config["research_path"]).resolve()
        sys.path.insert(0, str(repo))
        from pipeline import ProOutPipeline

        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[str(config.get("precision", "bf16"))]
        self.pipe = ProOutPipeline.from_pretrained(
            config["checkpoint"],
            pretrained_model_path=config["base_model"],
            device=device,
            torch_dtype=dtype,
        )
        BaseGenerationPipeline.configure_modules_device(
            (self.pipe.denoiser, self.pipe.vae, self.pipe.text_encoder),
            config,
            device,
        )
        self.config = config

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image

        with Image.open(request["input_image"]) as opened:
            size = opened.size
        box = input_box(request, size)
        emit("progress", step=0, total=int(request["num_inference_steps"]))
        result = self.pipe(
            request["input_image"],
            prompt=request["prompt"],
            width=int(request["width"]),
            height=int(request["height"]),
            input_left=box[0],
            input_top=box[1],
            window_size=int(self.config.get("window_size", 512)),
            overlap=int(self.config.get("overlap", 256)),
            window_schedule=str(self.config.get("window_schedule", "auto")),
            num_inference_steps=int(request["num_inference_steps"]),
            guidance_scale=float(self.config.get("guidance_scale", 7.0)),
            ddim_eta=float(self.config.get("ddim_eta", 0.0)),
            seed=int(request["seed"]),
            negative_prompt=request.get("negative_prompt") or None,
        )
        result.images[0].save(request["output"])
        emit("progress", step=int(request["num_inference_steps"]), total=int(request["num_inference_steps"]))
        return result.metadata


class PQDiffBackend:
    def __init__(self, config: dict[str, Any], device: str) -> None:
        repo = Path(config["research_path"]).resolve()
        sys.path.insert(0, str(repo))
        from scripts.eval_wikiart_benchmark import PQDiffPipeline, _checkpoint_state

        self.module = sys.modules["scripts.eval_wikiart_benchmark"]
        self.args = SimpleNamespace(
            device=device,
            base_model=str(config["base_model"]),
            seed=int(config.get("seed", 42)),
            steps=int(config.get("steps", 50)),
        )
        self.pipe = PQDiffPipeline(self.args, _checkpoint_state(Path(config["checkpoint"])))
        BaseGenerationPipeline.configure_modules_device(
            (self.pipe.model, self.pipe.vae), config, device
        )

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image

        with Image.open(request["input_image"]) as opened:
            source = opened.convert("RGB")
        box = input_box(request, source.size)
        row = {
            "id": 0,
            "input_image_path": request["input_image"],
            "width": int(request["width"]),
            "height": int(request["height"]),
            "evaluation_known_box": box,
        }
        self.args.seed = int(request["seed"])
        self.args.steps = int(request["num_inference_steps"])
        emit("progress", step=0, total=self.args.steps)
        paper = self.pipe.generate([row])[0]
        center = source.resize((self.module.PAPER_INPUT_SIZE, self.module.PAPER_INPUT_SIZE), Image.Resampling.BICUBIC)
        paper.paste(center, (self.module.PAPER_CENTER_MARGIN, self.module.PAPER_CENTER_MARGIN))
        generated = paper.resize((row["width"], row["height"]), Image.Resampling.LANCZOS)
        generated.paste(source, (box[0], box[1]))
        generated.save(request["output"])
        emit("progress", step=self.args.steps, total=self.args.steps)
        return {
            "input_box": box,
            "canvas_width": row["width"],
            "canvas_height": row["height"],
            "seed": self.args.seed,
            "method": "PQDiff",
            "prompt_used": False,
        }


class BlueOutBackend:
    def __init__(self, config: dict[str, Any], device: str) -> None:
        repo = Path(config["research_path"]).resolve()
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "scripts"))
        from scripts.eval_wikiart_benchmark import BlueOutBenchmarkPipeline

        self.args = SimpleNamespace(
            device=device,
            precision=str(config.get("precision", "fp16")),
            base_model=str(config["base_model"]),
            instdiff_model=str(config["instdiff_model"]),
            instdiff_variant=str(config.get("instdiff_variant", "fp16")),
            instdiff_config_only=bool(config.get("instdiff_config_only", True)),
            global_ckpt=str(config["global_checkpoint"]),
            main_ckpt=str(config["main_checkpoint"]),
            steps=int(config.get("steps", 30)),
            guidance_scale=float(config.get("guidance_scale", 3.0)),
            eta=float(config.get("eta", 0.0)),
            seed=int(config.get("seed", 42)),
            order=str(config.get("order", "N,E,S,W,NE,NW,SE,SW")),
            global_token_scale=float(config.get("global_token_scale", 1.0)),
            forward_strength=float(config.get("forward_strength", 1.0)),
            global_mask_dilate_px=int(config.get("global_mask_dilate_px", 1)),
            aam_iters=int(config.get("aam_iters", 10)),
            aam_lr=float(config.get("aam_lr", 1e-2)),
            aam_denoising_steps=int(config.get("aam_denoising_steps", 1)),
            aam_stop_loss=float(config.get("aam_stop_loss", 0.55)),
            aam_res=int(config.get("aam_res", 16)),
            aam_tau_out=float(config.get("aam_tau_out", 0.15)),
            aam_lambda_tau=float(config.get("aam_lambda_tau", 1.0)),
            local_prompt=str(config.get("local_prompt", "harmonized painting, high resolution, best quality")),
            negative_prompt=str(config.get("negative_prompt", "ugly, watermark, signature")),
        )
        self.pipe = BlueOutBenchmarkPipeline(self.args)
        BaseGenerationPipeline.configure_modules_device(
            (
                self.pipe.vae,
                self.pipe.text_encoder,
                self.pipe.global_unet,
                self.pipe.main_unet,
                self.pipe.patch_unifusion,
            ),
            config,
            device,
        )

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image

        with Image.open(request["input_image"]) as opened:
            source = opened.convert("RGB")
        width, height = int(request["width"]), int(request["height"])
        box = input_box(request, source.size)
        steps = int(request["num_inference_steps"])
        self.args.seed = int(request["seed"])
        self.args.steps = steps
        self.args.negative_prompt = str(request.get("negative_prompt") or self.args.negative_prompt)
        emit("progress", step=0, total=steps)
        generated = self.pipe({
            "id": 0,
            "input_image_path": request["input_image"],
            "prompt": str(request["prompt"]),
            "width": width,
            "height": height,
            "evaluation_known_box": box,
        })
        generated.save(request["output"])
        emit("progress", step=steps, total=steps)
        return {
            "input_box": box,
            "canvas_width": width,
            "canvas_height": height,
            "seed": int(request["seed"]),
            "method": "BlueOut",
            "layout_condition": "empty_no_benchmark_layout",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["proout", "pqdiff", "blueout"], required=True)
    method = parser.parse_args().method
    line = sys.stdin.readline()
    if not line:
        return
    initialize = json.loads(line)
    try:
        backend_cls = {"proout": ProOutBackend, "pqdiff": PQDiffBackend, "blueout": BlueOutBackend}[method]
        backend = backend_cls(initialize["config"], initialize["device"])
        emit("ready", ok=True)
    except Exception as exc:
        emit("ready", ok=False, error=f"{exc}\n{traceback.format_exc()}")
        return
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if message.get("type") != "generate":
                continue
            metadata = backend.generate(message["request"])
            emit("result", output=message["request"]["output"], metadata=metadata)
        except Exception as exc:
            emit("error", error=f"{exc}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
