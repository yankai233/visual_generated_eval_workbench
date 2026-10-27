from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import yaml

from src.pipelines.base import GenerationRequest
from src.pipelines.flux_3_klein_base.pipeline import Flux3KleinBaseWorkbenchPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Load FLUX.3 through the workbench adapter and reproduce one request.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import diffusers
    import torch
    import transformers

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sample = json.loads(args.request.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "diffusers": diffusers.__version__,
                "transformers": transformers.__version__,
                "visible_cuda_devices": torch.cuda.device_count(),
                "device": args.device,
                "model_path": config.get("model_path"),
                "prompt_upsampling": config.get("prompt_upsampling"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    try:
        adapter = Flux3KleinBaseWorkbenchPipeline.from_pretrained(config, args.device)
    except Exception:
        print("LOAD_FAILED", flush=True)
        traceback.print_exc()
        return 2

    pipeline = adapter.pipeline
    print("LOAD_RETURNED", flush=True)
    for name in (
        "prompt_upsampler",
        "prompt_upsampler_processor",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    ):
        value = getattr(pipeline, name, None)
        print(f"COMPONENT {name}: {type(value).__name__ if value is not None else 'None'}", flush=True)
    blocks = getattr(getattr(pipeline, "_blocks", None), "sub_blocks", {})
    print(f"BLOCKS: {list(blocks)}", flush=True)

    request = GenerationRequest(
        id=str(sample["id"]),
        task_type=str(sample["task_type"]),
        prompt=str(sample.get("prompt", "")),
        negative_prompt=str(sample.get("negative_prompt", "")),
        input_images=[Path(path) for path in sample.get("input_images", [])],
        seed=42,
        width=256,
        height=256,
        num_inference_steps=1,
        extra=dict(sample.get("extra", {})),
    )
    try:
        result = adapter(request)
    except Exception:
        print("CALL_FAILED", flush=True)
        traceback.print_exc()
        return 3

    print(f"CALL_SUCCEEDED images={len(result.images)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
