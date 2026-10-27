from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from src.pipelines.base import GenerationRequest
from src.pipelines.uno.pipeline import UNOWorkbenchPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Load UNO through the workbench adapter and generate one image.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    request = GenerationRequest(
        id="uno-smoke",
        task_type="TI2I" if args.input else "T2I",
        prompt="A clock on the beach is under a red sun umbrella",
        input_images=[args.input] if args.input else [],
        seed=0,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
    )
    print(json.dumps({"config": str(args.config), "request": request.id, "device": args.device}), flush=True)
    started = time.monotonic()
    pipeline = UNOWorkbenchPipeline.from_pretrained(config, args.device)
    print(f"LOAD_SUCCEEDED seconds={time.monotonic() - started:.2f}", flush=True)
    progress: list[tuple[int, int]] = []
    result = pipeline(request, progress_callback=lambda step, total: (progress.append((step, total)), print(f"PROGRESS {step}/{total}", flush=True)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(args.output)
    print(
        json.dumps(
            {
                "status": "success",
                "output": str(args.output),
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "progress": progress,
                "metadata": result.metadata,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
