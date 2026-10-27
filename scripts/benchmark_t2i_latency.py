from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from src.pipelines.base import GenerationRequest
from src.pipelines.registry import get_pipeline


def synchronize(torch, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure one workbench T2I request.")
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        default="一只橙色猫坐在木质窗台上，清晨阳光，写实摄影，高细节",
    )
    args = parser.parse_args()

    import torch

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pipeline_cls = get_pipeline(args.pipeline)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    pipeline = pipeline_cls.from_pretrained(config, args.device)
    synchronize(torch, args.device)
    load_seconds = time.perf_counter() - load_started

    request = GenerationRequest(
        id="latency",
        task_type="T2I",
        prompt=args.prompt,
        seed=args.seed,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
    )
    progress: list[dict[str, float | int]] = []
    generation_started = time.perf_counter()

    def report(step: int, total: int) -> None:
        synchronize(torch, args.device)
        progress.append(
            {
                "step": int(step),
                "total": int(total),
                "elapsed_seconds": time.perf_counter() - generation_started,
            }
        )
        print(f"DENOISE_PROGRESS {step}/{total}", flush=True)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    result = pipeline(request, progress_callback=report)
    synchronize(torch, args.device)
    generation_seconds = time.perf_counter() - generation_started
    peak_memory_gib = (
        torch.cuda.max_memory_allocated(args.device) / 1024**3
        if args.device.startswith("cuda")
        else None
    )
    image_path = args.output_dir / f"{args.pipeline}.png"
    result.images[0].save(image_path)
    record = {
        "pipeline": args.pipeline,
        "display_name": pipeline_cls.display_name,
        "model_path": config.get("model_path"),
        "cpu_offload": bool(config.get("cpu_offload", False)),
        "device": args.device,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "prompt": args.prompt,
        "load_seconds": round(load_seconds, 4),
        "generation_seconds": round(generation_seconds, 4),
        "seconds_per_step": round(generation_seconds / args.steps, 4),
        "peak_torch_memory_gib": round(peak_memory_gib, 4) if peak_memory_gib is not None else None,
        "progress_events": progress,
        "image": str(image_path),
    }
    result_path = args.output_dir / f"{args.pipeline}.json"
    result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BENCHMARK_RESULT " + json.dumps(record, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
