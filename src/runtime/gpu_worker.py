from __future__ import annotations

import gc
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from src.pipelines.base import (
    GenerationCancelled,
    GenerationRequest,
    PipelineControl,
)
from src.pipelines.registry import get_pipeline
from src.data.sample_identity import atomic_json
from src.runtime.timing import StepTiming


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    atomic_json(path, json.loads(json.dumps(data, default=_json_default)))


def _release_pipeline(instance: Any) -> None:
    del instance
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_gpu_worker(
    job_id: str,
    physical_gpu_id: str,
    work_units: list[dict[str, Any]],
    device_lock: Any,
    pause_event: Any,
    cancel_event: Any,
    message_queue: Any,
) -> None:
    if physical_gpu_id != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu_id
        device = "cuda:0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"

    control = PipelineControl(pause_event, cancel_event)
    current_key: tuple[str, str] | None = None
    current_pipeline: Any = None
    lock_acquired = False
    try:
        message_queue.put(
            {"type": "worker_waiting", "job_id": job_id, "gpu_id": physical_gpu_id}
        )
        while not cancel_event.is_set():
            lock_acquired = device_lock.acquire(timeout=0.25)
            if lock_acquired:
                break
        if not lock_acquired:
            return
        for unit in work_units:
            if cancel_event.is_set():
                break
            control.checkpoint()
            unit_id = unit["unit_id"]
            output_dir = Path(unit["output_dir"])
            save_images = unit.get("save_images", True)
            if save_images:
                output_dir.mkdir(parents=True, exist_ok=True)
            message_queue.put(
                {
                    "type": "unit_started",
                    "job_id": job_id,
                    "gpu_id": physical_gpu_id,
                    "unit_id": unit_id,
                    "pipeline": unit["pipeline"],
                    "sample_id": unit["sample"]["id"],
                    "total_steps": unit["sample"].get("num_inference_steps", 28),
                }
            )
            started_at = time.time()
            try:
                sample = unit["sample"]
                if save_images:
                    _write_json(output_dir / "request.json", dict(sample))
                    (output_dir / "config.yaml").write_text(
                        yaml.safe_dump(unit["config"], allow_unicode=True, sort_keys=False),
                        encoding="utf-8",
                    )
                cache_key = (unit["pipeline"], unit["config_name"])
                if cache_key != current_key:
                    if current_pipeline is not None:
                        _release_pipeline(current_pipeline)
                    pipeline_cls = get_pipeline(unit["pipeline"])
                    current_pipeline = pipeline_cls.from_pretrained(unit["config"], device)
                    current_key = cache_key

                sample = unit["sample"]
                request = GenerationRequest(
                    id=sample["id"],
                    task_type=sample["task_type"],
                    prompt=sample.get("prompt", ""),
                    negative_prompt=sample.get("negative_prompt", ""),
                    input_images=[Path(path) for path in sample.get("input_images", [])],
                    seed=sample.get("seed"),
                    width=sample.get("width", 1024),
                    height=sample.get("height", 1024),
                    num_inference_steps=sample.get("num_inference_steps", 28),
                    extra=sample.get("extra", {}),
                )

                control.checkpoint()
                generation_started = control.active_time()
                step_timing = StepTiming()
                message_queue.put({"type": "unit_generation_started", "job_id": job_id, "gpu_id": physical_gpu_id, "unit_id": unit_id})

                def report_progress(step: int, total: int) -> None:
                    control.checkpoint()
                    total = max(1, int(total))
                    step = max(0, min(int(step), total))
                    now = control.active_time()
                    message_queue.put(
                        {
                            "type": "denoise_progress",
                            "job_id": job_id,
                            "gpu_id": physical_gpu_id,
                            "unit_id": unit_id,
                            "step": step,
                            "total": total,
                            "eta_seconds": step_timing.update(step, total, now),
                            "generation_elapsed_seconds": max(0.0, now - generation_started),
                        }
                    )

                result = current_pipeline(
                    request,
                    progress_callback=report_progress,
                    control=control,
                )
                control.checkpoint()
                saved_images: list[str] = []
                preview_images: list[bytes] = []
                for image_index, image in enumerate(result.images):
                    image_name = "generated.png" if len(result.images) == 1 else f"generated_{image_index}.png"
                    image_path = output_dir / image_name
                    def encode(opened):
                        if save_images:
                            opened.save(image_path)
                            saved_images.append(str(image_path))
                        else:
                            buffer = io.BytesIO()
                            opened.save(buffer, format="PNG")
                            preview_images.append(buffer.getvalue())

                    if isinstance(image, Image.Image):
                        encode(image)
                    elif isinstance(image, (str, Path)):
                        with Image.open(image) as opened:
                            encode(opened)
                    else:
                        raise TypeError(f"Pipeline 返回了不支持的图像类型: {type(image)!r}")

                result_payload = {
                    "schema_version": 2,
                    "status": "completed",
                    "job_id": job_id,
                    "pipeline": unit["pipeline"],
                    "config_name": unit["config_name"],
                    "sample_id": sample["id"],
                    "benchmark": unit.get("benchmark"),
                    "batch_started_at": unit.get("batch_started_at"),
                    "comparison": sample.get("comparison", {}),
                    "physical_gpu_id": physical_gpu_id,
                    "device": device,
                    "duration_seconds": round(time.time() - started_at, 4),
                    "generation_duration_seconds": round(max(0.0, control.active_time() - generation_started), 4),
                    "images": saved_images,
                    "metadata": result.metadata,
                }
                if save_images:
                    _write_json(output_dir / "result.json", result_payload)
                message_queue.put(
                    {
                        "type": "unit_completed",
                        "job_id": job_id,
                        "gpu_id": physical_gpu_id,
                        "unit_id": unit_id,
                        "pipeline": unit["pipeline"],
                        "config_name": unit["config_name"],
                        "benchmark": unit.get("benchmark"),
                        "sample_id": sample["id"],
                        "output_dir": str(output_dir),
                        "images": saved_images,
                        "duration_seconds": result_payload["duration_seconds"],
                        "generation_duration_seconds": result_payload["generation_duration_seconds"],
                        "request": sample,
                        "metadata": result.metadata,
                        "preview_images": preview_images,
                        "preview_context": json.loads(json.dumps({
                            "unit_id": unit_id, "output_dir": str(output_dir),
                            "request": sample, "config": unit["config"], "result": result_payload,
                        }, default=_json_default)) if not save_images else None,
                    }
                )
            except GenerationCancelled:
                if save_images:
                    _write_json(
                        output_dir / "result.json",
                        {"status": "cancelled", "job_id": job_id, "unit_id": unit_id},
                    )
                message_queue.put(
                    {
                        "type": "unit_cancelled",
                        "job_id": job_id,
                        "gpu_id": physical_gpu_id,
                        "unit_id": unit_id,
                    }
                )
                break
            except Exception as exc:
                error_payload = {
                    "status": "failed",
                    "job_id": job_id,
                    "unit_id": unit_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                if save_images:
                    _write_json(output_dir / "result.json", error_payload)
                message_queue.put(
                    {
                        "type": "unit_failed",
                        "job_id": job_id,
                        "gpu_id": physical_gpu_id,
                        "unit_id": unit_id,
                        "error": str(exc),
                    }
                )
    except GenerationCancelled:
        pass
    except BaseException as exc:
        message_queue.put(
            {
                "type": "worker_failed",
                "job_id": job_id,
                "gpu_id": physical_gpu_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if current_pipeline is not None:
            _release_pipeline(current_pipeline)
        if lock_acquired:
            device_lock.release()
        message_queue.put(
            {"type": "worker_done", "job_id": job_id, "gpu_id": physical_gpu_id}
        )

