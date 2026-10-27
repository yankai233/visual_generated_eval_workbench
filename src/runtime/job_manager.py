from __future__ import annotations

import copy
import multiprocessing as mp
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.data.schemas import PipelineSelection, SampleModel
from src.runtime.gpu_worker import run_gpu_worker
from src.runtime.state_store import JsonJobStateStore
from src.runtime.timing import JobTiming
from src.runtime.preview_store import PreviewStore


TERMINAL_STATUSES = {
    "completed",
    "completed_with_errors",
    "cancelled",
    "failed",
    "interrupted",
}
SAFE_PATH_PART = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_path_part(value: str, fallback: str) -> str:
    cleaned = SAFE_PATH_PART.sub("_", value.strip()).strip("._")
    return cleaned[:120] or fallback


@dataclass
class JobRecord:
    id: str
    status: str
    batch_started_at: str
    total: int
    save_images: bool = True
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    message: str = "等待运行"
    active: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    processes: list[Any] = field(default_factory=list, repr=False)
    pause_event: Any = field(default=None, repr=False)
    cancel_event: Any = field(default=None, repr=False)
    message_queue: Any = field(default=None, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    last_persisted_at: float = field(default=0.0, repr=False)
    timing: JobTiming | None = field(default=None, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            progress = self.completed / self.total if self.total else 0.0
            timing = self.timing.snapshot(self.status) if self.timing else {"eta_seconds": None, "eta_phase": "estimating", "active": {}, "average_sample_seconds": None}
            active = copy.deepcopy(self.active)
            for gpu, item in active.items():
                item.update(timing["active"].get(gpu, {"eta_seconds": None, "eta_phase": "waiting"}))
            return {
                "id": self.id,
                "status": self.status,
                "batch_started_at": self.batch_started_at,
                "total": self.total,
                "save_images": self.save_images,
                "completed": self.completed,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "cancelled": self.cancelled,
                "progress": progress,
                "message": self.message,
                "active": active,
                "eta_seconds": timing["eta_seconds"],
                "eta_phase": timing["eta_phase"],
                "average_sample_seconds": timing["average_sample_seconds"],
                "results": copy.deepcopy(self.results[-100:]),
                "errors": copy.deepcopy(self.errors[-100:]),
            }


class JobManager:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.previews = PreviewStore(output_root)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._context = mp.get_context("spawn")
        self._device_locks: dict[str, Any] = {}
        self._state_store = JsonJobStateStore(output_root / ".jobs")
        self._historical_jobs: dict[str, dict[str, Any]] = {}
        for snapshot in self._state_store.load_all():
            if snapshot.get("status") not in TERMINAL_STATUSES:
                snapshot["status"] = "interrupted"
                snapshot["message"] = "服务重启时任务仍未结束，已标记为中断"
                snapshot["active"] = {}
                snapshot["eta_seconds"] = None
                snapshot["eta_phase"] = "stopped"
                self._state_store.save(snapshot)
            self._historical_jobs[str(snapshot["id"])] = snapshot

    def submit(
        self,
        *,
        benchmark_name: str,
        samples: list[SampleModel],
        selections: list[PipelineSelection],
        configs: dict[tuple[str, str], dict[str, Any]],
        save_images: bool = True,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S_%f%z")
        units_by_gpu: dict[str, list[dict[str, Any]]] = {}
        unit_count = 0

        for selection in selections:
            config = configs[(selection.pipeline, selection.config_name)]
            for sample_index, sample in enumerate(samples):
                sample_benchmark = sample.benchmark_name or benchmark_name
                gpu_id = selection.gpu_ids[sample_index % len(selection.gpu_ids)]
                sample_dir = f"{safe_path_part(sample.id, 'sample')}-{timestamp}"
                output_dir = (
                    self.output_root
                    / safe_path_part(selection.pipeline, "pipeline")
                    / safe_path_part(sample_benchmark, "users")
                    / sample_dir
                )
                unit_count += 1
                unit = {
                    "unit_id": f"{selection.pipeline}:{sample.id}:{unit_count}",
                    "pipeline": selection.pipeline,
                    "config_name": selection.config_name,
                    "config": config,
                    "sample": sample.model_dump(),
                    "benchmark": sample_benchmark,
                    "batch_started_at": timestamp,
                    "output_dir": str(output_dir),
                    "save_images": save_images,
                }
                units_by_gpu.setdefault(gpu_id, []).append(unit)

        pause_event = self._context.Event()
        cancel_event = self._context.Event()
        message_queue = self._context.Queue()
        record = JobRecord(
            id=job_id,
            status="queued",
            batch_started_at=timestamp,
            total=unit_count,
            save_images=save_images,
            pause_event=pause_event,
            cancel_event=cancel_event,
            message_queue=message_queue,
            timing=JobTiming(units_by_gpu),
        )
        with self._lock:
            self._jobs[job_id] = record

        for gpu_id, work_units in units_by_gpu.items():
            with self._lock:
                device_lock = self._device_locks.setdefault(
                    gpu_id, self._context.Lock()
                )
            process = self._context.Process(
                target=run_gpu_worker,
                args=(
                    job_id,
                    gpu_id,
                    work_units,
                    device_lock,
                    pause_event,
                    cancel_event,
                    message_queue,
                ),
                name=f"gen-{job_id}-gpu-{gpu_id}",
                daemon=True,
            )
            process.start()
            record.processes.append(process)

        with record.lock:
            record.status = "running"
            record.message = f"已启动 {len(record.processes)} 个 GPU 工作进程"
        self._persist(record, force=True)
        threading.Thread(
            target=self._monitor,
            args=(record,),
            name=f"monitor-{job_id}",
            daemon=True,
        ).start()
        return record

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_snapshot(self, job_id: str) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is not None:
            return job.snapshot()
        with self._lock:
            snapshot = self._historical_jobs.get(job_id)
            return copy.deepcopy(snapshot) if snapshot is not None else None

    def list_snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
            historical = copy.deepcopy(self._historical_jobs)
        live = [job.snapshot() for job in jobs]
        for snapshot in live:
            historical.pop(snapshot["id"], None)
        return sorted(
            [*live, *historical.values()],
            key=lambda item: str(item.get("batch_started_at", "")),
            reverse=True,
        )

    def save_preview(self, preview_id: str, image_index: int) -> dict[str, Any]:
        saved = self.previews.save(preview_id, image_index)
        job = self.get(saved["job_id"])
        if job is not None:
            with job.lock:
                for result in job.results:
                    if result.get("preview_id") == preview_id:
                        result["images"] = saved["images"]
                        result["saved_images"] = saved["saved_images"]
                        break
            self._persist(job, force=True)
        return saved

    def pause(self, job_id: str) -> JobRecord:
        job = self._require_active(job_id)
        with job.lock:
            if job.timing:
                job.timing.pause()
            job.pause_event.set()
            job.status = "paused"
            job.message = "暂停信号已发送，正在等待安全暂停点"
        self._persist(job, force=True)
        return job

    def resume(self, job_id: str) -> JobRecord:
        job = self._require_active(job_id)
        with job.lock:
            if job.timing:
                job.timing.resume()
            job.pause_event.clear()
            job.status = "running"
            job.message = "任务已继续"
        self._persist(job, force=True)
        return job

    def cancel(self, job_id: str) -> JobRecord:
        job = self._require_active(job_id)
        job.cancel_event.set()
        job.pause_event.clear()
        with job.lock:
            job.status = "cancelling"
            job.message = "取消信号已发送，已完成结果会保留"
        self._persist(job, force=True)
        return job

    def _persist(self, job: JobRecord, force: bool = False) -> None:
        now = time.monotonic()
        with job.lock:
            if not force and now - job.last_persisted_at < 0.5:
                return
            snapshot = job.snapshot()
            job.last_persisted_at = now
        self._state_store.save(snapshot)
        with self._lock:
            self._historical_jobs[job.id] = snapshot

    def _require_active(self, job_id: str) -> JobRecord:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status in TERMINAL_STATUSES:
            raise RuntimeError(f"任务已结束: {job.status}")
        return job

    def _monitor(self, job: JobRecord) -> None:
        done_workers: set[str] = set()
        expected_workers = len(job.processes)
        while len(done_workers) < expected_workers:
            try:
                message = job.message_queue.get(timeout=0.5)
            except queue.Empty:
                if not any(process.is_alive() for process in job.processes):
                    break
                continue
            message_type = message.get("type")
            gpu_id = str(message.get("gpu_id", "unknown"))
            with job.lock:
                if job.timing:
                    job.timing.handle(message)
                if message_type == "unit_started":
                    job.active[gpu_id] = {
                        "unit_id": message["unit_id"],
                        "pipeline": message["pipeline"],
                        "sample_id": message["sample_id"],
                        "step": 0,
                        "total_steps": message.get("total_steps", 0),
                        "progress": 0.0,
                    }
                    job.message = f"GPU {gpu_id} 正在生成 {message['sample_id']}"
                elif message_type == "worker_waiting":
                    job.active[gpu_id] = {"step": 0, "total_steps": 0, "progress": 0.0}
                    job.message = f"GPU {gpu_id} 正在等待可用设备"
                elif message_type == "denoise_progress":
                    active = job.active.setdefault(gpu_id, {})
                    active.update(
                        {
                            "unit_id": message["unit_id"],
                            "step": message["step"],
                            "total_steps": message["total"],
                            "progress": message["step"] / message["total"],
                        }
                    )
                elif message_type == "unit_completed":
                    job.completed += 1
                    job.active.pop(gpu_id, None)
                    preview_id = None
                    try:
                        if message.get("preview_context") is not None:
                            preview_id = self.previews.add(message["preview_images"], message["preview_context"])
                    except ValueError as exc:
                        job.failed += 1
                        job.errors.append({"unit_id": message["unit_id"], "error": str(exc)})
                    else:
                        job.succeeded += 1
                        job.results.append({
                            "unit_id": message["unit_id"],
                            "output_dir": message["output_dir"],
                            "pipeline": message.get("pipeline"),
                            "config_name": message.get("config_name"),
                            "benchmark": message.get("benchmark"),
                            "sample_id": message.get("sample_id"),
                            "images": message["images"],
                            "duration_seconds": message["duration_seconds"],
                            "request": message.get("request", {}),
                            "metadata": message.get("metadata", {}),
                            "preview_id": preview_id,
                            "preview_count": len(message.get("preview_images", [])),
                            "saved_images": {},
                        })
                elif message_type == "unit_failed":
                    job.completed += 1
                    job.failed += 1
                    job.active.pop(gpu_id, None)
                    job.errors.append(
                        {"unit_id": message["unit_id"], "error": message["error"]}
                    )
                elif message_type == "unit_cancelled":
                    job.completed += 1
                    job.cancelled += 1
                    job.active.pop(gpu_id, None)
                elif message_type == "worker_failed":
                    job.errors.append(
                        {"gpu_id": gpu_id, "error": message.get("error", "工作进程失败")}
                    )
                elif message_type == "worker_done":
                    done_workers.add(gpu_id)
                    job.active.pop(gpu_id, None)
            self._persist(job)

        for process in job.processes:
            process.join(timeout=1)
            if process.exitcode not in {0, None}:
                with job.lock:
                    job.errors.append(
                        {
                            "worker": process.name,
                            "error": f"工作进程异常退出，exit code={process.exitcode}",
                        }
                    )
        with job.lock:
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.cancelled += max(0, job.total - job.completed)
                job.completed = job.total
                job.message = "任务已取消，已完成结果已保留"
            elif job.failed or job.errors:
                missing = max(0, job.total - job.completed)
                if missing:
                    job.failed += missing
                    job.completed = job.total
                job.status = "completed_with_errors" if job.succeeded else "failed"
                job.message = (
                    f"任务结束，{job.failed} 个生成单元失败"
                    if job.failed
                    else "任务生成完成，但工作进程报告了错误"
                )
            else:
                job.status = "completed"
                job.completed = job.total
                job.message = "全部生成完成"
        self._persist(job, force=True)

    def shutdown(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.status not in TERMINAL_STATUSES:
                job.cancel_event.set()
                job.pause_event.clear()
        for job in jobs:
            for process in job.processes:
                process.join(timeout=2)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1)
        self.previews.close()

