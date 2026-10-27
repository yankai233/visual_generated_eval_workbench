from __future__ import annotations

import subprocess
import copy
import csv
import io
import json
import math
import sys
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any


def _gpu_number(value: str) -> float | None:
    try:
        number = float(value.strip())
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def query_gpu_usage() -> dict[str, Any]:
    """Read physical GPU telemetry without loading torch or initializing CUDA."""
    fields = "index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit"
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5,
        )
    except FileNotFoundError:
        return {"available": False, "gpus": [], "error": "未找到 nvidia-smi"}
    except subprocess.TimeoutExpired:
        return {"available": False, "gpus": [], "error": "GPU 状态读取超时"}
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "gpus": [], "error": "无法读取 GPU 状态，请检查 NVIDIA 驱动"}
    gpus = []
    seen = set()
    for row in csv.reader(io.StringIO(completed.stdout), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 10 or not row[0].strip().isdigit() or row[0].strip() in seen:
            return {"available": False, "gpus": [], "error": "GPU 状态解析失败"}
        gpu_id, name, driver = (part.strip() for part in row[:3])
        total, used, free, utilization, temperature, power, power_limit = (_gpu_number(part) for part in row[3:])
        percentage = min(100.0, used / total * 100) if used is not None and total is not None and total > 0 else None
        gpus.append({
            "id": gpu_id, "name": name, "driver_version": driver,
            "memory_total_mb": total, "memory_used_mb": used, "memory_free_mb": free,
            "memory_percent": percentage,
            "utilization_percent": min(100.0, utilization) if utilization is not None else None,
            "temperature_c": temperature, "power_w": power, "power_limit_w": power_limit,
        })
        seen.add(gpu_id)
    gpus.sort(key=lambda gpu: int(gpu["id"]))
    return {"available": bool(gpus), "gpus": gpus, "error": None if gpus else "未检测到 NVIDIA GPU"}


class GPUUsageMonitor:
    """Share short-lived snapshots across viewers; errors are never cached long-term."""
    def __init__(self, cache_seconds: float = 1.0):
        self.cache_seconds = cache_seconds
        self._lock = threading.Lock()
        self._snapshot: dict | None = None
        self._checked_at = 0.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            if self._snapshot is None or now - self._checked_at >= self.cache_seconds:
                self._snapshot = {**query_gpu_usage(), "updated_at": datetime.now(timezone.utc).isoformat()}
                self._checked_at = time.monotonic()
            return copy.deepcopy(self._snapshot)


def list_gpus() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=5
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    results: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=3)]
        if len(parts) == 4:
            results.append(
                {
                    "id": parts[0],
                    "name": parts[1],
                    "memory_total_mb": parts[2],
                    "memory_free_mb": parts[3],
                }
            )
    return results


@lru_cache(maxsize=1)
def torch_cuda_status() -> dict[str, Any]:
    """Probe torch in a child process so the web parent never initializes CUDA."""
    script = """
import json
try:
    import torch
    result = {
        'available': bool(torch.cuda.is_available()),
        'torch_version': str(torch.__version__),
        'cuda_build': str(torch.version.cuda),
        'issue': None,
    }
    if not result['available']:
        result['issue'] = 'PyTorch 无法使用 CUDA；请检查 PyTorch/CUDA 与 NVIDIA 驱动版本'
except Exception as exc:
    result = {'available': False, 'torch_version': None, 'cuda_build': None, 'issue': str(exc)}
print(json.dumps(result, ensure_ascii=False))
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        line = completed.stdout.strip().splitlines()[-1]
        result = json.loads(line)
        if completed.returncode != 0:
            result["available"] = False
            result["issue"] = completed.stderr.strip() or result.get("issue")
        return result
    except Exception as exc:
        return {
            "available": False,
            "torch_version": None,
            "cuda_build": None,
            "issue": f"CUDA 检测失败: {exc}",
        }

