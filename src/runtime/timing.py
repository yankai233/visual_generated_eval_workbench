from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Callable


def finite_seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


class StepTiming:
    """Use recent step intervals, excluding first-step setup and pause time."""

    def __init__(self) -> None:
        self.points: deque[tuple[int, float]] = deque(maxlen=7)
        self.total = 0

    def update(self, step: int, total: int, active_time: float) -> float | None:
        if total != self.total or self.points and step <= self.points[-1][0]:
            self.points.clear()
        self.total = total
        self.points.append((step, active_time))
        if step >= total:
            return 0.0
        if len(self.points) < 2:
            return None
        first_step, first_time = self.points[0]
        if step <= first_step:
            return None
        return max(0.0, (active_time - first_time) / (step - first_step) * (total - step))


class JobTiming:
    """Per-device forecasts; a parallel job finishes when its slowest queue does."""

    def __init__(self, units_by_gpu: dict[str, list[dict]], clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.paused_at: float | None = None
        self.paused_seconds = 0.0
        self.completed_generation_seconds = 0.0
        self.completed_sample_count = 0
        self.devices = {}
        for gpu, units in units_by_gpu.items():
            plans = {}
            for unit in units:
                sample = unit["sample"]
                plans[unit["unit_id"]] = {
                    "profile": (unit["pipeline"], unit["config_name"]),
                    "weight": max(1, sample.get("width", 1024) * sample.get("height", 1024) * sample.get("num_inference_steps", 28)),
                }
            self.devices[str(gpu)] = {"plans": plans, "current": None, "rates": {}, "tails": {}, "phase": "waiting"}

    def now(self) -> float:
        now = self.clock()
        return (self.paused_at if self.paused_at is not None else now) - self.paused_seconds

    def pause(self) -> None:
        if self.paused_at is None:
            self.paused_at = self.clock()

    def resume(self) -> None:
        if self.paused_at is not None:
            self.paused_seconds += max(0.0, self.clock() - self.paused_at)
            self.paused_at = None

    @staticmethod
    def smooth(mapping: dict, key: Any, value: float) -> None:
        mapping[key] = 0.3 * value + 0.7 * mapping[key] if key in mapping else value

    def handle(self, message: dict) -> None:
        device = self.devices.get(str(message.get("gpu_id")))
        if device is None:
            return
        kind = message.get("type")
        unit_id = message.get("unit_id")
        if kind == "unit_started":
            device["current"] = {"id": unit_id, "phase": "loading", "updated": self.now(), "generation_elapsed": 0.0, "step_eta": None, "step": 0, "total": 0}
            device["phase"] = "loading"
        elif kind == "unit_generation_started" and device["current"] and device["current"]["id"] == unit_id:
            device["current"].update(phase="estimating", updated=self.now())
            device["phase"] = "estimating"
        elif kind == "denoise_progress" and device["current"] and device["current"]["id"] == unit_id:
            device["current"].update(
                phase="denoising", updated=self.now(),
                generation_elapsed=finite_seconds(message.get("generation_elapsed_seconds")) or 0.0,
                step_eta=finite_seconds(message.get("eta_seconds")), step=message["step"], total=message["total"],
            )
            device["phase"] = "denoising"
        elif kind in {"unit_completed", "unit_failed", "unit_cancelled"}:
            plan = device["plans"].pop(unit_id, None)
            current = device["current"]
            duration = finite_seconds(message.get("generation_duration_seconds"))
            if kind == "unit_completed" and plan and duration is not None:
                self.completed_generation_seconds += duration
                self.completed_sample_count += 1
                self.smooth(device["rates"], plan["profile"], duration / plan["weight"])
                if current and current["id"] == unit_id and current["step"] >= current["total"] > 0:
                    self.smooth(device["tails"], plan["profile"], max(0.0, duration - current["generation_elapsed"]))
            device["current"] = None
            device["phase"] = "estimating" if device["plans"] else "done"
        elif kind == "worker_waiting":
            device["phase"] = "waiting"
        elif kind in {"worker_failed", "worker_done"} and device["plans"]:
            device["phase"] = "stopped"

    def snapshot(self, status: str) -> dict:
        average_sample_seconds = (
            round(self.completed_generation_seconds / self.completed_sample_count, 2)
            if self.completed_sample_count
            else None
        )
        if status in {"completed", "completed_with_errors"}:
            return {"eta_seconds": 0.0, "eta_phase": "done", "active": {}, "average_sample_seconds": average_sample_seconds}
        if status in {"cancelled", "failed", "interrupted", "cancelling"}:
            return {"eta_seconds": None, "eta_phase": "stopped", "active": {}, "average_sample_seconds": average_sample_seconds}
        now = self.now()
        forecasts = []
        active = {}
        phases = []
        for gpu, device in self.devices.items():
            plans = device["plans"]
            if not plans:
                continue
            current = device["current"]
            rates = dict(device["rates"])
            queue_eta: float | None = 0.0
            phase = device["phase"]
            current_plan = plans.get(current["id"]) if current else None
            if current and current_plan:
                delta = max(0.0, now - current["updated"])
                elapsed = current["generation_elapsed"] + delta
                step_eta = current["step_eta"]
                denoise_eta = max(0.0, step_eta - delta) if step_eta is not None else None
                tail = device["tails"].get(current_plan["profile"], 0.0)
                if current["phase"] == "loading":
                    queue_eta = None
                elif denoise_eta is not None:
                    queue_eta = max(0.0, step_eta + tail - delta)
                    inferred = (current["generation_elapsed"] + step_eta + tail) / current_plan["weight"]
                    rates.setdefault(current_plan["profile"], inferred)
                elif current_plan["profile"] in rates:
                    queue_eta = max(0.0, rates[current_plan["profile"]] * current_plan["weight"] - elapsed)
                else:
                    queue_eta = None
                active[gpu] = {"eta_seconds": denoise_eta, "eta_phase": current["phase"]}
            elif phase in {"waiting", "stopped"}:
                queue_eta = None
            for pending_id, plan in plans.items():
                if current and pending_id == current["id"]:
                    continue
                rate = rates.get(plan["profile"])
                if queue_eta is None or rate is None:
                    queue_eta = None
                    break
                queue_eta += rate * plan["weight"]
            forecasts.append(queue_eta)
            phases.append(phase)
        eta = max(forecasts) if forecasts and all(value is not None for value in forecasts) else None
        phase = "running" if eta is not None else "waiting" if "waiting" in phases else "loading" if "loading" in phases else "estimating"
        return {
            "eta_seconds": round(eta, 2) if eta is not None else None,
            "eta_phase": phase,
            "active": active,
            "average_sample_seconds": average_sample_seconds,
        }
