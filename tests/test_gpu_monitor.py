from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.utils.gpu import GPUUsageMonitor, query_gpu_usage
from src.web.server import create_app


def test_gpu_telemetry_is_numeric_and_uses_reported_memory(monkeypatch):
    def run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        assert "memory.used" in command[1]
        assert kwargs["timeout"] == 5
        assert "shell" not in kwargs
        return SimpleNamespace(stdout='1, NVIDIA A100, 570.86.10, 81920, 61440, 19456, 90, 58, 250.5, 300\n0, NVIDIA A100, 570.86.10, 81920, 40960, 39936, 42, 49, 120.25, 300\n')
    monkeypatch.setattr("src.utils.gpu.subprocess.run", run)
    data = query_gpu_usage()
    assert data["available"] and data["error"] is None
    assert [gpu["id"] for gpu in data["gpus"]] == ["0", "1"]
    gpu = data["gpus"][0]
    assert gpu["memory_used_mb"] == 40960
    assert gpu["memory_free_mb"] == 39936  # do not infer 'used' from total - free
    assert gpu["memory_percent"] == 50
    assert gpu["utilization_percent"] == 42
    assert gpu["temperature_c"] == 49
    assert gpu["power_w"] == 120.25


def test_gpu_unsupported_values_are_unknown_not_zero(monkeypatch):
    monkeypatch.setattr("src.utils.gpu.subprocess.run", lambda *args, **kwargs: SimpleNamespace(stdout="0, GPU, driver, 0, N/A, nan, [Not Supported], -1, inf, N/A\n"))
    gpu = query_gpu_usage()["gpus"][0]
    for field in ("memory_percent", "memory_used_mb", "memory_free_mb", "utilization_percent", "temperature_c", "power_w", "power_limit_w"):
        assert gpu[field] is None


@pytest.mark.parametrize("exception", [FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 5), subprocess.CalledProcessError(1, "nvidia-smi")])
def test_gpu_query_errors_are_reported(monkeypatch, exception):
    def fail(*args, **kwargs):
        raise exception
    monkeypatch.setattr("src.utils.gpu.subprocess.run", fail)
    result = query_gpu_usage()
    assert not result["available"] and result["gpus"] == []
    assert result["error"]


def test_gpu_malformed_rows_are_not_silently_omitted(monkeypatch):
    monkeypatch.setattr("src.utils.gpu.subprocess.run", lambda *args, **kwargs: SimpleNamespace(stdout="invalid,row\n"))
    assert query_gpu_usage()["error"] == "GPU 状态解析失败"


def test_monitor_cache_expires_and_returns_isolated_snapshots(monkeypatch):
    clock = [0.0]
    calls = []
    def query():
        calls.append(True)
        return {"available": True, "error": None, "gpus": [{"memory_used_mb": len(calls)}]}
    monkeypatch.setattr("src.utils.gpu.query_gpu_usage", query)
    monkeypatch.setattr("src.utils.gpu.time.monotonic", lambda: clock[0])
    monitor = GPUUsageMonitor(cache_seconds=1)
    first = monitor.snapshot()
    first["gpus"][0]["memory_used_mb"] = -1
    assert monitor.snapshot()["gpus"][0]["memory_used_mb"] == 1
    assert len(calls) == 1
    clock[0] = 1.1
    assert monitor.snapshot()["gpus"][0]["memory_used_mb"] == 2


def test_gpu_monitor_api_is_uncacheable_and_read_only(monkeypatch, tmp_path):
    monkeypatch.setattr("src.web.server.GEN_OUTPUT_ROOT", tmp_path / "gen")
    monkeypatch.setattr("src.utils.gpu.query_gpu_usage", lambda: {"available": True, "error": None, "gpus": [{"id": "0", "memory_used_mb": 1234}]})
    with TestClient(create_app()) as client:
        response = client.get("/api/system/gpus")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["gpus"][0]["memory_used_mb"] == 1234
        assert response.json()["updated_at"]
        assert client.post("/api/system/gpus").status_code == 405
