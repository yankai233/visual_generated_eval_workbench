from __future__ import annotations

import json
import os
import selectors
import signal
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from src.pipelines.base import (
    BaseGenerationPipeline,
    GenerationCancelled,
    GenerationRequest,
    GenerationResult,
    PipelineControl,
    ProgressCallback,
)


BRIDGE_SCRIPT = Path(__file__).with_name("outpainting_bridge.py")


def _path_issue(config: Mapping[str, Any], key: str, *, directory: bool = False) -> str | None:
    value = config.get(key)
    if not value:
        return f"缺少配置 {key}"
    path = Path(str(value)).expanduser()
    valid = path.is_dir() if directory else path.is_file()
    return None if valid else f"{key} 不存在: {path}"


class ExternalOutpaintingPipeline(BaseGenerationPipeline):
    """Persistent JSON-line bridge into the isolated Outpainting environment."""

    pipeline_group = "Outpainting"
    supported_tasks = ("Outpainting",)
    max_reference_images = 1
    reference_limit_note = "1 张输入图，原图默认居中"
    supported_devices = ("cuda",)
    method = ""
    bridge_script = BRIDGE_SCRIPT
    bridge_stderr: Any = subprocess.STDOUT
    required_files: tuple[str, ...] = ()
    required_directories: tuple[str, ...] = ()

    def __init__(self, config: Mapping[str, Any], device: str) -> None:
        self.config = dict(config)
        self.device = device
        python = str(Path(str(config["external_python"])).expanduser())
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self._process = subprocess.Popen(
            [python, "-u", str(self.bridge_script), "--method", self.method],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.bridge_stderr,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        self._selector = selectors.DefaultSelector()
        assert self._process.stdout is not None
        self._selector.register(self._process.stdout, selectors.EVENT_READ)
        self._send({"type": "initialize", "config": self.config, "device": device})
        message = self._receive("ready", timeout=float(config.get("startup_timeout", 900)))
        if not message.get("ok"):
            self.close()
            raise RuntimeError(message.get("error", f"{self.display_name} 初始化失败"))

    @classmethod
    def inspect_config(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        issues: list[str] = []
        python = Path(str(config.get("external_python", ""))).expanduser()
        if not python.is_file():
            issues.append(f"external_python 不存在: {python}")
        for key in cls.required_files:
            if issue := _path_issue(config, key):
                issues.append(issue)
        for key in cls.required_directories:
            if issue := _path_issue(config, key, directory=True):
                issues.append(issue)
        if cls.method == "pqdiff" and not issues:
            checkpoint = Path(str(config["checkpoint"])).expanduser()
            if checkpoint.stat().st_size < 100 * 1024 * 1024:
                issues.append(f"PQDiff 权重不完整（仅 {checkpoint.stat().st_size / 1024 / 1024:.1f} MiB）: {checkpoint}")
        return {
            "available": not issues,
            "issues": issues,
            "configured_model_path": config.get("checkpoint") or config.get("global_checkpoint"),
            "resolved_model_path": config.get("checkpoint") or config.get("global_checkpoint"),
        }

    @classmethod
    def from_pretrained(cls, config: Mapping[str, Any], device: str) -> "ExternalOutpaintingPipeline":
        status = cls.inspect_config(config)
        if status["issues"]:
            raise RuntimeError("；".join(status["issues"]))
        if not device.startswith("cuda"):
            raise ValueError(f"{cls.display_name} 仅支持 CUDA")
        return cls(config, device)

    def _send(self, value: dict[str, Any]) -> None:
        if self._process.poll() is not None or self._process.stdin is None:
            raise RuntimeError(f"{self.display_name} 外部进程已经退出")
        self._process.stdin.write(json.dumps(value, ensure_ascii=False) + "\n")
        self._process.stdin.flush()

    def _receive(
        self,
        expected: str,
        *,
        timeout: float,
        control: PipelineControl | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        recent_logs: list[str] = []
        while time.monotonic() < deadline:
            if control is not None:
                if control.paused and self._process.poll() is None:
                    self._process.send_signal(signal.SIGSTOP)
                    paused_at = time.monotonic()
                    try:
                        control.checkpoint()
                    except GenerationCancelled:
                        self._process.send_signal(signal.SIGCONT)
                        self.close()
                        raise
                    deadline += time.monotonic() - paused_at
                    self._process.send_signal(signal.SIGCONT)
                elif control.cancelled:
                    self.close()
                    raise GenerationCancelled("generation cancelled")
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"{self.display_name} 外部进程异常退出 ({self._process.returncode})"
                    + ("：" + " | ".join(recent_logs[-5:]) if recent_logs else "")
                )
            events = self._selector.select(timeout=0.2)
            if not events:
                continue
            line = self._process.stdout.readline().rstrip("\n")  # type: ignore[union-attr]
            if not line.startswith("WORKBENCH_JSON "):
                if line:
                    recent_logs.append(line[-500:])
                continue
            try:
                message = json.loads(line.removeprefix("WORKBENCH_JSON "))
            except json.JSONDecodeError:
                continue
            if message.get("type") == "progress" and progress_callback is not None:
                progress_callback(int(message.get("step", 0)), int(message.get("total", 1)))
                continue
            if message.get("type") == "error":
                raise RuntimeError(str(message.get("error", f"{self.display_name} 外部进程执行失败")))
            if message.get("type") == expected:
                return message
        self.close()
        raise TimeoutError(f"{self.display_name} 等待 {expected} 超时")

    def __call__(
        self,
        request: GenerationRequest,
        *,
        progress_callback: ProgressCallback | None = None,
        control: PipelineControl | None = None,
    ) -> GenerationResult:
        if request.task_type != "Outpainting" or len(request.input_images) != 1:
            raise ValueError("扩图任务需要且只能输入一张图像")
        temporary = Path(tempfile.mkdtemp(prefix=f"workbench-{self.method}-"))
        output = temporary / "generated.png"
        try:
            self._send({
                "type": "generate",
                "request": {
                    "id": request.id,
                    "input_image": str(request.input_images[0]),
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "seed": request.seed if request.seed is not None else 42,
                    "width": request.width,
                    "height": request.height,
                    "num_inference_steps": request.num_inference_steps,
                    "extra": request.extra,
                    "output": str(output),
                },
            })
            message = self._receive(
                "result",
                timeout=float(self.config.get("generation_timeout", 3600)),
                control=control,
                progress_callback=progress_callback,
            )
            with Image.open(message["output"]) as opened:
                image = opened.convert("RGB").copy()
            return GenerationResult(images=[image], metadata=message.get("metadata", {}))
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        try:
            self._selector.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
