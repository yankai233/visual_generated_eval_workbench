"""Bounded, expiring RAM previews. No image is written until explicitly saved."""
from __future__ import annotations

import copy
import os
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import yaml

from src.data.sample_identity import atomic_json, prepare_sample
from src.data.schemas import SampleModel


class PreviewUnavailable(LookupError):
    pass


class PreviewStore:
    def __init__(self, output_root: Path, *, max_bytes: int = 512 * 1024 * 1024,
                 ttl_seconds: float = 3600, clock: Callable[[], float] = time.monotonic):
        self.output_root = output_root.resolve()
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _remove(self, key: str) -> None:
        entry = self._entries.pop(key)
        self._bytes -= entry["size"]

    def _purge(self) -> None:
        for key, entry in list(self._entries.items()):
            if self.clock() - entry["created"] >= self.ttl_seconds:
                self._remove(key)

    def _sweep(self) -> None:
        while not self._stop.wait(60):
            with self._lock:
                self._purge()

    def add(self, images: list[bytes], context: dict[str, Any]) -> str:
        size = sum(len(image) for image in images)
        if not images or size > self.max_bytes:
            raise ValueError("临时预览为空或超过内存容量上限，请开启保存图像")
        with self._lock:
            if self._stop.is_set():
                raise ValueError("预览服务已关闭")
            self._purge()
            while self._bytes + size > self.max_bytes:
                self._remove(next(iter(self._entries)))
            key = uuid.uuid4().hex
            self._entries[key] = {"images": images, "context": copy.deepcopy(context),
                                  "saved_images": {}, "created": self.clock(), "size": size}
            self._bytes += size
            if self._thread is None:
                self._thread = threading.Thread(target=self._sweep, name="preview-expiry", daemon=True)
                self._thread.start()
            return key

    def _get(self, key: str, index: int) -> dict[str, Any]:
        self._purge()
        entry = self._entries.get(key)
        if entry is None:
            raise PreviewUnavailable("临时图像已过期、被容量回收或服务已重启，请重新生成")
        if index < 0 or index >= len(entry["images"]):
            raise PreviewUnavailable("图像不存在")
        return entry

    @staticmethod
    def _name(entry: dict, index: int) -> str:
        return "generated.png" if len(entry["images"]) == 1 else f"generated_{index}.png"

    def image(self, key: str, index: int) -> tuple[bytes, str]:
        with self._lock:
            entry = self._get(key, index)
            return entry["images"][index], self._name(entry, index)

    def save(self, key: str, index: int) -> dict[str, Any]:
        """Save only this image, with enough metadata for the comparison catalog."""
        with self._lock:
            entry = self._get(key, index)
            context = entry["context"]
            directory = Path(context["output_dir"]).resolve()
            if not directory.is_relative_to(self.output_root) or len(directory.relative_to(self.output_root).parts) != 3:
                raise ValueError("非法的生成结果目录")
            destination = directory / self._name(entry, index)
            if destination.is_symlink():
                raise ValueError("不能保存到符号链接文件")
            if index not in entry["saved_images"]:
                # Never replace a pre-existing image, even if an external process
                # has written into this timestamped directory in the meantime.
                if destination.exists():
                    raise ValueError("保存位置已有同名图像，未覆盖")
                directory.mkdir(parents=True, exist_ok=True)
                request = dict(context["request"])
                comparison = request.get("comparison", {})
                source = comparison.get("source") or ("users" if context["result"].get("benchmark") == "users" else f"benchmark:{context['result'].get('benchmark')}")
                original = SampleModel.model_validate({**request, "id": comparison.get("original_sample_id", request["id"])})
                archived = prepare_sample(original, source, self.output_root)
                request.update(input_images=archived.input_images, comparison=archived.comparison)
                atomic_json(directory / "request.json", request)
                config_path = directory / "config.yaml"
                if config_path.is_symlink() or (directory / "result.json").is_symlink():
                    raise ValueError("不能保存到符号链接文件")
                if not config_path.exists():
                    with config_path.open("x", encoding="utf-8") as handle:
                        yaml.safe_dump(context["config"], handle, allow_unicode=True, sort_keys=False)
                fd, temporary = tempfile.mkstemp(prefix=".preview-", dir=directory)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(entry["images"][index])
                    os.link(temporary, destination)  # exclusive, atomic publication
                finally:
                    Path(temporary).unlink(missing_ok=True)
                entry["saved_images"][index] = str(destination)
            # Retry also repairs metadata if the image was saved but JSON writing
            # failed (for example due to a transient disk-full condition).
            paths = [path for _, path in sorted(entry["saved_images"].items())]
            result = {**context["result"], "images": paths}
            atomic_json(directory / "result.json", result)
            return {"job_id": result["job_id"], "unit_id": context["unit_id"],
                    "image_index": index, "path": str(destination), "output_dir": str(directory),
                    "saved_images": {str(i): path for i, path in entry["saved_images"].items()},
                    "images": paths}

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        with self._lock:
            self._entries.clear()
            self._bytes = 0
