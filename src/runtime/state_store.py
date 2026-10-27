from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


class JsonJobStateStore:
    """Small atomic JSON store for job history and restart diagnostics."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def save(self, snapshot: dict[str, Any]) -> None:
        job_id = str(snapshot["id"])
        destination = self.directory / f"{job_id}.json"
        with self._lock:
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{job_id}-", suffix=".json", dir=self.directory
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(snapshot, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, destination)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)

    def load_all(self) -> list[dict[str, Any]]:
        snapshots: list[dict[str, Any]] = []
        with self._lock:
            paths = list(self.directory.glob("*.json"))
        for path in paths:
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(loaded, dict) and loaded.get("id"):
                snapshots.append(loaded)
        return sorted(
            snapshots,
            key=lambda item: str(item.get("batch_started_at", "")),
            reverse=True,
        )

