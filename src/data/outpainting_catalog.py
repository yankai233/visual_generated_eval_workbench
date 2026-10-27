from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OutpaintingRun:
    id: str
    label: str
    directory: Path
    samples: dict[str, Path]


def _sample_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value.casefold())


class OutpaintingResultCatalog:
    """Read-only index over external outpainting evaluation outputs."""

    def __init__(self, root: Path, cache_seconds: float = 3.0) -> None:
        self.root = root.resolve()
        self.cache_seconds = cache_seconds
        self._lock = threading.Lock()
        self._indexed_at = 0.0
        self._runs: dict[str, OutpaintingRun] = {}

    @staticmethod
    def _run_label(relative: Path) -> str:
        parts = list(relative.parts)
        aliases = {"blouout": "BlueOut", "proout": "ProOut", "pqdiff": "PQDiff"}
        if parts:
            parts[0] = aliases.get(parts[0].casefold(), parts[0])
        return " · ".join(parts)

    def _scan(self) -> dict[str, OutpaintingRun]:
        grouped: dict[Path, dict[str, Path]] = {}
        if self.root.is_dir():
            for info_path in self.root.glob("**/info/item_info.json"):
                sample_dir = info_path.parent.parent
                try:
                    sample_dir.relative_to(self.root)
                except ValueError:
                    continue
                if not any(sample_dir.glob("generated*.png")):
                    continue
                grouped.setdefault(sample_dir.parent, {})[sample_dir.name] = sample_dir

        runs: dict[str, OutpaintingRun] = {}
        for directory, samples in grouped.items():
            relative = directory.relative_to(self.root)
            run_id = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:16]
            runs[run_id] = OutpaintingRun(
                id=run_id,
                label=self._run_label(relative),
                directory=directory,
                samples=samples,
            )
        return dict(sorted(runs.items(), key=lambda item: item[1].label.casefold()))

    def _index(self, force: bool = False) -> dict[str, OutpaintingRun]:
        now = time.monotonic()
        with self._lock:
            if force or now - self._indexed_at >= self.cache_seconds:
                self._runs = self._scan()
                self._indexed_at = now
            return self._runs

    def catalog(self, force: bool = False) -> dict[str, Any]:
        return {
            "runs": [
                {"id": run.id, "label": run.label, "sample_count": len(run.samples)}
                for run in self._index(force).values()
            ]
        }

    @staticmethod
    def _read_sample(run: OutpaintingRun, sample_id: str) -> dict[str, Any] | None:
        directory = run.samples.get(sample_id)
        if directory is None:
            return None
        try:
            payload = json.loads((directory / "info" / "item_info.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        pipeline = payload.get("pipeline") if isinstance(payload.get("pipeline"), dict) else {}
        inference = pipeline.get("inference") if isinstance(pipeline.get("inference"), dict) else {}
        box = source.get("evaluation_known_box") or pipeline.get("input_box")
        generated = sorted(directory.glob("generated*.png"), key=lambda path: path.name)
        return {
            "prompt": str(source.get("prompt") or pipeline.get("resolved_prompt") or ""),
            "width": int(source.get("width") or inference.get("canvas_width") or 1),
            "height": int(source.get("height") or inference.get("canvas_height") or 1),
            "input_box": box if isinstance(box, list) and len(box) == 4 else None,
            "has_input": (directory / "input_image.png").is_file(),
            "has_target": (directory / "target_image.png").is_file(),
            "image_count": len(generated),
        }

    def view(self, run_ids: list[str], query: str = "", page: int = 1, page_size: int = 12) -> dict[str, Any]:
        runs = self._index()
        selected = [runs[run_id] for run_id in run_ids if run_id in runs]
        if not selected:
            return {"rows": [], "total": 0, "page": 1, "page_size": page_size, "runs": []}
        sample_ids = sorted(set().union(*(run.samples for run in selected)), key=_sample_key)
        query = query.strip().casefold()
        records: list[tuple[str, dict[str, Any], list[dict[str, Any] | None]]] = []
        for sample_id in sample_ids:
            cells = [self._read_sample(run, sample_id) for run in selected]
            representative = next((cell for cell in cells if cell), None)
            if representative is None:
                continue
            if query and query not in sample_id.casefold() and query not in representative["prompt"].casefold():
                continue
            records.append((sample_id, representative, cells))

        total = len(records)
        max_page = max(1, (total + page_size - 1) // page_size)
        page = min(max(1, page), max_page)
        start = (page - 1) * page_size
        rows = []
        for sample_id, representative, cells in records[start : start + page_size]:
            input_index = next((index for index, cell in enumerate(cells) if cell and cell["has_input"]), None)
            target_index = next((index for index, cell in enumerate(cells) if cell and cell["has_target"]), None)
            rows.append({
                "sample_id": sample_id,
                "prompt": representative["prompt"],
                "width": representative["width"],
                "height": representative["height"],
                "input_box": representative["input_box"],
                "input_url": (
                    f"/api/outpainting-comparison/image/{selected[input_index].id}/{sample_id}/input/0"
                    if input_index is not None else None
                ),
                "target_url": (
                    f"/api/outpainting-comparison/image/{selected[target_index].id}/{sample_id}/target/0"
                    if target_index is not None else None
                ),
                "cells": [
                    [] if cell is None else [
                        f"/api/outpainting-comparison/image/{run.id}/{sample_id}/generated/{index}"
                        for index in range(cell["image_count"])
                    ]
                    for run, cell in zip(selected, cells)
                ],
            })
        return {
            "rows": rows,
            "total": total,
            "page": page,
            "page_size": page_size,
            "runs": [{"id": run.id, "label": run.label} for run in selected],
        }

    def image(self, run_id: str, sample_id: str, kind: str, index: int) -> Path | None:
        run = self._index().get(run_id)
        if run is None or sample_id not in run.samples or index < 0:
            return None
        directory = run.samples[sample_id]
        if kind == "input" and index == 0:
            path = directory / "input_image.png"
        elif kind == "target" and index == 0:
            path = directory / "target_image.png"
        elif kind == "generated":
            images = sorted(directory.glob("generated*.png"), key=lambda item: item.name)
            path = images[index] if index < len(images) else directory / "__missing__"
        else:
            return None
        try:
            path.resolve().relative_to(self.root)
        except ValueError:
            return None
        return path if path.is_file() else None
