from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote

from src.data.sample_identity import atomic_json, input_identity, normalized_text


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


class ResultCatalog:
    def __init__(self, root: Path, benchmark_root: Path) -> None:
        self.root = root.resolve()
        self.output_root = self.root.parent
        self.benchmark_root = benchmark_root.resolve()
        self.groups_path = self.root / ".comparison" / "groups.json"
        self._lock = threading.RLock()

    def _url(self, path: Path) -> str | None:
        resolved = path.resolve()
        if not resolved.is_file():
            return None
        for root, prefix in ((self.output_root, "/files/output/"), (self.benchmark_root, "/files/benchmarks/")):
            try:
                return prefix + quote(resolved.relative_to(root).as_posix(), safe="/")
            except ValueError:
                continue
        return None

    def records(self, source: str | None = None, derive_benchmark_identity: bool = True) -> list[dict[str, Any]]:
        with self._lock:
            groups = read_object(self.groups_path)
        records = []
        for path in sorted(self.root.glob("*/*/*/result.json")):
            directory = path.parent
            relative = directory.relative_to(self.root)
            if relative.parts[0].startswith("."):
                continue
            result = read_object(path)
            if result.get("status") != "completed":
                continue
            request = read_object(directory / "request.json")
            comparison = request.get("comparison") or result.get("comparison") or {}
            if not isinstance(comparison, dict):
                comparison = {}
            pipeline, benchmark, _ = relative.parts
            record_id = relative.as_posix()
            record_source = comparison.get("source") or ("users" if benchmark == "users" else f"benchmark:{benchmark}")
            if source and source != record_source:
                continue
            image_paths = []
            for raw in result.get("images", []):
                candidate = Path(str(raw))
                if not candidate.is_file():
                    candidate = directory / candidate.name
                try:
                    candidate.resolve().relative_to(directory.resolve())
                except ValueError:
                    continue
                if candidate.is_file():
                    image_paths.append(candidate)
            if not image_paths:
                image_paths = sorted(directory.glob("generated*.png"))
            image_urls = [url for image in image_paths if (url := self._url(image))]
            if not image_urls:
                continue
            sample_id = str(result.get("sample_id") or request.get("id") or directory.name)
            fingerprint = comparison.get("input_fingerprint")
            if not fingerprint and (record_source == "users" or derive_benchmark_identity) and "task_type" in request and "prompt" in request:
                try:
                    fingerprint = input_identity(request)["fingerprint"]
                except (OSError, ValueError, TypeError):
                    pass
            match_kind = "automatic"
            if record_source != "users":
                case_id = f"{record_source}:{sample_id}"
                label = sample_id
            else:
                if fingerprint:
                    case_id = f"user:{fingerprint}"
                    label = f"自定义 · {fingerprint[:8]}"
                else:
                    case_id = f"legacy:{record_id}"
                    label = f"待分组 · {sample_id}"
                    match_kind = "unresolved"
            automatic_case_id = case_id
            group = groups.get(record_id)
            if record_source == "users" and isinstance(group, str) and group:
                label = group
                case_id = "manual:" + hashlib.sha256(group.encode()).hexdigest()
                match_kind = "manual"
            references = [Path(str(item)) for item in request.get("input_images", [])]
            manifest_name = comparison.get("input_manifest")
            if manifest_name:
                manifest_path = (self.root / str(manifest_name)).resolve()
                if manifest_path.is_relative_to(self.root):
                    manifest = read_object(manifest_path)
                    references = [self.root / item for item in manifest.get("input_images", [])]
            reference_urls = [url for item in references if (url := self._url(item))]
            raw_generation_duration = result.get("generation_duration_seconds")
            try:
                generation_duration = float(raw_generation_duration)
                if not math.isfinite(generation_duration) or generation_duration < 0:
                    generation_duration = None
            except (TypeError, ValueError):
                generation_duration = None
            records.append({
                "result_id": record_id, "pipeline": pipeline, "source": record_source,
                "config_name": str(result.get("config_name", "default")),
                "run_id": str(result.get("job_id") or directory.name),
                "batch_started_at": str(result.get("batch_started_at") or directory.name.rsplit("-", 1)[-1]),
                "case_id": case_id, "automatic_case_id": automatic_case_id,
                "sample_id": sample_id, "label": label, "match_kind": match_kind,
                "input_fingerprint": fingerprint, "task_type": request.get("task_type", "T2I"),
                "prompt": request.get("prompt", ""), "negative_prompt": request.get("negative_prompt", ""),
                "reference_urls": reference_urls, "image_urls": image_urls,
                "seed": request.get("seed"), "width": request.get("width"), "height": request.get("height"),
                "num_inference_steps": request.get("num_inference_steps"),
                "generation_duration_seconds": generation_duration,
            })
        return sorted(records, key=lambda record: (record["batch_started_at"], record["result_id"]), reverse=True)

    @staticmethod
    def _timing_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Average the newest timed result for each sample and method."""
        latest: dict[tuple[str, str, str], dict[str, Any]] = {}
        for record in records:
            key = (
                record["source"],
                record["pipeline"],
                record["case_id"],
            )
            latest.setdefault(key, record)

        groups: dict[tuple[str, str], list[float]] = {}
        for record in latest.values():
            if record["source"] == "users":
                continue
            duration = record.get("generation_duration_seconds")
            if duration is None:
                continue
            key = (record["source"], record["pipeline"])
            groups.setdefault(key, []).append(float(duration))

        rows = []
        for (source, pipeline), durations in groups.items():
            rows.append(
                {
                    "source": source,
                    "benchmark": "自定义" if source == "users" else source.removeprefix("benchmark:"),
                    "pipeline": pipeline,
                    "sample_count": len(durations),
                    "average_generation_seconds": round(sum(durations) / len(durations), 4),
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                ResultCatalog._natural_key(row["benchmark"]),
                row["pipeline"].casefold(),
            ),
        )

    def catalog(self) -> dict[str, Any]:
        records = self.records(derive_benchmark_identity=False)
        sources = []
        for source in sorted({record["source"] for record in records}):
            subset = [record for record in records if record["source"] == source]
            methods = []
            for pipeline in sorted({record["pipeline"] for record in subset}):
                items = [record for record in subset if record["pipeline"] == pipeline]
                runs = {}
                for item in items:
                    runs.setdefault(item["run_id"], {
                        "id": item["run_id"], "batch_started_at": item["batch_started_at"],
                        "config_name": item["config_name"],
                    })
                methods.append({"pipeline": pipeline, "configs": sorted({item["config_name"] for item in items}), "runs": list(runs.values())})
            sources.append({
                "id": source, "label": "自定义" if source == "users" else source.removeprefix("benchmark:"),
                "sample_count": len({record["case_id"] for record in subset}), "methods": methods,
            })
        return {
            "sources": sources,
            "record_count": len(records),
            "timing_summary": self._timing_summary(records),
        }

    def samples(self, source: str) -> list[dict[str, Any]]:
        samples = {}
        for record in self.records(source):
            samples.setdefault(record["case_id"], {
                key: record[key] for key in ("case_id", "label", "task_type", "prompt", "match_kind")
            })
        return sorted(samples.values(), key=lambda row: self._natural_key(row["label"]))

    @staticmethod
    def _natural_key(value: str) -> list:
        return [(0, int(part)) if part.isdigit() else (1, part.lower()) for part in re.split(r"(\d+)", value)]

    def view(self, payload: dict[str, Any]) -> dict[str, Any]:
        records = self.records(payload["source"])
        columns = payload["columns"]
        selected_ids = set(payload.get("sample_ids", []))
        query = payload.get("query", "").strip().casefold()
        matches: dict[str, dict[str, list]] = {}
        for index, column in enumerate(columns):
            for record in records:
                if record["pipeline"] != column["pipeline"]:
                    continue
                if column.get("config_name") and record["config_name"] != column["config_name"]:
                    continue
                if column.get("run_id") and record["run_id"] != column["run_id"]:
                    continue
                matches.setdefault(record["case_id"], {}).setdefault(str(index), []).append(record)
        rows = []
        for case_id, cells in matches.items():
            all_items = [item for items in cells.values() for item in items]
            sample = all_items[0]
            if selected_ids and case_id not in selected_ids:
                continue
            if query and not any(query in f"{item['label']} {item['sample_id']} {item['prompt']}".casefold() for item in all_items):
                continue
            if payload.get("only_common") and len(cells) != len(columns):
                continue
            fingerprints = {item["input_fingerprint"] for item in all_items if item["input_fingerprint"]}
            rows.append({
                "case_id": case_id, "label": sample["label"], "prompt": sample["prompt"],
                "task_type": sample["task_type"], "match_kind": sample["match_kind"],
                "reference_urls": sample["reference_urls"], "different_inputs": len(fingerprints) > 1,
                "cells": [cells.get(str(index), []) for index in range(len(columns))],
            })
        rows.sort(key=lambda row: self._natural_key(row["label"]))
        total = len(rows)
        page_size = payload.get("page_size", 20)
        page = min(payload.get("page", 1), max(1, (total + page_size - 1) // page_size))
        start = (page - 1) * page_size
        return {"rows": rows[start:start + page_size], "total": total, "page": page, "page_size": page_size}

    def set_group(self, result_ids: list[str], name: str | None) -> None:
        available = {record["result_id"] for record in self.records("users")}
        if not result_ids or set(result_ids) - available:
            raise ValueError("请选择有效的自定义生成结果")
        normalized = normalized_text(name) if name is not None else None
        if name is not None and not normalized:
            raise ValueError("分组名称不能为空")
        with self._lock:
            groups = read_object(self.groups_path)
            for result_id in result_ids:
                if normalized is None:
                    groups.pop(result_id, None)
                else:
                    groups[result_id] = normalized
            atomic_json(self.groups_path, groups)
