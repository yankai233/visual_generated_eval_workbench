from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import SampleModel


class BenchmarkError(ValueError):
    pass


TASK_TYPE_ALIASES = {
    "t2i": "T2I",
    "ti2i": "TI2I",
    "multi_reference": "MultiReference",
    "multireference": "MultiReference",
    "outpainting": "Outpainting",
}


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise BenchmarkError(f"非法的 Benchmark 图片路径: {relative}") from exc
    return candidate


def list_benchmarks(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    results: list[dict[str, Any]] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            samples = load_benchmark(root, directory.name)
            results.append({"name": directory.name, "sample_count": len(samples)})
        except BenchmarkError as exc:
            results.append({"name": directory.name, "sample_count": 0, "error": str(exc)})
    return results


def _sort_path(path: Path) -> tuple[int, int | str]:
    stem = path.stem if path.is_file() else path.name
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def _native_records(benchmark_dir: Path) -> list[tuple[dict[str, Any], Path]]:
    records: list[tuple[dict[str, Any], Path]] = []
    root_json_files = sorted(
        (
            path
            for path in benchmark_dir.glob("*.json")
            if path.name not in {"manifest.json", "info.json"}
        ),
        key=_sort_path,
    )
    info_files = sorted(benchmark_dir.glob("*/info.json"), key=lambda path: _sort_path(path.parent))
    if (benchmark_dir / "info.json").is_file():
        info_files.insert(0, benchmark_dir / "info.json")
    for path in [*root_json_files, *info_files]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise BenchmarkError(f"读取 {path} 失败: {exc}") from exc
        if not isinstance(raw, dict):
            raise BenchmarkError(f"{path} 必须包含一个 JSON 对象")
        records.append((raw, path.parent))
    return records


def _external_outpainting_records(benchmark_dir: Path) -> list[tuple[dict[str, Any], Path]] | None:
    descriptor = benchmark_dir / "outpainting-source.json"
    if not descriptor.is_file():
        return None
    try:
        settings = json.loads(descriptor.read_text(encoding="utf-8"))
        manifest = Path(settings["manifest"]).expanduser().resolve()
        allowed_root = Path(settings["allowed_root"]).expanduser().resolve()
        manifest.relative_to(allowed_root)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"读取扩图 Benchmark 描述失败: {exc}") from exc
    if not manifest.is_file():
        raise BenchmarkError(f"扩图 Benchmark manifest 不存在: {manifest}")

    records: list[tuple[dict[str, Any], Path]] = []
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            source = json.loads(line)
            target = source.get("target_image_info", {})
            evaluation = source.get("evaluation_image_info", {})
            input_path = Path(str(source["input_image_path"])).expanduser().resolve()
            input_path.relative_to(allowed_root)
            known_box = [int(value) for value in evaluation["evaluation_known_box"]]
            records.append(({
                "id": str(source["id"]),
                "task_type": "Outpainting",
                "prompt": str(target.get("prompt", "")),
                "input_images": [input_path.name],
                "seed": 42 + int(source["id"]),
                "width": int(target["width"]),
                "height": int(target["height"]),
                "num_inference_steps": 30,
                "extra": {
                    "input_box": known_box,
                    "source_manifest": str(manifest),
                    "target_image_path": str(target.get("image_path", "")),
                },
            }, input_path.parent))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"读取扩图 manifest 失败: {manifest}: {exc}") from exc
    return records


def _normalize_record(
    raw: dict[str, Any], image_root: Path, language: str
) -> tuple[dict[str, Any], Path]:
    normalized = dict(raw)
    normalized["id"] = str(raw.get("id", ""))
    raw_task = str(raw.get("task_type", "T2I"))
    normalized["task_type"] = TASK_TYPE_ALIASES.get(raw_task.lower(), raw_task)

    prompt = raw.get("prompt", "")
    if isinstance(prompt, dict):
        selected_prompt = prompt.get(language) or prompt.get("zh") or prompt.get("en")
        if selected_prompt is None and prompt:
            selected_prompt = next(iter(prompt.values()))
        normalized["prompt"] = str(selected_prompt or "")
    else:
        normalized["prompt"] = str(prompt or "")

    input_images = list(raw.get("input_images", []))
    if not input_images and raw.get("reference_image"):
        input_images = [raw["reference_image"]]
    reference_images = raw.get("reference_images", [])
    if not input_images and isinstance(reference_images, list):
        input_images = [
            item.get("file") if isinstance(item, dict) else item
            for item in reference_images
        ]
    normalized["input_images"] = [str(path) for path in input_images if path]

    extra = dict(raw.get("extra", {}))
    if isinstance(prompt, dict):
        extra["prompts"] = prompt
    for key in ("evaluation_focus", "source"):
        if key in raw:
            extra[key] = raw[key]
    if isinstance(reference_images, list) and reference_images:
        extra["reference_images"] = reference_images
    normalized["extra"] = extra
    return normalized, image_root


def load_benchmark(root: Path, name: str, language: str = "zh") -> list[SampleModel]:
    if language not in {"zh", "en"}:
        raise BenchmarkError(f"不支持的 Prompt 语言: {language}")
    benchmark_dir = _safe_child(root, name)
    if benchmark_dir.parent != root.resolve() or not benchmark_dir.is_dir():
        raise BenchmarkError(f"Benchmark 不存在: {name}")

    jsonl_path = benchmark_dir / "manifest.jsonl"
    json_path = benchmark_dir / "manifest.json"
    records: list[tuple[dict[str, Any], Path]]
    try:
        external_records = _external_outpainting_records(benchmark_dir)
        if external_records is not None:
            records = external_records
        elif jsonl_path.is_file():
            records = [
                (json.loads(line), benchmark_dir)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif json_path.is_file():
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            raw_samples = loaded.get("samples", []) if isinstance(loaded, dict) else loaded
            records = [(raw, benchmark_dir) for raw in raw_samples]
        else:
            records = _native_records(benchmark_dir)
    except (json.JSONDecodeError, OSError) as exc:
        raise BenchmarkError(f"读取 Benchmark {name} 失败: {exc}") from exc
    if not isinstance(records, list):
        raise BenchmarkError(f"Benchmark {name} 的 manifest 必须是数组或 JSONL")
    if not records:
        raise BenchmarkError(
            f"{name} 未找到 manifest，且未找到 *.json 或 */info.json 样本"
        )

    samples: list[SampleModel] = []
    seen: set[str] = set()
    for index, (raw, image_root) in enumerate(records, start=1):
        if not isinstance(raw, dict):
            raise BenchmarkError(f"{name} 第 {index} 条样本必须是 JSON 对象")
        normalized, image_root = _normalize_record(raw, image_root, language)
        try:
            sample = SampleModel.model_validate(normalized)
        except Exception as exc:
            raise BenchmarkError(f"{name} 第 {index} 条样本无效: {exc}") from exc
        if sample.id in seen:
            raise BenchmarkError(f"{name} 中样本 id 重复: {sample.id}")
        seen.add(sample.id)
        resolved_images: list[str] = []
        for image_path in sample.input_images:
            resolved = _safe_child(image_root, image_path)
            if not resolved.is_file():
                raise BenchmarkError(f"样本 {sample.id} 的图片不存在: {image_path}")
            resolved_images.append(str(resolved))
        sample.input_images = resolved_images
        samples.append(sample)
    return samples

