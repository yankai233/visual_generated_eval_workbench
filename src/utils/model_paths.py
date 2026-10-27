from __future__ import annotations

import json
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from packaging.version import InvalidVersion, Version


class ModelPathError(ValueError):
    pass


def resolve_pretrained_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise ModelPathError(f"模型目录不存在: {path}")
    if (path / "model_index.json").is_file() or (
        path / "modular_model_index.json"
    ).is_file():
        return path

    refs_main = path / "refs" / "main"
    snapshots = path / "snapshots"
    if refs_main.is_file():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = (snapshots / revision).resolve()
        if not snapshot.is_dir():
            raise ModelPathError(f"HF cache 的 main snapshot 不存在: {snapshot}")
        return snapshot
    if snapshots.is_dir():
        candidates = sorted(
            (child for child in snapshots.iterdir() if child.is_dir()),
            key=lambda child: child.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0].resolve()
    raise ModelPathError(
        f"{path} 既不是 Diffusers 模型目录，也不是可解析的 Hugging Face cache"
    )


@lru_cache(maxsize=64)
def validate_model_weights(resolved_path: str) -> tuple[str, ...]:
    root = Path(resolved_path)
    issues: list[str] = []
    model_index = root / "model_index.json"
    modular_index = root / "modular_model_index.json"
    if not model_index.is_file() and not modular_index.is_file():
        issues.append("缺少 model_index.json 或 modular_model_index.json")

    index_files = sorted(root.rglob("*.safetensors.index.json"))
    for index_path in index_files:
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = sorted(set(data.get("weight_map", {}).values()))
        except (json.JSONDecodeError, OSError) as exc:
            issues.append(f"无法读取权重索引 {index_path.relative_to(root)}: {exc}")
            continue
        for shard_name in shard_names:
            shard = index_path.parent / shard_name
            if not shard.is_file() or shard.stat().st_size == 0:
                issues.append(
                    f"缺少权重分片: {(index_path.parent / shard_name).relative_to(root)}"
                )

    for component in ("transformer", "text_encoder", "text_encoder_2", "vae"):
        component_dir = root / component
        if component_dir.is_dir() and not any(component_dir.glob("*.safetensors")):
            issues.append(f"组件 {component} 中没有 safetensors 权重")
    return tuple(issues)


def check_package_requirements(requirements: Mapping[str, str]) -> list[str]:
    issues: list[str] = []
    for package, minimum in requirements.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            issues.append(f"缺少依赖 {package}>={minimum}")
            continue
        try:
            if Version(installed) < Version(minimum):
                issues.append(f"{package}=={installed}，需要 >= {minimum}")
        except InvalidVersion:
            issues.append(f"无法解析 {package} 版本: {installed}")
    return issues


def inspect_model_config(
    config: Mapping[str, Any], requirements: Mapping[str, str]
) -> dict[str, Any]:
    issues = check_package_requirements(requirements)
    raw_path = str(config.get("model_path", "")).strip()
    resolved_path: str | None = None
    if not raw_path:
        issues.append("配置缺少 model_path")
    else:
        try:
            resolved = resolve_pretrained_path(raw_path)
            resolved_path = str(resolved)
            issues.extend(validate_model_weights(resolved_path))
        except ModelPathError as exc:
            issues.append(str(exc))
    return {
        "available": not issues,
        "issues": issues,
        "configured_model_path": raw_path,
        "resolved_model_path": resolved_path,
    }

