from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache
from pathlib import Path
from typing import Type

from .base import BaseGenerationPipeline


class PipelineRegistryError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _scan_pipelines() -> tuple[
    dict[str, Type[BaseGenerationPipeline]], tuple[str, ...]
]:
    package_dir = Path(__file__).resolve().parent
    found: dict[str, Type[BaseGenerationPipeline]] = {}
    errors: list[str] = []

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if not module_info.ispkg or module_info.name.startswith("_"):
            continue
        module_name = f"{__package__}.{module_info.name}"
        try:
            module = importlib.import_module(module_name)
            pipeline_cls = getattr(module, "PIPELINE_CLASS")
        except Exception as exc:
            errors.append(f"{module_info.name}: {exc}")
            continue
        if not isinstance(pipeline_cls, type) or not issubclass(
            pipeline_cls, BaseGenerationPipeline
        ):
            errors.append(f"{module_info.name}: PIPELINE_CLASS 不是 BaseGenerationPipeline 子类")
            continue
        name = pipeline_cls.pipeline_name or module_info.name
        if name in found:
            errors.append(f"{module_info.name}: pipeline_name 重复: {name}")
            continue
        found[name] = pipeline_cls

    if errors and not found:
        raise PipelineRegistryError("Pipeline 加载失败: " + "; ".join(errors))
    return found, tuple(errors)


def discover_pipelines() -> dict[str, Type[BaseGenerationPipeline]]:
    return _scan_pipelines()[0]


def pipeline_discovery_errors() -> list[str]:
    return list(_scan_pipelines()[1])


def get_pipeline(name: str) -> Type[BaseGenerationPipeline]:
    try:
        return discover_pipelines()[name]
    except KeyError as exc:
        raise PipelineRegistryError(f"未找到 Pipeline: {name}") from exc


def clear_pipeline_cache() -> None:
    _scan_pipelines.cache_clear()

