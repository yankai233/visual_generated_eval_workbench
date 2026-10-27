from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml


SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class ConfigError(ValueError):
    pass


def validate_name(value: str, label: str) -> str:
    if not value or not SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise ConfigError(f"{label} 只能包含字母、数字、点、下划线和连字符")
    return value


def config_path(root: Path, pipeline: str, config_name: str) -> Path:
    pipeline = validate_name(pipeline, "Pipeline 名称")
    config_name = validate_name(config_name, "配置名称")
    return root / pipeline / config_name / "config.yaml"


def parse_yaml_mapping(content: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 语法错误: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError("YAML 顶层必须是键值映射")
    return data


def list_configs(root: Path, pipeline: str) -> list[str]:
    directory = root / validate_name(pipeline, "Pipeline 名称")
    if not directory.exists():
        return []
    return sorted(
        child.name
        for child in directory.iterdir()
        if child.is_dir() and (child / "config.yaml").is_file()
    )


def read_config(root: Path, pipeline: str, config_name: str) -> tuple[str, dict[str, Any]]:
    path = config_path(root, pipeline, config_name)
    if not path.is_file():
        raise ConfigError(f"配置不存在: {pipeline}/{config_name}")
    content = path.read_text(encoding="utf-8")
    return content, parse_yaml_mapping(content)


def write_config(root: Path, pipeline: str, config_name: str, content: str) -> Path:
    parse_yaml_mapping(content)
    path = config_path(root, pipeline, config_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".config-", suffix=".yaml", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content.rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path

