"""Validated, revision-checked source editing for explicitly managed plugins.

AST validation is an interface check, NOT a Python sandbox. The HTTP endpoint
requires an administrator token because registering a plugin executes Python.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import secrets
import tempfile
import threading
from pathlib import Path
from typing import Callable

from src.utils.config_io import parse_yaml_mapping


class PipelineSourceError(ValueError):
    pass


def pipeline_template(name: str) -> dict:
    validate_name(name)
    return {"name": name, "code": f'''from src.pipelines.base import (
    BaseGenerationPipeline, GenerationRequest, GenerationResult,
    PipelineControl, ProgressCallback,
)


class UserPipeline(BaseGenerationPipeline):
    pipeline_name = "{name}"
    display_name = "{name}"
    description = "自定义模型"
    pipeline_group = "Other"  # FLUX / Qwen / Other / Test
    parameter_size = "未说明"  # 如 12B
    supported_tasks = ("T2I",)  # TI2I / MultiReference
    max_reference_images = 0  # None 表示不限制
    supported_devices = ("cuda",)

    @classmethod
    def from_pretrained(cls, config, device):
        # config 来自下方 YAML，device 如 cuda:0。
        # 加载模型后调用 cls.configure_model_device(model, config, device)，
        # 即可统一支持 YAML 中的 cpu_offload；不要在模块顶层加载模型。
        raise NotImplementedError("请实现模型加载")

    def __call__(self, request: GenerationRequest, *,
                 progress_callback: ProgressCallback | None = None,
                 control: PipelineControl | None = None) -> GenerationResult:
        # request.prompt / input_images (Path 列表) / seed / width / height
        # request.num_inference_steps / negative_prompt / extra
        # 每个去噪步骤先 control.checkpoint()，再 progress_callback(step, total)。
        # 返回 GenerationResult(images=[PIL.Image, ...])。
        raise NotImplementedError("请实现生成入口")
''', "yaml": 'model_path: /path/to/model\ncpu_offload: false\n# guidance_scale: 3.5\n', "revision": None}


def validate_name(name: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
        raise PipelineSourceError("名称仅支持小写字母、数字、下划线，以字母开头，最多 64 字符")


def validate_source(name: str, code: str, yaml: str) -> str:
    validate_name(name)
    if len(code.encode()) > 256 * 1024 or len(yaml.encode()) > 64 * 1024:
        raise PipelineSourceError("Python 或 YAML 内容过大")
    try:
        tree = ast.parse(code)
        compile(tree, "pipeline.py", "exec")
    except SyntaxError as exc:
        raise PipelineSourceError(f"Python 第 {exc.lineno} 行: {exc.msg}") from exc
    parse_yaml_mapping(yaml)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and any(
        isinstance(base, ast.Name) and base.id == "BaseGenerationPipeline" for base in node.bases
    )]
    if len(classes) != 1:
        raise PipelineSourceError("需要且只能有一个直接继承 BaseGenerationPipeline 的类")
    cls = classes[0]
    attributes = {target.id: node.value for node in cls.body if isinstance(node, ast.Assign)
                  for target in node.targets if isinstance(target, ast.Name)}
    if not isinstance(attributes.get("pipeline_name"), ast.Constant) or attributes["pipeline_name"].value != name:
        raise PipelineSourceError("pipeline_name 必须是与名称一致的字符串")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    for method_name, required in (("from_pretrained", ("config", "device")),
                                  ("__call__", ("request", "progress_callback", "control"))):
        method = methods.get(method_name)
        if method is None:
            raise PipelineSourceError(f"必须实现同步方法 {method_name}")
        parameters = {arg.arg for arg in [*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs]}
        if not set(required).issubset(parameters):
            raise PipelineSourceError(f"{method_name} 需要参数: {', '.join(required)}")
        if method_name == "from_pretrained" and not any(
            isinstance(dec, ast.Name) and dec.id == "classmethod" for dec in method.decorator_list
        ):
            raise PipelineSourceError("from_pretrained 必须使用 @classmethod")
        body = [node for node in method.body if not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        )]
        if not body or all(isinstance(node, ast.Pass) or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and node.value.value is Ellipsis
        ) for node in body) or any(isinstance(node, ast.Name) and node.id == "NotImplementedError" for node in ast.walk(method)):
            raise PipelineSourceError(f"请先完成 {method_name}，不能保存未实现的示例")
    return cls.name


def atomic_text(path: Path, content: str, mode: int = 0o644) -> None:
    descriptor, filename = tempfile.mkstemp(prefix=".workbench-", dir=path.parent)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(filename, path)
    finally:
        Path(filename).unlink(missing_ok=True)


class PipelineSourceStore:
    def __init__(self, pipeline_root: Path, config_root: Path, token_path: Path):
        self.pipeline_root = pipeline_root.resolve()
        self.config_root = config_root.resolve()
        self.token_path = token_path
        self.lock = threading.RLock()

    def token(self) -> str:
        configured = os.environ.get("WORKBENCH_PIPELINE_TOKEN")
        if configured:
            return configured
        with self.lock:
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.token_path.exists():
                # Exclusive creation; do not replace a token created by another process.
                try:
                    descriptor = os.open(self.token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                        handle.write(secrets.token_urlsafe(32))
                except FileExistsError:
                    pass
            return self.token_path.read_text().strip()

    def authorized(self, provided: str) -> bool:
        return bool(provided) and secrets.compare_digest(provided, self.token())

    def paths(self, name: str) -> tuple[Path, Path, Path, Path]:
        validate_name(name)
        if (self.pipeline_root / f"{name}.py").exists():
            raise PipelineSourceError("名称与项目内置模块冲突")
        package = self.pipeline_root / name
        config_dir = self.config_root / name / "default"
        for root, leaf in ((self.pipeline_root, package), (self.config_root, config_dir)):
            current = leaf
            while current != root:
                if current.is_symlink():
                    raise PipelineSourceError("不能编辑符号链接目录")
                current = current.parent
        files = (package / "pipeline.py", package / "__init__.py", config_dir / "config.yaml", package / ".workbench-user.json")
        if any(path.is_symlink() for path in files):
            raise PipelineSourceError("不能编辑符号链接文件")
        return files

    def read(self, name: str) -> dict:
        code, init, config, marker = self.paths(name)
        if not marker.is_file():
            raise PipelineSourceError("仅可编辑通过网页创建的 Pipeline；内置或手动创建的代码不会被覆盖")
        content, yaml, init_text = code.read_text(), config.read_text(), init.read_text()
        revision = hashlib.sha256((content + "\0" + yaml + "\0" + init_text).encode()).hexdigest()
        return {"name": name, "code": content, "yaml": yaml, "revision": revision}

    def list_managed(self) -> list[str]:
        return sorted(path.parent.name for path in self.pipeline_root.glob("*/.workbench-user.json")
                      if not path.parent.is_symlink())

    def save(self, name: str, code: str, yaml: str, revision: str | None,
             register: Callable[[str], None]) -> dict:
        class_name = validate_source(name, code, yaml)
        with self.lock:
            code_path, init_path, config_path, marker = self.paths(name)
            existing = code_path.parent.exists() or config_path.parent.parent.exists()
            if existing:
                current = self.read(name)
                if revision != current["revision"]:
                    raise PipelineSourceError("文件已被修改，请重新打开后再保存")
            elif revision:
                raise PipelineSourceError("Pipeline 不存在，请作为新 Pipeline 保存")
            paths = (code_path, init_path, config_path, marker)
            previous = {path: path.read_text() if path.exists() else None for path in paths}
            code_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                atomic_text(code_path, code)
                atomic_text(init_path, f"from .pipeline import {class_name}\n\nPIPELINE_CLASS = {class_name}\n")
                atomic_text(config_path, yaml)
                atomic_text(marker, json.dumps({"version": 1, "name": name}))
                register(name)
            except Exception:
                for path, content in previous.items():
                    if content is None:
                        path.unlink(missing_ok=True)
                    else:
                        atomic_text(path, content)
                # Imports may have compiled the rejected source. Remove only these
                # two generated cache files, so retries cannot see stale bytecode.
                for source in (code_path, init_path):
                    cached = Path(importlib.util.cache_from_source(str(source)))
                    if cached.is_file() and not cached.is_symlink():
                        cached.unlink()
                try:
                    (code_path.parent / "__pycache__").rmdir()
                except OSError:
                    pass
                if not existing:
                    # Only remove the empty directories created by this save.
                    for directory in (config_path.parent, config_path.parent.parent, code_path.parent):
                        try:
                            directory.rmdir()
                        except OSError:
                            pass
                raise
            return {**self.read(name), "source_path": str(code_path), "config_path": str(config_path)}
