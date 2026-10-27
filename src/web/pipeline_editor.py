from __future__ import annotations

import importlib
import inspect
import sys

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.pipelines.registry import clear_pipeline_cache, get_pipeline
from src.utils.paths import CONFIG_ROOT, SRC_ROOT
from src.utils.pipeline_sources import PipelineSourceStore, pipeline_template


class PipelineSourceRequest(BaseModel):
    name: str = Field(max_length=64)
    code: str = Field(max_length=256 * 1024)
    yaml: str = Field(max_length=64 * 1024)
    revision: str | None = None


def reload_pipeline(name: str) -> None:
    prefix = f"src.pipelines.{name}"
    for module in list(sys.modules):
        if module == prefix or module.startswith(prefix + "."):
            del sys.modules[module]
    # Source editors can produce same-length edits in one filesystem timestamp tick.
    for cached in (SRC_ROOT / "pipelines" / name / "__pycache__").glob("*.pyc"):
        if cached.is_file() and not cached.is_symlink():
            cached.unlink()
    importlib.invalidate_caches()
    clear_pipeline_cache()
    cls = get_pipeline(name)
    if cls.__abstractmethods__:
        raise ValueError("Pipeline 仍有未实现的抽象接口")
    # Match the actual worker invocation without instantiating/loading the model.
    inspect.signature(cls.from_pretrained).bind({}, "cpu")
    inspect.signature(cls.__call__).bind(None, None, progress_callback=None, control=None)


def pipeline_editor_router(manager, payload_builder) -> APIRouter:
    router = APIRouter(prefix="/api/pipeline-editor")
    store = PipelineSourceStore(SRC_ROOT / "pipelines", CONFIG_ROOT, CONFIG_ROOT / ".pipeline-editor-token")

    def authenticate(request: Request) -> None:
        # Token is never sent in bootstrap, URL, logs, or browser persistent storage.
        authorization = request.headers.get("authorization", "")
        token = authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
        if not store.authorized(token):
            raise HTTPException(403, "保存口令无效；从服务器 configs/.pipeline-editor-token 获取，或设置 WORKBENCH_PIPELINE_TOKEN")
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.url.netloc}":
            raise HTTPException(403, "不允许跨站编辑代码")

    @router.get("/template")
    async def template(name: str = "my_pipeline") -> dict:
        try:
            # Create the protected token file without returning its contents.
            store.token()
            return {**pipeline_template(name), "managed": store.list_managed()}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.get("/source/{name}")
    async def read(name: str, request: Request) -> dict:
        authenticate(request)
        try:
            return store.read(name)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.post("/source")
    async def save(body: PipelineSourceRequest, request: Request) -> dict:
        authenticate(request)
        if any(job["status"] not in {"completed", "completed_with_errors", "cancelled", "failed", "interrupted"}
               for job in manager.list_snapshots()):
            raise HTTPException(409, "有任务正在运行，请结束任务后再注册 Python 代码")
        try:
            metadata = {}

            def register(name):
                reload_pipeline(name)
                metadata.update(payload_builder(name, get_pipeline(name)))

            saved = await run_in_threadpool(store.save, body.name, body.code, body.yaml, body.revision, register)
            saved["pipeline"] = metadata
            return saved
        except Exception as exc:
            prefix = f"src.pipelines.{body.name}"
            for module in list(sys.modules):
                if module == prefix or module.startswith(prefix + "."):
                    del sys.modules[module]
            clear_pipeline_cache()
            raise HTTPException(400, f"保存或注册失败：{exc}") from exc

    return router
