from __future__ import annotations

from collections import OrderedDict
from contextlib import asynccontextmanager

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.tools.markdown_store import DocumentConflict, MarkdownStore, render_markdown
from src.tools.translation import translate_text, translation_status
from src.utils.paths import TEMPLATE_ROOT


class MarkdownContent(BaseModel):
    content: str = Field(default="", max_length=500_000)


class SaveMarkdown(MarkdownContent):
    name: str = Field(min_length=1, max_length=128)
    revision: str | None = Field(default=None, max_length=64)


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    source: str = Field(default="auto", max_length=10)
    target: str = Field(default="en", max_length=10)


def auxiliary_router(store: MarkdownStore) -> APIRouter:
    translation_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30, connect=10, pool=5),
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
            keepalive_expiry=120,
        ),
    )
    translation_cache: OrderedDict[tuple[str, str, str], dict] = OrderedDict()

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await translation_client.aclose()

    router = APIRouter(lifespan=lifespan)

    @router.get("/tools", include_in_schema=False)
    async def page() -> FileResponse:
        return FileResponse(
            TEMPLATE_ROOT / "tools.html",
            headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
        )

    @router.get("/api/tools/status")
    def status() -> dict:
        return {"translation": translation_status(), "markdown_directory": "output/tools/markdown"}

    @router.get("/api/tools/markdown/files")
    def files() -> dict:
        return {"files": store.list_files()}

    @router.get("/api/tools/markdown/file")
    def read(name: str) -> dict:
        try:
            return store.read(name)
        except FileNotFoundError as exc:
            raise HTTPException(404, "文件不存在") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(400, "无法读取文件：请检查文件名、UTF-8 编码及文件大小") from exc

    @router.post("/api/tools/markdown/file")
    def save(payload: SaveMarkdown) -> dict:
        try:
            return store.save(payload.name, payload.content, payload.revision)
        except DocumentConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(400, "无法保存文件：请检查文件名、文件大小及目录写权限") from exc

    @router.post("/api/tools/markdown/preview")
    def preview(payload: MarkdownContent) -> dict:
        return {"html": render_markdown(payload.content)}

    @router.post("/api/tools/translate")
    async def translate(payload: TranslateRequest) -> dict:
        text = payload.text.strip()
        if not text:
            raise HTTPException(400, "请输入需要翻译的文本")
        cache_key = (text, payload.source, payload.target)
        if cached := translation_cache.get(cache_key):
            translation_cache.move_to_end(cache_key)
            return dict(cached)
        try:
            result = await translate_text(
                text,
                payload.source,
                payload.target,
                client=translation_client,
            )
            translation_cache[cache_key] = dict(result)
            translation_cache.move_to_end(cache_key)
            while len(translation_cache) > 256:
                translation_cache.popitem(last=False)
            return result
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from exc

    return router
