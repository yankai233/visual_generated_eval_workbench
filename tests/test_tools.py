from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.tools.markdown_store import DocumentConflict, MarkdownStore, render_markdown
from src.tools.translation import translate_text, translation_status
from src.web.auxiliary import auxiliary_router


def test_markdown_create_update_conflict_history_and_paths(tmp_path: Path) -> None:
    root = tmp_path / "markdown"
    store = MarkdownStore(root)
    assert store.list_files() == []
    first = store.save("中文笔记.md", "# Hello\n", None)
    assert store.read("中文笔记.md") == first
    assert [item["name"] for item in store.list_files()] == ["中文笔记.md"]
    with pytest.raises(DocumentConflict):
        store.save(first["name"], "overwrite", None)
    second = store.save(first["name"], "# Updated\n", first["revision"])
    assert second["revision"] != first["revision"]
    with pytest.raises(DocumentConflict):
        store.save(first["name"], "stale", first["revision"])
    assert store.read(first["name"])["content"] == "# Updated\n"
    backups = list((root / ".history" / first["name"]).glob("*.md"))
    assert len(backups) == 1
    assert backups[0].read_text() == first["content"]
    extensionless = store.save("随手记录", "不需要手动输入后缀", None)
    assert extensionless["name"] == "随手记录.md"
    assert store.read("随手记录")["content"] == "不需要手动输入后缀"
    for name in ("../outside.md", "/tmp/outside.md", "nested/x.md", "x\\file.md", "code.py", ".hidden.md", "bad\x00.md"):
        with pytest.raises(ValueError):
            store.save(name, "bad", None)
    external = tmp_path / "external.md"
    external.write_text("private")
    (root / "link.md").symlink_to(external)
    with pytest.raises(ValueError):
        store.read("link.md")
    assert external.read_text() == "private"


def test_markdown_render_supports_tables_code_and_rejects_active_html() -> None:
    rendered = render_markdown('# Header\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n```python\nprint("hi")\n```\n\n<script>alert(1)</script>\n\n[bad](javascript:alert(1))')
    assert "<h1>Header</h1>" in rendered
    assert "<table>" in rendered
    assert 'class="language-python"' in rendered
    assert "<script>" not in rendered
    assert 'href="javascript:' not in rendered


def test_tool_routes_crud_conflicts_limits_and_no_translation_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_TRANSLATE_API_KEY", raising=False)
    real_client = httpx.AsyncClient
    translation_requests = []
    def handle(request):
        translation_requests.append(request)
        assert request.url.params["client"] == "gtx"
        return httpx.Response(200, json=[[["你好", "Hello", None, None]], None, "en"])
    monkeypatch.setattr("src.tools.translation.httpx.AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))
    app = FastAPI()
    app.include_router(auxiliary_router(MarkdownStore(tmp_path / "markdown")))
    with TestClient(app) as client:
        tools_page = client.get("/tools")
        assert tools_page.status_code == 200
        assert tools_page.headers["cache-control"] == "no-store, max-age=0"
        assert "translate.google.com" not in tools_page.text
        assert client.get("/api/tools/status").json()["translation"]["mode"] == "web"
        assert client.get("/api/tools/markdown/files").json()["files"] == []
        first = client.post("/api/tools/markdown/file", json={"name": "note.md", "content": "# Note"})
        assert first.status_code == 200
        extensionless = client.post("/api/tools/markdown/file", json={"name": "new note", "content": "works"})
        assert extensionless.status_code == 200
        assert extensionless.json()["name"] == "new note.md"
        assert client.get("/api/tools/markdown/file", params={"name": "note.md"}).json()["content"] == "# Note"
        assert client.post("/api/tools/markdown/file", json={"name": "note.md", "content": "different"}).status_code == 409
        assert client.post("/api/tools/markdown/file", json={**first.json(), "content": "updated"}).status_code == 200
        assert client.get("/api/tools/markdown/file", params={"name": "missing.md"}).status_code == 404
        invalid = client.post("/api/tools/markdown/file", json={"name": "../escape.md"})
        assert invalid.status_code == 400
        assert "有效的 Markdown 文件名" in invalid.json()["detail"]
        assert "<h1>OK</h1>" in client.post("/api/tools/markdown/preview", json={"content": "# OK"}).json()["html"]
        assert client.post("/api/tools/markdown/preview", json={"content": "a" * 500001}).status_code == 422
        translated = client.post("/api/tools/translate", json={"text": "Hello", "target": "zh-CN"})
        assert translated.status_code == 200
        assert translated.json() == {"text": "你好", "detected_source": "en"}
        assert client.post("/api/tools/translate", json={"text": "Hello", "target": "zh-CN"}).json() == translated.json()
        assert len(translation_requests) == 1
        assert client.post("/api/tools/translate", json={"text": " "}).status_code == 400
        assert client.post("/api/tools/translate", json={"text": "a" * 5001}).status_code == 422


def test_google_official_api_payload_and_no_secret_in_status(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_TRANSLATE_API_KEY", "test-secret")
    received = []
    real_client = httpx.AsyncClient

    def handle(request):
        import json
        received.append((str(request.url), request.headers.get("X-Goog-Api-Key"), json.loads(request.content)))
        return httpx.Response(200, json={"data": {"translations": [{"translatedText": "你好 &amp; 欢迎", "detectedSourceLanguage": "en"}]}})

    monkeypatch.setattr("src.tools.translation.httpx.AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))
    result = asyncio.run(translate_text("Hello & welcome", "auto", "zh-CN"))
    assert result == {"text": "你好 & 欢迎", "detected_source": "en"}
    assert received[0] == ("https://translation.googleapis.com/language/translate/v2", "test-secret", {"q": "Hello & welcome", "target": "zh-CN", "format": "text"})
    assert "test-secret" not in str(translation_status())
    asyncio.run(translate_text("你好", "zh-CN", "en"))
    assert received[1][2]["source"] == "zh-CN"
    with pytest.raises(ValueError):
        asyncio.run(translate_text("Hello", "auto", "bad"))


def test_google_failures_do_not_expose_key_or_provider_payload(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_TRANSLATE_API_KEY", "secret")
    real_client = httpx.AsyncClient
    monkeypatch.setattr("src.tools.translation.httpx.AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(lambda request: httpx.Response(403, json={"error": "secret"})), **kwargs))
    with pytest.raises(RuntimeError) as error:
        asyncio.run(translate_text("Hello", "auto", "zh-CN"))
    assert "403" in str(error.value)
    assert "secret" not in str(error.value)
