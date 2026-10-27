from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from markdown_it import MarkdownIt


MAX_MARKDOWN_BYTES = 1024 * 1024


class DocumentConflict(ValueError):
    pass


def render_markdown(content: str) -> str:
    # Raw HTML is escaped and dangerous link schemes are rejected by the parser.
    return MarkdownIt("commonmark", {"html": False, "maxNesting": 50}).enable("table").enable("strikethrough").render(content)


class MarkdownStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._lock = threading.RLock()

    def normalize_name(self, name: str) -> str:
        name = name.strip()
        if name and not Path(name).suffix:
            name += ".md"
        if not name or len(name) > 128 or name.startswith(".") or any(ord(char) < 32 or char in "/\\" for char in name):
            raise ValueError("请输入有效的 Markdown 文件名，不允许目录路径")
        if Path(name).suffix.lower() not in {".md", ".markdown"}:
            raise ValueError("文件名必须以 .md 或 .markdown 结尾")
        return name

    def path(self, name: str) -> Path:
        name = self.normalize_name(name)
        path = self.root / name
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("只允许工具目录内的 .md / .markdown 文件")
        return path

    def list_files(self) -> list[dict]:
        if not self.root.exists():
            return []
        files = []
        for path in self.root.iterdir():
            if path.is_file() and not path.is_symlink() and not path.name.startswith(".") and path.suffix.lower() in {".md", ".markdown"}:
                stat = path.stat()
                files.append({"name": path.name, "size": stat.st_size, "modified_at": stat.st_mtime})
        return sorted(files, key=lambda row: row["name"].casefold())

    def read(self, name: str) -> dict:
        with self._lock:
            path = self.path(name)
            if path.stat().st_size > MAX_MARKDOWN_BYTES:
                raise ValueError("Markdown 文件不能超过 1 MiB")
            raw = path.read_bytes()
            return {"name": path.name, "content": raw.decode("utf-8-sig"), "revision": hashlib.sha256(raw).hexdigest()}

    def save(self, name: str, content: str, revision: str | None) -> dict:
        raw = content.encode("utf-8")
        if len(raw) > MAX_MARKDOWN_BYTES:
            raise ValueError("Markdown 文件不能超过 1 MiB")
        with self._lock:
            name = self.normalize_name(name)
            path = self.path(name)
            if path.exists():
                current = self.read(name)
                if not revision or revision != current["revision"]:
                    raise DocumentConflict("文件已存在或被其他页面修改，请重新打开或另存为新名称")
                if current["content"] == content:
                    return current
                backup = self.root / ".history" / name / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ".md")
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_bytes(path.read_bytes())
            elif revision:
                raise DocumentConflict("原文件已被移走，请另存为新文件")
            self.root.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".markdown-", dir=self.root)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return self.read(name)
