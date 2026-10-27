from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.utils.pipeline_sources import PipelineSourceError, PipelineSourceStore, pipeline_template, validate_source
from src.web.pipeline_editor import pipeline_editor_router


def implemented(name="custom"):
    source = pipeline_template(name)
    source["code"] = source["code"].replace('raise NotImplementedError("请实现模型加载")', 'return cls()')
    source["code"] = source["code"].replace('raise NotImplementedError("请实现生成入口")', 'return GenerationResult(images=[])')
    return source


def make_store(tmp_path):
    pipeline_root = tmp_path / "pipelines"
    pipeline_root.mkdir()
    return PipelineSourceStore(pipeline_root, tmp_path / "configs", tmp_path / "private" / "token")


def test_template_requires_implementation_and_correct_interface():
    template = pipeline_template("custom")
    with pytest.raises(PipelineSourceError, match="完成 from_pretrained"):
        validate_source("custom", template["code"], template["yaml"])
    source = implemented()
    assert validate_source("custom", source["code"], source["yaml"]) == "UserPipeline"
    for code in ("not python!!", source["code"].replace("@classmethod", ""),
                 source["code"].replace("def __call__", "async def __call__"),
                 source["code"].replace('pipeline_name = "custom"', 'pipeline_name = "other"')):
        with pytest.raises(PipelineSourceError):
            validate_source("custom", code, source["yaml"])
    for name in ("../evil", "__init__", "Mixed", "name-with-dash", ""):
        with pytest.raises(PipelineSourceError):
            pipeline_template(name)


def test_save_revision_protection_builtin_symlink_and_rollback(tmp_path):
    store = make_store(tmp_path)
    registered = []
    source = implemented()
    saved = store.save(**source, register=registered.append)
    assert registered == ["custom"]
    assert Path(saved["source_path"]).read_text() == source["code"]
    assert Path(saved["config_path"]).read_text() == source["yaml"]
    assert "PIPELINE_CLASS = UserPipeline" in (store.pipeline_root / "custom" / "__init__.py").read_text()
    assert store.list_managed() == ["custom"]
    with pytest.raises(PipelineSourceError, match="已被修改"):
        store.save(**source, register=registered.append)
    saved2 = store.save("custom", source["code"] + "\n# changed\n", source["yaml"], saved["revision"], registered.append)
    assert saved2["revision"] != saved["revision"]
    def fail(_):
        raise RuntimeError("import failed")
    with pytest.raises(RuntimeError):
        store.save("custom", source["code"], source["yaml"], saved2["revision"], fail)
    assert store.read("custom")["revision"] == saved2["revision"]
    builtin = store.pipeline_root / "builtin"
    builtin.mkdir()
    with pytest.raises(PipelineSourceError, match="不会被覆盖"):
        store.save(**implemented("builtin"), register=registered.append)
    (store.pipeline_root / "base.py").touch()
    with pytest.raises(PipelineSourceError, match="内置模块冲突"):
        store.save(**implemented("base"), register=registered.append)
    (store.pipeline_root / "linked").symlink_to(builtin, target_is_directory=True)
    with pytest.raises(PipelineSourceError, match="符号链接"):
        store.save(**implemented("linked"), register=registered.append)
    with pytest.raises(PipelineSourceError):
        store.save("bad", "bad python!", "{}", None, registered.append)
    assert not (store.pipeline_root / "bad").exists()


def test_token_is_private_and_stable(tmp_path, monkeypatch):
    monkeypatch.delenv("WORKBENCH_PIPELINE_TOKEN", raising=False)
    store = make_store(tmp_path)
    assert not store.authorized("")
    token = store.token()
    assert store.authorized(token)
    assert store.token_path.stat().st_mode & 0o777 == 0o600
    assert store.token() == token


def test_editor_api_authentication_execution_and_active_job_guard(tmp_path, monkeypatch):
    import src.web.pipeline_editor as editor
    pipeline_root = tmp_path / "src" / "pipelines"
    pipeline_root.mkdir(parents=True)
    monkeypatch.setattr(editor, "SRC_ROOT", tmp_path / "src")
    monkeypatch.setattr(editor, "CONFIG_ROOT", tmp_path / "configs")
    monkeypatch.setenv("WORKBENCH_PIPELINE_TOKEN", "test-private-token")
    registrations = []
    monkeypatch.setattr(editor, "reload_pipeline", registrations.append)
    monkeypatch.setattr(editor, "get_pipeline", lambda name: object)
    jobs = []
    app = FastAPI()
    app.include_router(pipeline_editor_router(SimpleNamespace(list_snapshots=lambda: jobs), lambda name, cls: {"name": name}))
    with TestClient(app) as client:
        assert client.get("/api/pipeline-editor/template").status_code == 200
        payload = implemented()
        assert client.post("/api/pipeline-editor/source", json=payload).status_code == 403
        headers = {"Authorization": "Bearer test-private-token"}
        assert client.post("/api/pipeline-editor/source", json=payload, headers={**headers, "Origin": "https://evil.example"}).status_code == 403
        jobs.append({"status": "running"})
        assert client.post("/api/pipeline-editor/source", json=payload, headers=headers).status_code == 409
        jobs.clear()
        response = client.post("/api/pipeline-editor/source", json=payload, headers=headers)
        assert response.status_code == 200, response.text
        assert response.json()["pipeline"]["name"] == "custom"
        assert registrations == ["custom"]
        assert client.get("/api/pipeline-editor/source/custom", headers=headers).status_code == 200
        assert client.get("/api/pipeline-editor/source/custom").status_code == 403


def test_actual_registration_failed_import_retry_and_revision_reload(tmp_path, monkeypatch):
    import sys
    import src.pipelines as package
    import src.pipelines.registry as registry
    import src.web.pipeline_editor as editor
    pipeline_root = tmp_path / "src" / "pipelines"
    pipeline_root.mkdir(parents=True)
    monkeypatch.setattr(package, "__path__", [*package.__path__, str(pipeline_root)])
    monkeypatch.setattr(registry, "__file__", str(pipeline_root / "registry.py"))
    monkeypatch.setattr(editor, "SRC_ROOT", tmp_path / "src")
    monkeypatch.setattr(editor, "CONFIG_ROOT", tmp_path / "configs")
    monkeypatch.setenv("WORKBENCH_PIPELINE_TOKEN", "test-token")
    registry.clear_pipeline_cache()
    app = FastAPI()
    app.include_router(pipeline_editor_router(SimpleNamespace(list_snapshots=lambda: []), lambda name, cls: cls.metadata()))
    name = "web_test_plugin"
    source = implemented(name)
    headers = {"Authorization": "Bearer test-token"}
    try:
        with TestClient(app) as client:
            bad = {**source, "code": "import nonexistent_workbench_dependency\n" + source["code"]}
            response = client.post("/api/pipeline-editor/source", json=bad, headers=headers)
            assert response.status_code == 400
            assert not (pipeline_root / name).exists()
            response = client.post("/api/pipeline-editor/source", json=source, headers=headers)
            assert response.status_code == 200, response.text
            registered = registry.get_pipeline(name)
            instance = registered.from_pretrained({}, "cpu")
            assert instance(None).images == []
            revision = response.json()["revision"]
            changed = {**source, "revision": revision, "code": source["code"].replace('display_name = "web_test_plugin"', 'display_name = "edited_pipeline"')}
            response = client.post("/api/pipeline-editor/source", json=changed, headers=headers)
            assert response.status_code == 200, response.text
            assert registry.get_pipeline(name).display_name == "edited_pipeline"
    finally:
        for module in list(sys.modules):
            if module == f"src.pipelines.{name}" or module.startswith(f"src.pipelines.{name}."):
                del sys.modules[module]
        registry.clear_pipeline_cache()
