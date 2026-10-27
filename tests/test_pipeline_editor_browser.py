from __future__ import annotations

import os

import pytest

from src.utils.pipeline_sources import pipeline_template


pytestmark = pytest.mark.skipif(not os.environ.get("WORKBENCH_BROWSER_URL"), reason="requires running workbench and Playwright")


def test_web_editor_saves_code_and_registers_only_in_target_model():
    playwright = pytest.importorskip("playwright.sync_api")
    pipeline = {"name": "demo", "display_name": "Demo", "group": "Test", "supported_tasks": ["T2I"],
                "supported_devices": ["cpu"], "max_reference_images": 0, "configs": ["default"],
                "config_statuses": {"default": {"available": True, "issues": []}}}
    bootstrap = {"pipelines": [pipeline], "gpus": [], "cuda": {"available": False}, "jobs": [], "benchmarks": []}
    posts = []

    def save(route):
        posts.append(route.request.post_data_json)
        assert route.request.headers["authorization"] == "Bearer browser-test-secret"
        if len(posts) == 1:
            route.fulfill(status=400, json={"detail": "请先完成 from_pretrained"})
        else:
            route.fulfill(json={**posts[-1], "revision": "saved-revision", "pipeline": {**pipeline, "name": "my_pipeline", "display_name": "Custom", "group": "Other"}})

    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/api/bootstrap", lambda route: route.fulfill(json=bootstrap))
        page.route("**/api/pipeline-editor/template", lambda route: route.fulfill(json={**pipeline_template("my_pipeline"), "managed": []}))
        page.route("**/api/pipeline-editor/source", save)
        page.goto(os.environ["WORKBENCH_BROWSER_URL"], wait_until="networkidle")
        page.locator("#addModelBtn").click()
        rows = page.locator("pipeline-workspace")
        rows.nth(1).locator("#newPipelineBtn").click()
        dialog = page.locator("#pipelineEditorDialog")
        playwright.expect(dialog).to_be_visible()
        playwright.expect(dialog.locator("#pipelineSourceCode")).to_have_value(pipeline_template("my_pipeline")["code"])
        dialog.locator("#pipelineEditorToken").fill("browser-test-secret")
        dialog.locator("#savePipelineSourceBtn").click()
        playwright.expect(dialog.locator("#pipelineEditorError")).to_have_text("请先完成 from_pretrained")
        playwright.expect(dialog).to_be_visible()
        code = pipeline_template("my_pipeline")["code"].replace('raise NotImplementedError("请实现模型加载")', 'return cls()').replace('raise NotImplementedError("请实现生成入口")', 'return GenerationResult(images=[])')
        dialog.locator("#pipelineSourceCode").fill(code)
        dialog.locator("#pipelineSourceYaml").fill("model_path: /local/model\n")
        dialog.locator("#savePipelineSourceBtn").click()
        playwright.expect(dialog).not_to_be_visible()
        playwright.expect(rows.nth(1).locator("[data-action=pipeline]")).to_have_value("my_pipeline")
        playwright.expect(rows.nth(0).locator("[data-action=pipeline]")).to_have_value("demo")
        assert posts[-1]["code"] == code
        assert posts[-1]["yaml"] == "model_path: /local/model\n"
        assert posts[-1]["revision"] is None
        assert page.locator("#pipelineEditorToken").input_value() == ""
        assert "browser-test-secret" not in page.evaluate("JSON.stringify(sessionStorage)")
        assert not errors
        browser.close()
