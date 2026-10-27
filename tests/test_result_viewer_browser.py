from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from PIL import Image


pytestmark = pytest.mark.skipif(not os.environ.get("WORKBENCH_BROWSER_URL"), reason="requires running workbench and Playwright")


def test_image_preview_close_download_manual_save_expiry_and_auto_save():
    playwright = pytest.importorskip("playwright.sync_api")
    pipeline = {"name": "demo", "display_name": "Demo", "group": "Test", "supported_tasks": ["T2I"],
                "supported_devices": ["cpu"], "max_reference_images": 0, "configs": ["default"],
                "config_statuses": {"default": {"available": True, "issues": []}}}
    posts = []
    saves = []
    jobs = []
    expired = [False]
    buffer = io.BytesIO()
    Image.new("RGB", (360, 240), "#8da596").save(buffer, format="PNG")
    png = buffer.getvalue()

    def bootstrap(route):
        route.fulfill(json={"pipelines": [pipeline], "gpus": [], "cuda": {"available": False}, "benchmarks": [], "jobs": jobs})

    def submit(route):
        body = route.request.post_data_json
        posts.append(body)
        result = {"unit_id": "demo:sample:1", "sample_id": "sample", "benchmark": "users", "output_dir": "output/gen/demo/users/sample-batch", "images": []}
        if not body["save_images"]:
            result.update(preview_id="temporary", preview_count=2, saved_images={}, image_urls=["/api/previews/temporary/0", "/api/previews/temporary/1"])
        else:
            result["image_urls"] = ["/test-saved.png"]
        job = {"id": f"preview-test-{len(posts)}", "status": "completed", "progress": 1, "total": 1,
               "completed": 1, "results": [result], "errors": [], "active": {}}
        jobs.insert(0, job)
        route.fulfill(status=201, json=job)

    def preview(route):
        if expired[0] and "/1" in route.request.url:
            route.fulfill(status=410, json={"detail": "临时图像已过期"})
        else:
            headers = {"Cache-Control": "no-store"}
            if route.request.url.endswith("/download"):
                headers["Content-Disposition"] = 'attachment; filename="generated.png"'
            route.fulfill(body=png, content_type="image/png", headers=headers)

    def save(route):
        saves.append(route.request.url)
        if len(saves) == 1:
            route.fulfill(status=400, json={"detail": "保存失败：测试错误"})
        else:
            jobs[-1]["results"][0]["saved_images"]["0"] = "/tmp/generated.png"
            jobs[-1]["results"][0]["image_urls"][0] = "/test-saved.png"
            route.fulfill(json={"url": "/test-saved.png", "image_index": 0, "output_dir": "output/gen/demo/users/sample-batch"})

    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/api/bootstrap", bootstrap)
        page.route("**/api/jobs", submit)
        page.route("**/api/previews/temporary/*", preview)
        page.route("**/api/previews/temporary/*/download", preview)
        page.route("**/api/previews/temporary/*/save", save)
        page.route("**/test-saved.png", lambda route: route.fulfill(body=png, content_type="image/png"))
        page.goto(os.environ["WORKBENCH_BROWSER_URL"], wait_until="networkidle")
        first = page.locator("pipeline-workspace").first
        playwright.expect(first.locator("#saveImagesInput")).to_be_checked()
        first.locator("#saveImagesInput").uncheck()
        playwright.expect(first.locator("#outputLocationField")).not_to_be_visible()
        first.locator("#promptInput").fill("preview test")
        first.locator("#runBtn").click()
        gallery = page.locator("model-results").first
        playwright.expect(gallery.locator(".result-card")).to_have_count(2)
        assert posts[0]["save_images"] is False
        gallery.locator(".result-open").first.click()
        viewer = gallery.locator("#imageViewerDialog")
        playwright.expect(viewer).to_be_visible()
        playwright.expect(viewer.locator("#saveResultBtn")).to_be_enabled()
        assert viewer.locator("#viewerImage").evaluate("image => image.complete && image.naturalWidth > 0")
        viewer.locator("#closeImageViewerBtn").click()
        playwright.expect(viewer).not_to_be_visible()
        assert saves == []
        gallery.locator(".result-open").first.click()
        with page.expect_download() as download_info:
            viewer.locator("#downloadResultBtn").click()
        download = download_info.value
        assert Path(download.path()).read_bytes() == png
        assert saves == []
        viewer.locator("#saveResultBtn").click()
        playwright.expect(viewer.locator("#imageViewerError")).to_have_text("保存失败：测试错误")
        viewer.locator("#saveResultBtn").click()
        playwright.expect(viewer.locator("#saveResultBtn")).to_have_text("已保存")
        playwright.expect(viewer.locator("#saveResultBtn")).to_be_disabled()
        playwright.expect(gallery.locator(".result-card")).to_have_count(2)
        playwright.expect(gallery.locator(".result-open img").first).to_have_attribute("src", "/test-saved.png")
        page.screenshot(path="logs/ui-result-viewer.png", full_page=False)
        page.keyboard.press("Escape")
        playwright.expect(viewer).not_to_be_visible()
        expired[0] = True
        page.reload(wait_until="networkidle")
        playwright.expect(first.locator("#saveImagesInput")).not_to_be_checked()
        playwright.expect(gallery.locator(".result-card")).to_have_count(2)
        gallery.locator(".result-open").first.click()
        playwright.expect(viewer.locator("#saveResultBtn")).to_be_disabled()
        viewer.locator("#closeImageViewerBtn").click()
        gallery.locator(".result-open").nth(1).click()
        playwright.expect(viewer.locator("#imageViewerError")).to_contain_text("过期")
        playwright.expect(viewer.locator("#saveResultBtn")).to_be_disabled()
        playwright.expect(viewer.locator("#downloadResultBtn")).to_be_disabled()
        viewer.locator("#closeImageViewerBtn").click()
        page.locator("#addModelBtn").click()
        second = page.locator("pipeline-workspace").nth(1)
        playwright.expect(second.locator("#saveImagesInput")).not_to_be_checked()
        second.locator("#saveImagesInput").check()
        second.locator("#runBtn").click()
        assert posts[-1]["save_images"] is True
        second_gallery = page.locator("model-results").nth(1)
        second_gallery.locator(".result-open").click()
        second_viewer = second_gallery.locator("#imageViewerDialog")
        playwright.expect(second_viewer.locator("#saveResultBtn")).to_be_disabled()
        with page.expect_download() as download_info:
            second_viewer.locator("#downloadResultBtn").click()
        assert Path(download_info.value.path()).read_bytes() == png
        page.set_viewport_size({"width": 390, "height": 844})
        assert second_viewer.bounding_box()["width"] <= 390
        second_viewer.locator("#closeImageViewerBtn").click()
        assert len(saves) == 2
        assert not errors
        browser.close()
