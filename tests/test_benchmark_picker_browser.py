from __future__ import annotations

import json
import os
from urllib.parse import parse_qs, urlparse

import pytest


pytestmark = pytest.mark.skipif(not os.environ.get("WORKBENCH_BROWSER_URL"), reason="requires running workbench and Playwright")


def test_multiple_benchmarks_filter_cancel_pagination_restore_and_independence() -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    posts = []
    requests = []
    bootstrap = {
        "pipelines": [{"name": "demo", "display_name": "Demo", "group": "Test", "supported_tasks": ["T2I", "TI2I", "MultiReference"], "supported_devices": ["cpu"], "max_reference_images": None, "configs": ["default"], "config_statuses": {"default": {"available": True, "issues": []}}}],
        "gpus": [], "cuda": {"available": False}, "jobs": [],
        "benchmarks": [{"name": "first", "sample_count": 61}, {"name": "second", "sample_count": 2}],
    }

    def samples(route):
        parsed = urlparse(route.request.url)
        name = parsed.path.split("/")[-2]
        language = parse_qs(parsed.query)["language"][0]
        requests.append((name, language))
        rows = [{"id": str(index), "task_type": "T2I" if name == "first" else "TI2I", "prompt": f"{language} {name} sample {index}", "input_images": [] if name == "first" else ["ref.png"], "input_image_urls": []} for index in range(1, 62 if name == "first" else 3)]
        route.fulfill(json={"samples": rows})

    def submit(route):
        posts.append(route.request.post_data_json)
        route.fulfill(status=201, json={"id": "multi-ui", "status": "running", "progress": 0, "completed": 0, "total": 61, "message": "running", "results": [], "errors": [], "active": {}})

    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1550, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.add_init_script("window.EventSource = class { constructor(url) { this.url = url; } close() {} };")
        page.route("**/api/bootstrap", lambda route: route.fulfill(json=bootstrap))
        page.route("**/api/benchmarks/*/samples?*", samples)
        page.route("**/api/jobs", submit)
        page.goto(os.environ["WORKBENCH_BROWSER_URL"], wait_until="networkidle")
        workspace = page.locator("pipeline-workspace").first
        workspace.locator('[data-mode="benchmark"]').click()
        playwright.expect(workspace.locator(".benchmark-item")).to_have_count(2)
        playwright.expect(workspace.locator("#sampleTotal")).to_have_text("61")
        workspace.locator('[data-benchmark-check="1"]').check()
        playwright.expect(workspace.locator("#sampleTotal")).to_have_text("63")
        workspace.locator('[data-benchmark-filter="0"]').click()
        dialog = workspace.locator("#benchmarkSamplesDialog")
        playwright.expect(dialog).to_be_visible()
        playwright.expect(dialog.locator(".sample-item")).to_have_count(50)
        dialog.locator("#nextSamplesBtn").click()
        playwright.expect(dialog.locator(".sample-item")).to_have_count(11)
        dialog.locator("#sampleSearch").fill("sample 61")
        playwright.expect(dialog.locator(".sample-item")).to_have_count(1)
        dialog.locator("#clearBenchmarkSamplesBtn").click()
        dialog.locator("#applyBenchmarkSamplesBtn").click()
        playwright.expect(workspace.locator("#sampleTotal")).to_have_text("62")
        workspace.locator('[data-benchmark-filter="1"]').click()
        playwright.expect(dialog.locator(".sample-item")).to_have_count(2)
        dialog.locator("#clearBenchmarkSamplesBtn").click()
        dialog.locator("#sampleTaskFilter").select_option("TI2I")
        dialog.locator("#sampleSearch").fill("sample 1")
        dialog.locator("#allBenchmarkSamplesBtn").click()
        dialog.locator("#applyBenchmarkSamplesBtn").click()
        playwright.expect(workspace.locator("#sampleTotal")).to_have_text("61")
        workspace.locator('[data-benchmark-language="1"]').select_option("en")
        playwright.expect(workspace.locator("#runBtn")).to_be_enabled()
        assert ("second", "en") in requests
        workspace.locator('[data-benchmark-filter="0"]').click()
        playwright.expect(dialog.locator(".sample-item")).to_have_count(50)
        dialog.locator("#clearBenchmarkSamplesBtn").click()
        dialog.get_by_role("button", name="取消").click()
        playwright.expect(workspace.locator("#sampleTotal")).to_have_text("61")
        workspace.locator('[data-benchmark-check="0"]').uncheck()
        playwright.expect(workspace.locator("#sampleTotal")).to_have_text("1")
        workspace.locator('[data-benchmark-check="0"]').check()
        playwright.expect(workspace.locator("#sampleTotal")).to_have_text("61")
        page.reload(wait_until="networkidle")
        playwright.expect(workspace.locator("#sampleTotal")).to_have_text("61")
        playwright.expect(workspace.locator('[data-benchmark-language="1"]')).to_have_value("en")
        workspace.locator('[data-benchmark-filter="0"]').click()
        dialog.locator("#sampleSearch").fill("sample 61")
        playwright.expect(dialog.locator('[data-sample-id="61"]')).not_to_be_checked()
        dialog.get_by_role("button", name="取消").click()
        page.locator("#addModelBtn").click()
        second = page.locator("pipeline-workspace").nth(1)
        second.locator('[data-mode="benchmark"]').click()
        playwright.expect(second.locator("#sampleTotal")).to_have_text("61")
        playwright.expect(second.locator('[data-benchmark-check="1"]')).to_be_checked()
        playwright.expect(second.locator('[data-benchmark-language="1"]')).to_have_value("en")
        second.locator('[data-benchmark-check="1"]').uncheck()
        playwright.expect(second.locator("#sampleTotal")).to_have_text("60")
        playwright.expect(workspace.locator("#sampleTotal")).to_have_text("61")
        page.screenshot(path="logs/ui-benchmark-multiple.png", full_page=False)
        workspace.locator('[data-benchmark-filter="1"]').click()
        playwright.expect(dialog.locator(".sample-item")).to_have_count(2)
        page.screenshot(path="logs/ui-benchmark-filter.png", full_page=False)
        dialog.get_by_role("button", name="取消").click()
        workspace.locator("#runBtn").click()
        assert len(posts) == 1
        assert posts[0]["benchmarks"] == [
            {"name": "first", "language": "zh", "sample_ids": [str(index) for index in range(1, 61)]},
            {"name": "second", "language": "en", "sample_ids": ["1"]},
        ]
        playwright.expect(workspace.locator('[data-benchmark-filter="0"]')).to_be_disabled()
        playwright.expect(second.locator('[data-benchmark-filter="0"]')).to_be_enabled()
        assert not errors
        # Legacy browser sessions are migrated without losing their subset/language.
        legacy_page = browser.new_page()
        legacy_page.route("**/api/bootstrap", lambda route: route.fulfill(json=bootstrap))
        legacy_page.route("**/api/benchmarks/*/samples?*", samples)
        legacy_page.add_init_script("sessionStorage.setItem('visual-gen-eval-workspaces-v1', " + json.dumps(json.dumps([{"id": "legacy", "number": 1, "mode": "benchmark", "benchmark": "first", "language": "en", "selectedSamples": ["1"]}])) + ");")
        legacy_page.goto(os.environ["WORKBENCH_BROWSER_URL"], wait_until="networkidle")
        legacy = legacy_page.locator("pipeline-workspace").first
        playwright.expect(legacy.locator("#sampleTotal")).to_have_text("1")
        playwright.expect(legacy.locator('[data-benchmark-language="0"]')).to_have_value("en")
        browser.close()
