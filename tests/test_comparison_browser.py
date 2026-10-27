from __future__ import annotations

import io
import os
from urllib.parse import parse_qs, urlparse

import pytest
from PIL import Image

from src.data.result_catalog import ResultCatalog
from src.data.schemas import SampleModel
from test_comparison import add_result


pytestmark = pytest.mark.skipif(not os.environ.get("WORKBENCH_BROWSER_URL"), reason="requires running workbench and Playwright")


def test_comparison_matrix_filters_versions_and_manual_groups(tmp_path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    root = tmp_path / "gen"
    sample = SampleModel(id="1", prompt="first case")
    add_result(root, "a", "20261019T100000", sample, "benchmark:test")
    add_result(root, "a", "20261019T110000", sample, "benchmark:test", image_count=2)
    add_result(root, "b", "20261019T100000", sample, "benchmark:test")
    add_result(root, "a", "20261019T100000", sample.model_copy(update={"id": "2", "prompt": "second case"}), "benchmark:test")
    add_result(root, "a", "20261019T120000", sample)
    add_result(root, "b", "20261019T120000", sample)
    add_result(root, "b", "20261019T130000", sample.model_copy(update={"id": "other", "prompt": "other input"}))
    catalog = ResultCatalog(root, tmp_path / "benchmarks")

    def handle_api(route):
        parsed = urlparse(route.request.url)
        operation = parsed.path.rsplit("/", 1)[-1]
        query = parse_qs(parsed.query)
        payload = route.request.post_data_json if route.request.method == "POST" else {}
        if operation == "catalog":
            response = {**catalog.catalog(), "pipeline_labels": {"a": "Method A", "b": "Method B"}}
            response["timing_summary"].append({
                "source": "benchmark:other", "benchmark": "other", "pipeline": "a",
                "sample_count": 1, "average_generation_seconds": 4.0,
            })
        elif operation == "view":
            response = catalog.view(payload)
        elif operation == "samples":
            response = {"samples": catalog.samples(query["source"][0])}
        elif operation == "records":
            response = {"records": catalog.records(query["source"][0])}
        elif operation == "group":
            catalog.set_group(payload["result_ids"], payload["name"])
            response = {"ok": True}
        else:
            raise AssertionError(operation)
        route.fulfill(json=response)

    buffer = io.BytesIO()
    Image.new("RGB", (100, 80), "#80a08a").save(buffer, format="PNG")
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1550, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/api/comparison/**", handle_api)
        page.route("**/files/output/**", lambda route: route.fulfill(body=buffer.getvalue(), content_type="image/png"))
        page.goto(os.environ["WORKBENCH_BROWSER_URL"].rstrip("/") + "/compare", wait_until="networkidle")
        playwright.expect(page.locator(".comparison-table thead th")).to_have_count(3)
        playwright.expect(page.locator(".comparison-table tbody tr")).to_have_count(2)
        playwright.expect(page.locator(".cell-missing")).to_have_count(1)
        playwright.expect(page.locator("[data-cell-version]")).to_have_count(1)
        playwright.expect(page.locator("[data-cell-image]")).to_have_count(1)
        page.locator(".cell-details").first.locator("summary").click()
        playwright.expect(page.locator(".cell-details").first).to_contain_text("生成耗时 2.50 秒")
        preview = page.locator(".cell-image-button img").first
        assert preview.bounding_box()["width"] <= 260
        assert preview.bounding_box()["height"] <= 260
        page.set_viewport_size({"width": 2200, "height": 1200})
        assert preview.bounding_box()["width"] <= 260  # no stretching on wide screens
        assert page.locator(".comparison-table").bounding_box()["width"] == 780
        page.locator("[data-remove-column]").last.click()
        playwright.expect(page.locator(".comparison-table thead th")).to_have_count(2)
        assert preview.bounding_box()["width"] <= 260  # one method stays compact too
        page.locator("#addMethodBtn").click()
        playwright.expect(page.locator(".comparison-table thead th")).to_have_count(3)
        page.set_viewport_size({"width": 1550, "height": 1000})
        page.locator("[data-cell-image]").select_option("1")
        page.locator(".cell-image-button").first.click()
        playwright.expect(page.locator("#imageDialog")).to_be_visible()
        page.locator("#closeImageBtn").click()
        page.locator("[data-cell-version]").select_option("1")
        playwright.expect(page.locator("[data-cell-image]")).to_have_count(0)
        page.locator("#onlyCommonInput").check()
        playwright.expect(page.locator(".comparison-table tbody tr")).to_have_count(1)
        page.locator("#onlyCommonInput").uncheck()
        page.locator("#samplesBtn").click()
        page.locator("#noSamplesBtn").click()
        page.locator("#applySamplesBtn").click()
        playwright.expect(page.locator("#rowCount")).to_have_text("0 个样本")
        page.locator("#samplesBtn").click()
        page.locator("#allSamplesBtn").click()
        page.locator("#applySamplesBtn").click()
        playwright.expect(page.locator(".comparison-table tbody tr")).to_have_count(2)
        page.screenshot(path="logs/ui-comparison-matrix.png", full_page=True)
        page.locator("#sourceSelect").select_option("users")
        playwright.expect(page.locator(".comparison-table tbody tr")).to_have_count(2)
        page.locator("#groupsBtn").click()
        page.locator("#groupNameInput").fill("对比组")
        for choice in page.locator("[data-result-key]").all():
            choice.check()
        page.locator("#applyGroupsBtn").click()
        playwright.expect(page.locator(".comparison-table tbody tr")).to_have_count(1)
        playwright.expect(page.locator(".sample-column")).to_contain_text("对比组")
        playwright.expect(page.locator(".warning-tag")).to_contain_text("输入不同")
        page.reload(wait_until="networkidle")
        page.locator("#sourceSelect").select_option("users")
        playwright.expect(page.locator(".comparison-table tbody tr")).to_have_count(1)
        page.locator("#groupsBtn").click()
        playwright.expect(page.locator("#groupsDialog")).to_be_visible()
        playwright.expect(page.locator("[data-result-key]")).to_have_count(3)
        for choice in page.locator("[data-result-key]").all():
            choice.check()
        page.locator("#resetGroupsBtn").click()
        playwright.expect(page.locator(".comparison-table tbody tr")).to_have_count(2)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert preview.bounding_box()["width"] <= 220
        assert page.locator("#matrixViewport").evaluate("element => element.scrollWidth > element.clientWidth")
        assert not errors
        browser.close()
