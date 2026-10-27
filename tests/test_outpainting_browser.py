from __future__ import annotations

import io
import os

import pytest
from PIL import Image


pytestmark = pytest.mark.skipif(not os.environ.get("WORKBENCH_BROWSER_URL"), reason="requires running workbench and Playwright")


def test_outpainting_comparison_layout_and_frame() -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    image = io.BytesIO()
    Image.new("RGB", (400, 300), "#8aaa94").save(image, format="PNG")
    runs = [
        {"id": "blue", "label": "BlueOut", "sample_count": 2},
        {"id": "pro", "label": "ProOut", "sample_count": 2},
    ]
    rows = [{
        "sample_id": "0", "prompt": "a painting with a wider forest", "width": 400, "height": 300,
        "input_box": [100, 75, 300, 225], "input_url": "/mock/input.png", "target_url": "/mock/target.png",
        "cells": [["/mock/blue.png"], ["/mock/pro-1.png", "/mock/pro-2.png"]],
    }]

    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1600, "height": 950})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/api/outpainting-comparison/catalog*", lambda route: route.fulfill(json={"runs": runs}))
        page.route("**/api/outpainting-comparison/view*", lambda route: route.fulfill(json={"rows": rows, "runs": runs, "total": 1, "page": 1, "page_size": 12}))
        page.route("**/mock/*.png", lambda route: route.fulfill(body=image.getvalue(), content_type="image/png"))
        page.goto(os.environ["WORKBENCH_BROWSER_URL"].rstrip("/") + "/outpainting-compare", wait_until="networkidle")
        playwright.expect(page.locator("[data-auxiliary-trigger]")).to_be_visible()
        page.locator("[data-auxiliary-trigger]").click()
        playwright.expect(page.locator("#auxiliaryWindow")).to_be_visible()
        playwright.expect(page.frame_locator("#auxiliaryWindowFrame").locator("#translateTab")).to_be_visible()
        page.locator("#closeAuxiliaryWindow").click()
        playwright.expect(page.locator(".outpaint-table thead th")).to_have_count(3)
        playwright.expect(page.locator(".outpaint-image")).to_have_count(3)
        playwright.expect(page.locator(".known-frame")).to_have_count(4)
        playwright.expect(page.locator(".source-pair figcaption")).to_have_text(["输入图", "GT"])
        first = page.locator(".outpaint-image").first
        assert first.bounding_box()["width"] <= 300
        first.click()
        playwright.expect(page.locator("#imageDialog")).to_be_visible()
        playwright.expect(page.locator("#largeImageStage .known-frame")).to_have_count(1)
        page.locator("#closeImageBtn").click()
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert page.locator("#matrixViewport").evaluate("element => element.scrollWidth > element.clientWidth")
        assert not errors
        browser.close()
