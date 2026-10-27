from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.skipif(not os.environ.get("WORKBENCH_BROWSER_URL"), reason="requires running workbench and Playwright")


def test_gpu_monitor_updates_stops_when_closed_and_handles_failure():
    playwright = pytest.importorskip("playwright.sync_api")
    pipeline = {"name": "demo", "display_name": "Demo", "group": "Test", "supported_tasks": ["T2I"], "supported_devices": ["cpu"], "max_reference_images": 0, "configs": ["default"], "config_statuses": {"default": {"available": True, "issues": []}}}
    bootstrap = {"pipelines": [pipeline], "gpus": [], "cuda": {"available": False}, "jobs": [], "benchmarks": []}
    calls = []
    fail = [False]
    def usage(route):
        calls.append(True)
        if fail[0]:
            route.fulfill(status=503, json={"detail": "GPU 读取失败"})
            return
        used = 32768 if len(calls) == 1 else 65536
        route.fulfill(json={"available": True, "error": None, "updated_at": "2026-10-19T00:00:00+00:00", "gpus": [
            {"id": str(index), "name": "NVIDIA A100", "driver_version": "570.86.10", "memory_total_mb": 81920, "memory_used_mb": used,
             "memory_percent": used / 81920 * 100, "utilization_percent": 73, "temperature_c": 49, "power_w": 132, "power_limit_w": 300}
            for index in range(8)
        ]})
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1500, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/api/bootstrap", lambda route: route.fulfill(json=bootstrap))
        page.route("**/api/system/gpus", usage)
        page.goto(os.environ["WORKBENCH_BROWSER_URL"], wait_until="networkidle")
        playwright.expect(page.locator("#serviceStatusText")).to_have_text("服务可用")
        assert "CUDA" not in page.locator("#systemStatus").inner_text()
        assert calls == []
        page.locator("#gpuUsageBtn").click()
        dialog = page.locator("#gpuMonitorDialog")
        playwright.expect(dialog).to_be_visible()
        playwright.expect(dialog.locator("tbody tr")).to_have_count(8)
        first = dialog.locator('[data-gpu-id="0"]')
        playwright.expect(first).to_contain_text("32.00 / 80.00 GiB")
        playwright.expect(first.get_by_role("progressbar", name="GPU 0 显存")).to_have_attribute("aria-valuenow", "40.0")
        playwright.expect(first).to_contain_text("64.00 / 80.00 GiB", timeout=6000)
        playwright.expect(first.get_by_role("progressbar", name="GPU 0 显存")).to_have_attribute("aria-valuenow", "80.0")
        playwright.expect(first).to_contain_text("73.0%")
        page.screenshot(path="logs/ui-gpu-monitor.png", full_page=False)
        page.evaluate("Object.defineProperty(document, 'hidden', {configurable: true, get: () => true}); document.dispatchEvent(new Event('visibilitychange'));")
        hidden_count = len(calls)
        page.wait_for_timeout(2300)
        assert len(calls) == hidden_count
        with page.expect_response("**/api/system/gpus"):
            page.evaluate("Object.defineProperty(document, 'hidden', {configurable: true, get: () => false}); document.dispatchEvent(new Event('visibilitychange'));")
        dialog.locator("#closeGpuMonitorBtn").click()
        playwright.expect(dialog).not_to_be_visible()
        count = len(calls)
        page.wait_for_timeout(2300)
        assert len(calls) == count
        page.locator("#gpuUsageBtn").click()
        playwright.expect(dialog.locator("tbody tr")).to_have_count(8)
        page.keyboard.press("Escape")
        playwright.expect(dialog).not_to_be_visible()
        fail[0] = True
        page.locator("#gpuUsageBtn").click()
        playwright.expect(dialog.locator("#gpuMonitorMessage")).to_have_text("GPU 读取失败")
        playwright.expect(dialog.locator("#gpuMonitorTable")).not_to_be_visible()
        fail[0] = False
        playwright.expect(dialog.locator("tbody tr")).to_have_count(8, timeout=6000)
        page.set_viewport_size({"width": 390, "height": 844})
        assert dialog.bounding_box()["width"] <= 390
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        dialog.locator("#closeGpuMonitorBtn").click()
        assert not errors
        browser.close()


def test_service_failure_is_not_reported_as_available():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page()
        page.route("**/api/bootstrap", lambda route: route.fulfill(status=503, json={"detail": "unavailable"}))
        page.goto(os.environ["WORKBENCH_BROWSER_URL"], wait_until="networkidle")
        playwright.expect(page.locator("#serviceStatusText")).to_have_text("服务不可用")
        playwright.expect(page.locator("#systemStatus")).to_have_class("system-status offline")
        browser.close()
