from __future__ import annotations

import os
from urllib.parse import parse_qs, urlparse

import pytest

from src.tools.markdown_store import DocumentConflict, MarkdownStore, render_markdown
from src.tools.translation import LANGUAGES


pytestmark = pytest.mark.skipif(not os.environ.get("WORKBENCH_BROWSER_URL"), reason="requires running workbench and Playwright")


def test_tools_live_translation_markdown_preview_save_conflicts_and_drafts(tmp_path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    store = MarkdownStore(tmp_path / "markdown")
    mode = {"value": "web"}
    translations = []

    def handle_api(route):
        parsed = urlparse(route.request.url)
        query = parse_qs(parsed.query)
        payload = route.request.post_data_json if route.request.method == "POST" else {}
        status = 200
        if parsed.path.endswith("/status"):
            response = {"translation": {"provider": "Google", "mode": mode["value"], "languages": LANGUAGES}, "markdown_directory": "output/tools/markdown"}
        elif parsed.path.endswith("/files"):
            response = {"files": store.list_files()}
        elif parsed.path.endswith("/preview"):
            response = {"html": render_markdown(payload["content"])}
        elif parsed.path.endswith("/translate"):
            translations.append(payload)
            response = {"text": f"译：{payload['text']}", "detected_source": "en"}
        elif route.request.method == "POST":
            try:
                response = store.save(payload["name"], payload["content"], payload["revision"])
            except DocumentConflict as error:
                response = {"detail": str(error)}
                status = 409
        else:
            response = store.read(query["name"][0])
        route.fulfill(json=response, status=status)

    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1550, "height": 1000}, accept_downloads=True)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("dialog", lambda dialog: dialog.accept())
        page.route("**/api/tools/**", handle_api)
        page.goto(os.environ["WORKBENCH_BROWSER_URL"].rstrip("/") + "/tools", wait_until="networkidle")
        playwright.expect(page.locator("#translationMode")).to_have_text("Google · 实时")
        playwright.expect(page.locator("#swapLanguagesBtn")).to_be_disabled()
        page.locator("#translationInput").fill("猫 & <test>")
        playwright.expect(page.locator("#translationOutput")).to_have_value("译：猫 & <test>")
        assert translations[-1]["text"] == "猫 & <test>"
        playwright.expect(page.locator("#googleTranslateLink")).to_have_count(0)
        page.locator("#sourceLanguage").select_option("en")
        page.locator("#targetLanguage").select_option("zh-CN")
        page.locator("#swapLanguagesBtn").click()
        playwright.expect(page.locator("#sourceLanguage")).to_have_value("zh-CN")
        playwright.expect(page.locator("#targetLanguage")).to_have_value("en")
        page.locator("#markdownTab").click()
        page.locator("#markdownImport").set_input_files({"name": "note.md", "mimeType": "text/markdown", "buffer": b'# Test note\n\n**bold**\n\n<script>parent.hacked = 1</script>\n\n|A|B|\n|---|---|\n|1|2|'})
        preview = page.frame_locator("#markdownPreview")
        playwright.expect(preview.locator("h1")).to_have_text("Test note")
        playwright.expect(preview.locator("table")).to_have_count(1)
        assert page.evaluate("window.hacked === undefined")
        page.locator("#markdownFilename").fill("note")
        page.locator("#saveMarkdownBtn").click()
        playwright.expect(page.locator("#markdownStatus")).to_have_text("已保存")
        playwright.expect(page.locator("#markdownFilename")).to_have_value("note.md")
        assert store.read("note.md")["content"].startswith("# Test note")
        page.locator("#markdownEditor").fill("# Unsaved draft")
        page.reload(wait_until="networkidle")
        playwright.expect(page.locator("#markdownEditor")).to_have_value("# Unsaved draft")
        playwright.expect(page.locator("#markdownStatus")).to_have_text("未保存")
        current = store.read("note.md")
        store.save("note.md", "# Other page", current["revision"])
        page.locator("#saveMarkdownBtn").click()
        playwright.expect(page.locator("#markdownError")).to_contain_text("其他页面修改")
        assert store.read("note.md")["content"] == "# Other page"
        page.locator("#markdownFileSelect").select_option("note.md")
        playwright.expect(page.locator("#markdownEditor")).to_have_value("# Other page")
        page.locator("#markdownEditor").fill("# Saved revision")
        page.locator("#saveMarkdownBtn").click()
        playwright.expect(page.locator("#markdownStatus")).to_have_text("已保存")
        with page.expect_download() as download_info:
            page.locator("#downloadMarkdownBtn").click()
        assert download_info.value.suggested_filename == "note.md"
        page.screenshot(path="logs/ui-tools-markdown.png", full_page=True)
        mode["value"] = "api"
        page.goto(os.environ["WORKBENCH_BROWSER_URL"].rstrip("/") + "/tools#translate", wait_until="networkidle")
        page.reload(wait_until="networkidle")
        playwright.expect(page.locator("#translationMode")).to_have_text("Google Cloud · 实时")
        page.locator("#translationInput").fill("Hello")
        page.locator("#targetLanguage").select_option("zh-CN")
        playwright.expect(page.locator("#translationOutput")).to_have_value("译：Hello")
        assert translations[-1]["text"] == "Hello"
        page.locator("#translationInput").fill("New input")
        playwright.expect(page.locator("#translationOutput")).to_have_value("译：New input")
        assert translations[-1]["text"] == "New input"
        page.locator("#clearTranslationBtn").click()
        playwright.expect(page.locator("#translationOutput")).to_have_value("")
        playwright.expect(page.locator("#translateBtn")).to_be_hidden()
        page.set_viewport_size({"width": 390, "height": 844})
        for tab in ("#translateTab", "#markdownTab"):
            page.locator(tab).click()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors
        browser.close()


def test_auxiliary_tools_open_in_resizable_collapsible_window() -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1500, "height": 950})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(os.environ["WORKBENCH_BROWSER_URL"].rstrip("/") + "/", wait_until="domcontentloaded")

        page.locator("[data-auxiliary-trigger]").click()
        panel = page.locator("#auxiliaryWindow")
        playwright.expect(panel).to_be_visible()
        assert page.evaluate("getComputedStyle(document.querySelector('#auxiliaryWindow')).resize") == "both"

        tools = page.frame_locator("#auxiliaryWindowFrame")
        playwright.expect(tools.locator("#translateTab")).to_be_visible()
        playwright.expect(tools.locator(".tools-header")).to_be_hidden()
        tools.locator("#translationInput").fill("保留这段文字")

        before_drag = panel.bounding_box()
        heading = page.locator(".auxiliary-window-heading")
        box = heading.bounding_box()
        page.mouse.move(box["x"] + 80, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.mouse.move(box["x"] - 90, box["y"] + 90, steps=5)
        page.mouse.up()
        after_drag = panel.bounding_box()
        assert after_drag["x"] < before_drag["x"] - 100
        assert after_drag["y"] > before_drag["y"] + 40

        stable_size = {"width": after_drag["width"], "height": after_drag["height"]}
        tools.locator("#markdownTab").click()
        page.wait_for_timeout(150)
        markdown_size = panel.bounding_box()
        assert abs(markdown_size["width"] - stable_size["width"]) <= 1
        assert abs(markdown_size["height"] - stable_size["height"]) <= 1
        tools.locator("#translateTab").click()

        page.locator("#auxiliaryWindow").evaluate("node => { node.style.width = '680px'; node.style.height = '560px'; }")
        page.wait_for_timeout(100)
        page.locator("#collapseAuxiliaryWindow").click()
        assert panel.evaluate("node => node.classList.contains('collapsed')")
        playwright.expect(page.locator("#auxiliaryWindowFrame")).to_be_hidden()
        page.locator("#collapseAuxiliaryWindow").click()
        playwright.expect(tools.locator("#translationInput")).to_have_value("保留这段文字")

        page.locator("#closeAuxiliaryWindow").click()
        playwright.expect(panel).to_be_hidden()
        page.locator("[data-auxiliary-trigger]").click()
        playwright.expect(tools.locator("#translationInput")).to_have_value("保留这段文字")
        size = panel.evaluate("node => ({ width: node.getBoundingClientRect().width, height: node.getBoundingClientRect().height })")
        assert abs(size["width"] - 680) <= 2
        assert abs(size["height"] - 560) <= 2

        tools.locator("#markdownTab").click()
        tools.locator("#markdownFilename").fill("跨页面草稿")
        tools.locator("#markdownEditor").fill("# 页面切换后仍然存在")
        saved_box = panel.bounding_box()
        page.goto(os.environ["WORKBENCH_BROWSER_URL"].rstrip("/") + "/outpainting-compare", wait_until="domcontentloaded")
        playwright.expect(page.locator("[data-auxiliary-trigger]")).to_be_visible()
        panel = page.locator("#auxiliaryWindow")
        playwright.expect(panel).to_be_visible()
        tools = page.frame_locator("#auxiliaryWindowFrame")
        playwright.expect(tools.locator("#markdownTab")).to_have_attribute("aria-selected", "true")
        playwright.expect(tools.locator("#markdownFilename")).to_have_value("跨页面草稿")
        playwright.expect(tools.locator("#markdownEditor")).to_have_value("# 页面切换后仍然存在")
        restored_box = panel.bounding_box()
        assert abs(restored_box["width"] - saved_box["width"]) <= 2
        assert abs(restored_box["height"] - saved_box["height"]) <= 2

        page.locator("#collapseAuxiliaryWindow").click()
        page.goto(os.environ["WORKBENCH_BROWSER_URL"].rstrip("/") + "/compare", wait_until="domcontentloaded")
        panel = page.locator("#auxiliaryWindow")
        playwright.expect(panel).to_be_visible()
        assert panel.evaluate("node => node.classList.contains('collapsed')")
        page.locator("#closeAuxiliaryWindow").click()
        page.goto(os.environ["WORKBENCH_BROWSER_URL"].rstrip("/") + "/", wait_until="domcontentloaded")
        playwright.expect(page.locator("#auxiliaryWindow")).to_be_hidden()
        page.screenshot(path="logs/ui-auxiliary-window.png")
        assert not errors
        browser.close()
