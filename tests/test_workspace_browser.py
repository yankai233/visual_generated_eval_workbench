"""Opt-in browser regression: WORKBENCH_BROWSER_URL=http://127.0.0.1:17861 pytest ..."""
from __future__ import annotations

import io
import os

import pytest
from PIL import Image


pytestmark = pytest.mark.skipif(
    not os.environ.get("WORKBENCH_BROWSER_URL"), reason="requires a running workbench and Playwright Chromium"
)


def test_workspace_inputs_jobs_and_controls_are_independent() -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    posts: list[dict] = []
    controls: list[tuple[str, str]] = []
    jobs: dict[str, dict] = {}
    bootstrap = {
        "pipelines": [{
            "name": "demo", "display_name": "Demo Pipeline", "group": "Test",
            "supported_tasks": ["T2I", "TI2I", "MultiReference"],
            "supported_devices": ["cpu", "cuda"], "max_reference_images": None,
            "default_num_inference_steps": 28,
            "configs": ["default"], "config_statuses": {"default": {"available": True, "issues": []}},
        }, {
            "name": "qwen_image_2_1_turbo", "display_name": "Qwen-Image-2.1-Turbo", "group": "Qwen",
            "supported_tasks": ["T2I", "TI2I", "MultiReference"],
            "supported_devices": ["cuda"], "max_reference_images": 10,
            "default_num_inference_steps": 8,
            "configs": ["default"], "config_statuses": {"default": {"available": True, "issues": []}},
        }],
        "gpus": [{"id": "0", "name": "Test 0"}, {"id": "1", "name": "Test 1"}],
        "cuda": {"available": True}, "benchmarks": [{"name": "test", "sample_count": 30, "task_types": ["T2I"]}], "jobs": [],
    }

    def create_job(route) -> None:
        posts.append(route.request.post_data_json)
        job_id = f"isolated-{len(posts)}"
        jobs[job_id] = {
            "id": job_id, "status": "running", "progress": 0, "total": 1,
            "completed": 0, "message": "running", "results": [], "errors": [], "active": {},
        }
        route.fulfill(json=jobs[job_id], status=201)

    def control_job(route) -> None:
        job_id, action = route.request.url.split("/")[-2:]
        controls.append((job_id, action))
        jobs[job_id]["status"] = {"pause": "paused", "resume": "running", "cancel": "cancelled"}[action]
        route.fulfill(json=jobs[job_id])

    buffer = io.BytesIO()
    Image.new("RGB", (400, 200), "#8da596").save(buffer, format="PNG")
    reference = {"name": "row-one.png", "mimeType": "image/png", "buffer": buffer.getvalue()}
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1700, "height": 1050})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.add_init_script("""
            window.streams = [];
            window.EventSource = class {
                constructor(url) { this.url = url; this.closed = false; window.streams.push(this); }
                close() { this.closed = true; }
            };
            window.emitJob = job => window.streams.forEach(stream => {
                if (!stream.closed && stream.url.includes('/' + job.id + '/'))
                    stream.onmessage({data: JSON.stringify(job)});
            });
        """)
        page.route("**/api/bootstrap", lambda route: route.fulfill(json={**bootstrap, "jobs": list(jobs.values())}))
        page.route("**/api/comparison/catalog", lambda route: route.fulfill(json={"timing_summary": [
            {"benchmark": "benchmark_a", "pipeline": "demo", "sample_count": 2, "average_generation_seconds": 2.5},
            {"benchmark": "benchmark_b", "pipeline": "demo", "sample_count": 2, "average_generation_seconds": 65.0},
            {"benchmark": "benchmark_a", "pipeline": "qwen_image_2_1_turbo", "sample_count": 2, "average_generation_seconds": 8.0},
        ]}))
        page.route("**/api/benchmarks/test/samples?*", lambda route: route.fulfill(json={"samples": [
            {"id": str(index), "task_type": "T2I", "prompt": f"Benchmark sample {index}", "input_image_urls": []}
            for index in range(1, 31)
        ]}))
        page.route("**/api/jobs", create_job)
        for action in ("pause", "resume", "cancel"):
            page.route(f"**/api/jobs/*/{action}", control_job)
        page.route("**/api/uploads", lambda route: route.fulfill(json={"path": ".uploads/mock/row-one.png"}))
        page.goto(os.environ["WORKBENCH_BROWSER_URL"], wait_until="networkidle")
        page.locator("#timingBtn").click()
        playwright.expect(page.locator("#timingDialog")).to_be_visible()
        playwright.expect(page.locator("#timingTableHead th")).to_have_count(3)
        playwright.expect(page.locator("#timingTableBody tr")).to_have_count(2)
        playwright.expect(page.locator("#timingTableBody")).to_contain_text("1分 05.0秒")
        playwright.expect(page.locator(".timing-missing")).to_have_text("-")
        page.locator("#timingDialog .icon-button").click()
        rows = page.locator("pipeline-workspace")
        first, second = rows.nth(0), rows.nth(1)
        outputs = page.locator("pipeline-output")
        first_output, second_output = outputs.nth(0), outputs.nth(1)

        first.locator('[data-action="pipeline"]').select_option("qwen_image_2_1_turbo")
        playwright.expect(first.locator("#stepsInput")).to_have_value("8")
        first.locator('[data-action="pipeline"]').select_option("demo")
        playwright.expect(first.locator("#stepsInput")).to_have_value("28")

        def assert_equal_panels(workspace) -> None:
            playwright.expect(workspace.locator(".workspace-grid > .panel").first).to_have_css("height", "360px")
            sizes = workspace.evaluate("""host => [...host.shadowRoot.querySelector('.workspace-grid').children].map(panel => {
                const box = panel.getBoundingClientRect(); return {height: box.height, top: box.top};
            })""")
            assert max(panel["height"] for panel in sizes) - min(panel["height"] for panel in sizes) < 1
            assert max(panel["top"] for panel in sizes) - min(panel["top"] for panel in sizes) < 1
            assert all(panel["height"] == 360 for panel in sizes)

        assert_equal_panels(first)
        first.locator('[data-mode="benchmark"]').click()
        first.locator('[data-benchmark-filter="0"]').click()
        playwright.expect(first.locator("#benchmarkSamplesDialog")).to_be_visible()
        playwright.expect(first.locator(".sample-item")).to_have_count(30)
        assert_equal_panels(first)
        input_body = first.locator("#inputFields")
        assert first.locator("#sampleList").evaluate("element => element.scrollHeight > element.clientHeight")
        first.locator("#sampleList").evaluate("element => { element.scrollTop = element.scrollHeight; }")
        assert first.locator("#sampleList").evaluate("element => element.scrollTop > 0")
        first.locator("#benchmarkSamplesDialog").get_by_role("button", name="取消").click()
        first.locator('[data-mode="custom"]').click()
        first.locator("#promptInput").fill("row one")
        first.locator("#taskType").select_option("TI2I")
        first.locator("#imageInput").set_input_files(reference)
        playwright.expect(first.locator("#referenceCount")).to_have_text("已上传 1/1 张参考图")
        assert_equal_panels(first)
        assert input_body.evaluate("element => element.scrollHeight > element.clientHeight")
        first.locator("#stepsInput").fill("24")  # auto-scroll reaches controls below the fold
        playwright.expect(first.locator("#stepsInput")).to_be_in_viewport()
        assert_equal_panels(first)
        page.locator("#addModelBtn").click()
        playwright.expect(rows).to_have_count(2)
        assert_equal_panels(second)
        playwright.expect(second.locator("#promptInput")).to_have_value("row one")
        playwright.expect(second.locator("#stepsInput")).to_have_value("24")
        playwright.expect(second.locator("#referenceCount")).to_have_text("已上传 1/1 张参考图")
        second.locator("#clearImagesBtn").click()
        playwright.expect(first.locator("#referenceCount")).to_have_text("已上传 1/1 张参考图")
        second.locator("#taskType").select_option("T2I")
        second.locator("#promptInput").fill("row two")
        playwright.expect(second.locator("#referenceCount")).to_have_text("已上传 0 张参考图")
        first.locator("#runBtn").click()
        playwright.expect(first_output.locator("#progressSection")).to_have_attribute("data-job-id", "isolated-1")
        playwright.expect(first_output.locator("#sampleEta")).to_have_text("估算中")
        assert "样本进度" not in first_output.locator("#progressSection").inner_text()
        assert "去噪进度" not in first_output.locator("#progressSection").inner_text()
        assert first_output.locator(".step-badge, #jobId, #jobPipelineLabel, #jobMessage").count() == 0
        playwright.expect(second.locator("#promptInput")).to_be_enabled()
        playwright.expect(second.locator("#runBtn")).to_be_enabled()
        second.locator("[data-gpu='0']").uncheck()
        second.locator("[data-gpu='1']").check()
        second.locator("#runBtn").click()
        playwright.expect(second_output.locator("#progressSection")).to_have_attribute("data-job-id", "isolated-2")
        assert posts[0]["custom_samples"][0]["prompt"] == "row one"
        assert posts[0]["custom_samples"][0]["num_inference_steps"] == 24
        assert posts[0]["custom_samples"][0]["input_images"] == [".uploads/mock/row-one.png"]
        assert posts[1]["custom_samples"][0]["prompt"] == "row two"
        assert posts[1]["custom_samples"][0]["input_images"] == []
        assert posts[0]["pipelines"][0]["gpu_ids"] == ["0"]
        assert posts[1]["pipelines"][0]["gpu_ids"] == ["1"]
        jobs["isolated-1"].update(eta_seconds=None, eta_phase="loading", active={"1": {"step": 0, "total_steps": 10, "progress": 0, "eta_seconds": None, "eta_phase": "loading"}})
        page.evaluate("job => window.emitJob(job)", jobs["isolated-1"])
        playwright.expect(first_output.locator("#sampleEta")).to_have_text("加载模型中")
        playwright.expect(first_output.locator("#sampleAverage")).to_have_text("平均每样本 —")
        playwright.expect(first_output.locator("[data-denoise-eta]")).to_have_text("加载模型中")
        for number, step in ((1, 2), (2, 8)):
            job = jobs[f"isolated-{number}"]
            job["eta_seconds"] = 125 if number == 1 else 65
            job["eta_phase"] = "running"
            job["average_sample_seconds"] = 15 if number == 1 else 61
            job["active"] = {str(number): {"step": step, "total_steps": 10, "progress": step / 10, "eta_seconds": 8 if number == 1 else 2, "eta_phase": "denoising"}}
            page.evaluate("job => window.emitJob(job)", job)
        playwright.expect(first_output.locator("#gpuProgressList")).to_contain_text("2 / 10")
        playwright.expect(second_output.locator("#gpuProgressList")).to_contain_text("8 / 10")
        playwright.expect(first_output.locator("#sampleEta")).to_have_text("预计剩余 2分5秒")
        playwright.expect(second_output.locator("#sampleEta")).to_have_text("预计剩余 1分5秒")
        playwright.expect(first_output.locator("#sampleAverage")).to_have_text("平均每样本 15秒")
        playwright.expect(second_output.locator("#sampleAverage")).to_have_text("平均每样本 1分1秒")
        playwright.expect(first_output.locator("[data-denoise-eta]")).to_have_text("预计剩余 8秒")
        playwright.expect(second_output.locator("[data-denoise-eta]")).to_have_text("预计剩余 2秒")
        page.locator("#controlModel").select_option(first.get_attribute("data-workspace-id"))
        page.locator("#pauseAllBtn").click()
        playwright.expect(first_output.locator("#jobStatus")).to_have_text("paused")
        playwright.expect(first_output.locator("#sampleEta")).to_contain_text("已暂停")
        playwright.expect(first_output.locator("[data-denoise-eta]")).to_contain_text("已暂停")
        playwright.expect(second_output.locator("#jobStatus")).to_have_text("running")
        page.locator("#resumeAllBtn").click()
        playwright.expect(first_output.locator("#jobStatus")).to_have_text("running")
        page.route("**/test-generated.png", lambda route: route.fulfill(body=buffer.getvalue(), content_type="image/png"))
        jobs["isolated-1"]["results"] = [{"unit_id": "demo:default:one", "output_dir": "test", "duration_seconds": 1, "image_urls": ["/test-generated.png"]}]
        page.evaluate("job => window.emitJob(job)", jobs["isolated-1"])
        first_results = page.locator("model-results").nth(0)
        playwright.expect(first_results.locator("#resultsSection")).to_be_visible()
        playwright.expect(first_results.locator("#modelResultsTitle")).to_contain_text("Model 01")
        jobs["isolated-2"]["results"] = [{"unit_id": "demo:sample-b:1", "sample_id": "sample-b", "benchmark": "users", "pipeline": "demo", "config_name": "default", "output_dir": "second", "duration_seconds": 1, "image_urls": ["/test-generated.png?model=2"]}]
        page.route("**/test-generated.png?model=2", lambda route: route.fulfill(body=buffer.getvalue(), content_type="image/png"))
        page.evaluate("job => window.emitJob(job)", jobs["isolated-2"])
        second_results = page.locator("model-results").nth(1)
        playwright.expect(second_results.locator("#resultsSection")).to_be_visible()
        playwright.expect(second_results.locator("#modelResultsTitle")).to_contain_text("Model 02")
        playwright.expect(second_results.locator(".result-card strong")).to_have_text("users / sample-b")
        playwright.expect(first_results.locator(".result-card")).to_have_count(1)
        playwright.expect(second_results.locator(".result-card")).to_have_count(1)
        page.locator("#addModelBtn").click()
        playwright.expect(rows).to_have_count(3)
        assert_equal_panels(rows.nth(2))
        assert rows.nth(2).bounding_box()["y"] < first_output.bounding_box()["y"]
        assert page.locator("#workspaces #resultsSection").count() == 0
        assert page.locator("#modelResults #resultsSection").count() == 3
        assert page.locator("#sharedProgress").count() == 1
        assert page.locator("#sharedResults").count() == 1
        playwright.expect(rows.nth(2).locator("#promptInput")).to_have_value("row two")
        page.screenshot(path="logs/ui-workspace-aligned-layout.png", full_page=True)
        rows.nth(2).locator("[data-action=remove]").click()
        playwright.expect(outputs).to_have_count(2)
        page.reload(wait_until="networkidle")
        playwright.expect(rows).to_have_count(2)
        playwright.expect(first_output.locator("#progressSection")).to_have_attribute("data-job-id", "isolated-1")
        playwright.expect(second_output.locator("#progressSection")).to_have_attribute("data-job-id", "isolated-2")
        playwright.expect(first_results.locator("#resultsSection")).to_be_visible()
        assert page.evaluate("window.streams.filter(s => !s.closed).length") == 2
        page.locator("#controlModel").select_option("all")
        page.locator("#pauseAllBtn").click()
        playwright.expect(first_output.locator("#jobStatus")).to_have_text("paused")
        playwright.expect(second_output.locator("#jobStatus")).to_have_text("paused")
        page.locator("#resumeAllBtn").click()
        playwright.expect(first_output.locator("#jobStatus")).to_have_text("running")
        playwright.expect(second_output.locator("#jobStatus")).to_have_text("running")
        page.locator("#controlModel").select_option(first.get_attribute("data-workspace-id"))
        page.locator("#cancelAllBtn").click()
        playwright.expect(first_output.locator("#jobStatus")).to_have_text("cancelled")
        playwright.expect(first_output.locator("#sampleEta")).to_have_text("已取消")
        playwright.expect(second_output.locator("#jobStatus")).to_have_text("running")
        assert controls == [("isolated-1", "pause"), ("isolated-1", "resume"), ("isolated-1", "pause"), ("isolated-2", "pause"), ("isolated-1", "resume"), ("isolated-2", "resume"), ("isolated-1", "cancel")]
        first.locator("[data-action=remove]").click()
        playwright.expect(rows).to_have_count(1)
        playwright.expect(outputs).to_have_count(1)
        playwright.expect(outputs.first.locator("#progressSection")).to_have_attribute("data-job-id", "isolated-2")
        assert rows.first.get_attribute("data-workspace-id") == outputs.first.get_attribute("data-workspace-id")
        for width in (1200, 980):
            page.set_viewport_size({"width": width, "height": 900})
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert rows.first.locator("#inputFields").evaluate("element => element.scrollHeight > element.clientHeight")
        assert not errors
        browser.close()
