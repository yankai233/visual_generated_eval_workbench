from fastapi.testclient import TestClient
from types import SimpleNamespace

from src.web.server import create_app
from src.utils.paths import CONFIG_ROOT


def test_bootstrap_and_native_benchmark_api() -> None:
    with TestClient(create_app()) as client:
        bootstrap = client.get("/api/bootstrap")
        assert bootstrap.status_code == 200
        benchmark_counts = {
            item["name"]: item["sample_count"]
            for item in bootstrap.json()["benchmarks"]
        }
        assert benchmark_counts["t2i"] == 5
        assert benchmark_counts["ti2i"] == 4
        assert benchmark_counts["multi_reference"] == 5

        samples = client.get(
            "/api/benchmarks/multi_reference/samples", params={"language": "en"}
        )
        assert samples.status_code == 200
        first = samples.json()["samples"][0]
        assert first["task_type"] == "MultiReference"
        assert len(first["input_image_urls"]) >= 2
        assert first["prompt"].startswith("Create")


def test_index_and_static_assets_are_served() -> None:
    with TestClient(create_app()) as client:
        homepage = client.get("/").text
        assert "图像生成可视化评估工作台" in homepage
        assert 'href="/compare"' in homepage
        assert 'id="timingBtn"' in homepage
        assert "data-auxiliary-trigger" in homepage
        assert client.get("/tools").status_code == 200
        assert client.get("/static/auxiliary-window.js").status_code == 200
        assert client.get("/static/auxiliary-window.css").status_code == 200
        assert client.get("/static/tools.js").status_code == 200
        assert client.get("/static/tools.css").status_code == 200
        assert client.get("/compare").status_code == 200
        assert client.get("/static/compare.js").status_code == 200
        assert client.get("/static/compare.css").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/styles.css").status_code == 200
        assert client.get("/static/workbench.css").status_code == 200


def test_temporary_yaml_is_validated_and_used_without_changing_saved_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.web.server.GEN_OUTPUT_ROOT", tmp_path / "gen")
    path = CONFIG_ROOT / "demo" / "default" / "config.yaml"
    original = path.read_bytes()
    app = create_app()
    submitted = {}

    def capture_submit(**kwargs):
        submitted.update(kwargs)
        return SimpleNamespace(snapshot=lambda: {"id": "temporary-config-test", "results": []})

    app.state.job_manager.submit = capture_submit
    with TestClient(app) as client:
        invalid = client.post("/api/pipelines/demo/validate-config", json={"content": "- invalid"})
        assert invalid.status_code == 400
        valid = client.post("/api/pipelines/demo/validate-config", json={"content": "step_delay: 0.125"})
        assert valid.status_code == 200
        response = client.post("/api/jobs", json={
            "input_mode": "custom",
            "custom_samples": [{"id": "sample", "prompt": "a tree"}],
            "pipelines": [{"pipeline": "demo", "config_name": "default", "gpu_ids": ["cpu"], "config_yaml": "step_delay: 0.125"}],
        })
        assert response.status_code == 201
        assert submitted["configs"][("demo", "default")]["step_delay"] == 0.125
        assert path.read_bytes() == original

