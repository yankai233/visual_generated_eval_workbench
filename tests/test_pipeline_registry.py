from src.pipelines.registry import discover_pipelines


def test_demo_pipeline_is_discoverable() -> None:
    pipelines = discover_pipelines()
    assert "demo" in pipelines


def test_built_in_model_pipelines_are_discoverable() -> None:
    pipelines = discover_pipelines()
    assert {
        "qwen_image_2_1",
        "flux_3_klein",
        "flux_3_klein_base",
        "flux_1_dev",
        "qwen_image_edit_2511",
        "qwen_image_2_1_turbo",
        "uno",
    }.issubset(pipelines)
    assert "T2I" in pipelines["demo"].supported_tasks

