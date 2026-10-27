from __future__ import annotations

from pathlib import Path

from src.pipelines.base import GenerationRequest
from src.pipelines.uno.pipeline import UNOWorkbenchPipeline


def request(task_type: str, count: int) -> GenerationRequest:
    return GenerationRequest(
        id="uno-test",
        task_type=task_type,
        prompt="test",
        input_images=[Path(f"ref-{index}.png") for index in range(count)],
    )


def test_uno_metadata_exposes_supported_tasks_and_reference_limit() -> None:
    metadata = UNOWorkbenchPipeline.metadata()
    assert metadata["supported_tasks"] == ["T2I", "TI2I", "MultiReference"]
    assert metadata["max_reference_images"] == 4
    assert metadata["supports_cpu_offload"] is True


def test_uno_request_validation() -> None:
    UNOWorkbenchPipeline._validate_request(request("T2I", 0))
    UNOWorkbenchPipeline._validate_request(request("TI2I", 1))
    UNOWorkbenchPipeline._validate_request(request("MultiReference", 4))


def test_uno_request_validation_rejects_invalid_reference_counts() -> None:
    for task_type, count in (("T2I", 1), ("TI2I", 0), ("MultiReference", 0), ("MultiReference", 5)):
        try:
            UNOWorkbenchPipeline._validate_request(request(task_type, count))
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected {task_type}/{count} to be rejected")
