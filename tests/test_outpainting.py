from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from src.data.outpainting_catalog import OutpaintingResultCatalog
from src.data.schemas import SampleModel
from src.web.outpainting_comparison import outpainting_comparison_router


def add_external_result(root: Path, run: str, sample_id: str, *, prompt: str = "expand", images: int = 1) -> Path:
    directory = root / run / sample_id
    (directory / "info").mkdir(parents=True)
    Image.new("RGB", (20, 10), "white").save(directory / "input_image.png")
    Image.new("RGB", (40, 30), "gray").save(directory / "target_image.png")
    for index in range(images):
        name = "generated_image.png" if images == 1 else f"generated_{index}.png"
        Image.new("RGB", (40, 30), "green").save(directory / name)
    (directory / "info" / "item_info.json").write_text(json.dumps({
        "source": {
            "prompt": prompt,
            "width": 40,
            "height": 30,
            "evaluation_known_box": [10, 10, 30, 20],
        }
    }), encoding="utf-8")
    return directory


def test_outpainting_task_requires_one_input() -> None:
    assert SampleModel(id="one", task_type="Outpainting", prompt="x", input_images=["input.png"])
    for inputs in ([], ["one.png", "two.png"]):
        try:
            SampleModel(id="bad", task_type="Outpainting", prompt="x", input_images=inputs)
        except ValueError:
            pass
        else:
            raise AssertionError("Outpainting must require exactly one input")


def test_outpainting_catalog_aligns_samples_and_exposes_only_indexed_images(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    add_external_result(root, "BlouOut", "0")
    add_external_result(root, "BlouOut", "2", prompt="second", images=2)
    add_external_result(root, "proout/v1/run", "0")
    catalog = OutpaintingResultCatalog(root, cache_seconds=60)
    runs = catalog.catalog()["runs"]
    assert [run["label"] for run in runs] == ["BlueOut", "ProOut · v1 · run"]
    view = catalog.view([run["id"] for run in runs], page_size=10)
    assert view["total"] == 2
    assert [len(cell) for cell in view["rows"][0]["cells"]] == [1, 1]
    assert [len(cell) for cell in view["rows"][1]["cells"]] == [2, 0]
    assert view["rows"][0]["input_box"] == [10, 10, 30, 20]
    assert view["rows"][0]["target_url"].endswith("/target/0")
    assert catalog.image(runs[0]["id"], "0", "input", 0).name == "input_image.png"
    assert catalog.image(runs[0]["id"], "0", "target", 0).name == "target_image.png"
    assert catalog.image(runs[0]["id"], "../../etc", "generated", 0) is None
    assert catalog.image("missing", "0", "generated", 0) is None


def test_outpainting_comparison_routes(tmp_path: Path) -> None:
    root = tmp_path / "eval"
    add_external_result(root, "BlueOut", "7")
    catalog = OutpaintingResultCatalog(root, cache_seconds=60)
    app = FastAPI()
    app.include_router(outpainting_comparison_router(catalog))
    with TestClient(app) as client:
        assert client.get("/outpainting-compare").status_code == 200
        run_id = client.get("/api/outpainting-comparison/catalog").json()["runs"][0]["id"]
        response = client.get("/api/outpainting-comparison/view", params={"runs": run_id})
        assert response.status_code == 200
        assert response.json()["rows"][0]["prompt"] == "expand"
        assert client.get(f"/api/outpainting-comparison/image/{run_id}/7/generated/0").headers["content-type"] == "image/png"
        assert client.get(f"/api/outpainting-comparison/image/{run_id}/7/target/0").headers["content-type"] == "image/png"
        assert client.get(f"/api/outpainting-comparison/image/{run_id}/7/generated/9").status_code == 404
