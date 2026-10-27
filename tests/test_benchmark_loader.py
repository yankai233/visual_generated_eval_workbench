import json
from pathlib import Path

import pytest

from src.data.benchmark_loader import BenchmarkError, load_benchmark


def test_load_jsonl_benchmark(tmp_path: Path) -> None:
    benchmark = tmp_path / "tiny"
    benchmark.mkdir()
    (benchmark / "manifest.jsonl").write_text(
        json.dumps({"id": "1", "task_type": "T2I", "prompt": "hello"}) + "\n",
        encoding="utf-8",
    )
    samples = load_benchmark(tmp_path, "tiny")
    assert len(samples) == 1
    assert samples[0].prompt == "hello"


def test_benchmark_rejects_duplicate_ids(tmp_path: Path) -> None:
    benchmark = tmp_path / "tiny"
    benchmark.mkdir()
    rows = [{"id": "1", "prompt": "a"}, {"id": "1", "prompt": "b"}]
    (benchmark / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(BenchmarkError, match="重复"):
        load_benchmark(tmp_path, "tiny")


def test_load_native_t2i_with_bilingual_prompt(tmp_path: Path) -> None:
    benchmark = tmp_path / "t2i"
    benchmark.mkdir()
    (benchmark / "1.json").write_text(
        json.dumps(
            {
                "id": 1,
                "task_type": "t2i",
                "prompt": {"zh": "中文提示", "en": "English prompt"},
                "evaluation_focus": ["composition"],
            }
        ),
        encoding="utf-8",
    )
    sample = load_benchmark(tmp_path, "t2i", language="en")[0]
    assert sample.id == "1"
    assert sample.task_type == "T2I"
    assert sample.prompt == "English prompt"
    assert sample.extra["evaluation_focus"] == ["composition"]


def test_load_native_multi_reference_images(tmp_path: Path) -> None:
    case = tmp_path / "multi_reference" / "10"
    case.mkdir(parents=True)
    (case / "reference_1.png").write_bytes(b"placeholder")
    (case / "reference_2.png").write_bytes(b"placeholder")
    (case / "info.json").write_text(
        json.dumps(
            {
                "id": 10,
                "task_type": "multi_reference",
                "prompt": {"zh": "组合参考图", "en": "combine references"},
                "reference_images": [
                    {"file": "reference_1.png", "role": "subject"},
                    {"file": "reference_2.png", "role": "style"},
                ],
            }
        ),
        encoding="utf-8",
    )
    sample = load_benchmark(tmp_path, "multi_reference")[0]
    assert sample.task_type == "MultiReference"
    assert sample.input_images == [
        (case / "reference_1.png").resolve().as_posix(),
        (case / "reference_2.png").resolve().as_posix(),
    ]
    assert sample.extra["reference_images"][0]["role"] == "subject"


