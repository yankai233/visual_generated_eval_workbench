from __future__ import annotations

import json
from pathlib import Path

from src.utils.model_paths import resolve_pretrained_path, validate_model_weights


def test_resolve_direct_diffusers_directory(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    assert resolve_pretrained_path(tmp_path) == tmp_path.resolve()


def test_resolve_huggingface_cache_main_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    assert resolve_pretrained_path(tmp_path) == snapshot.resolve()


def test_detect_missing_safetensor_shard(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    existing = transformer / "model-00002-of-00002.safetensors"
    existing.write_bytes(b"ok")
    index = {
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": existing.name,
        }
    }
    (transformer / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    issues = validate_model_weights(str(tmp_path))
    assert any("model-00001-of-00002.safetensors" in issue for issue in issues)
