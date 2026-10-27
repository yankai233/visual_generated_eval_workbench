from pathlib import Path

import pytest

from src.utils.config_io import ConfigError, read_config, write_config


def test_config_round_trip(tmp_path: Path) -> None:
    path = write_config(tmp_path, "demo", "test", "model_path: /tmp/model\ncfg: 4.5\n")
    assert path == tmp_path / "demo" / "test" / "config.yaml"
    content, parsed = read_config(tmp_path, "demo", "test")
    assert "cfg: 4.5" in content
    assert parsed["model_path"] == "/tmp/model"


def test_config_rejects_unsafe_name(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        write_config(tmp_path, "../demo", "test", "cfg: 1")


def test_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        write_config(tmp_path, "demo", "test", "- one\n- two\n")

