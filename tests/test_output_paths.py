from src.runtime.job_manager import safe_path_part


def test_safe_path_part() -> None:
    assert safe_path_part("sample / 01", "sample") == "sample_01"
    assert safe_path_part("../../", "sample") == "sample"

