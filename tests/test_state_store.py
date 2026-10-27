from pathlib import Path

from src.runtime.state_store import JsonJobStateStore


def test_job_state_store_round_trip(tmp_path: Path) -> None:
    store = JsonJobStateStore(tmp_path / ".jobs")
    store.save({"id": "job-a", "status": "running", "batch_started_at": "2"})
    store.save({"id": "job-b", "status": "completed", "batch_started_at": "1"})
    snapshots = store.load_all()
    assert [item["id"] for item in snapshots] == ["job-a", "job-b"]
    assert not list((tmp_path / ".jobs").glob(".*.json"))
