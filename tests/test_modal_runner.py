import base64
import builtins
import json
from pathlib import Path

from rl_intern.run_store import RunStore
from rl_intern.runners.modal import fetch_modal_artifacts, launch_modal_experiment


def test_modal_runner_returns_clean_error_when_modal_missing(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "modal":
            raise ImportError("no modal")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = launch_modal_experiment(
        "CartPole-v1",
        run_id="run_modal_missing",
        run_dir=str(tmp_path / "run_modal_missing"),
    )

    assert result["runner"] == "modal"
    assert result["status"] == "error"
    assert "uv sync --extra modal" in result["error"]


def test_fetch_modal_artifacts_writes_completed_result(monkeypatch, tmp_path):
    encoded = base64.b64encode(b"model").decode("ascii")

    def fake_status(modal_call_id, run_dir=None, timeout=0):
        target = Path(run_dir) / "model.zip"
        target.write_bytes(b"model")
        return {
            "runner": "modal",
            "status": "succeeded",
            "modal_call_id": modal_call_id,
            "synced_artifacts": [str(target)],
            "result": {"artifacts": {"model.zip": encoded}},
        }

    monkeypatch.setattr("rl_intern.runners.modal.get_modal_run_status", fake_status)
    run_dir = tmp_path / "run_modal"

    result = fetch_modal_artifacts("fc-test", str(run_dir))

    assert result["status"] == "succeeded"
    assert (run_dir / "model.zip").exists()
    assert json.loads((run_dir / "modal_status.json").read_text())["modal_call_id"] == "fc-test"


def test_run_store_records_runner_metadata(tmp_path):
    store = RunStore(tmp_path / "runs")
    record = store.create_run(run_id="run_modal_meta", runner="modal")

    metadata = store.load_metadata(record.run_id)

    assert metadata["runner"] == "modal"
