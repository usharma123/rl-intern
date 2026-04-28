import asyncio

from rl_intern.runners import modal_backend
from agent.tools import modal_primitives
from rl_intern.modal_jobs.generic import _run_job_locally


def test_modal_job_run_returns_clean_error_when_modal_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(modal_backend, "_modal_import_error", lambda: "no modal")

    result = modal_backend.run_modal_job(
        run_id="run_test",
        stage="train",
        run_dir=str(tmp_path),
        command=["echo", "hi"],
    )

    assert result["status"] == "failed"
    assert "no modal" in result["error"]


def test_modal_artifact_bucket_mapping():
    assert modal_backend._bucket_for(__import__("pathlib").Path("x.log")) == "logs"
    assert modal_backend._bucket_for(__import__("pathlib").Path("eval.json")) == "metrics"
    assert modal_backend._bucket_for(__import__("pathlib").Path("rollout.mp4")) == "videos"


def test_generic_job_preserves_command_array_quotes():
    result = _run_job_locally(
        {
            "run_id": "run_test",
            "command": ["python", "-c", "print('hello from modal')"],
            "timeout": "30s",
        }
    )

    assert result["status"] == "succeeded"
    assert "hello from modal" in result["stdout"]


def test_modal_job_run_handler_treats_running_launch_as_success(monkeypatch):
    def fake_run_modal_job(**kwargs):
        return {"status": "running", "error": None, "backend_id": "fc-test"}

    monkeypatch.setattr(modal_primitives, "run_modal_job", fake_run_modal_job)

    _, success = asyncio.run(modal_primitives.modal_job_run_handler({}))

    assert success is True
