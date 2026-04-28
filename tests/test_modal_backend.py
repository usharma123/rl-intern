import asyncio
import json
import sys
from types import SimpleNamespace

from rl_intern.runners import modal_backend
from agent.tools import modal_primitives
from rl_intern.modal_jobs.generic import (
    GPU_T4_FUNCTION_NAME,
    _python_unbuffered,
    _run_job_locally,
    function_name_for_hardware,
)
from rl_intern.orchestrator.models import JobRecord


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
    assert result["command"][:2] == ["python", "-u"]


def test_generic_job_uses_unbuffered_python_for_script():
    result = _run_job_locally(
        {
            "run_id": "run_test",
            "script": "print('script output', flush=True)",
            "timeout": "30s",
        }
    )

    assert result["status"] == "succeeded"
    assert result["command"][:2] == ["python", "-u"]
    assert "script output" in result["stdout"]


def test_python_unbuffered_preserves_existing_flag():
    assert _python_unbuffered(["python", "-u", "-c", "print(1)"]) == [
        "python",
        "-u",
        "-c",
        "print(1)",
    ]


def test_modal_job_run_handler_treats_running_launch_as_success(monkeypatch):
    def fake_run_modal_job(**kwargs):
        return {"status": "running", "error": None, "backend_id": "fc-test"}

    monkeypatch.setattr(modal_primitives, "run_modal_job", fake_run_modal_job)

    _, success = asyncio.run(modal_primitives.modal_job_run_handler({}))

    assert success is True


def test_modal_hardware_selects_gpu_function():
    assert function_name_for_hardware("gpu-t4") == GPU_T4_FUNCTION_NAME
    assert function_name_for_hardware("cpu-basic") == "run_job"


def test_modal_job_run_uses_gpu_function_for_t4(monkeypatch, tmp_path):
    requested = {}

    class FakeCall:
        object_id = "fc-test"

    class FakeFunction:
        @classmethod
        def from_name(cls, app_name, function_name):
            requested["app_name"] = app_name
            requested["function_name"] = function_name
            return cls()

        def spawn(self, payload):
            requested["payload"] = payload
            return FakeCall()

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Function=FakeFunction))

    result = modal_backend.run_modal_job(
        run_id="run_test",
        stage="train",
        run_dir=str(tmp_path),
        command=["python", "-c", "print('ok')"],
        hardware="gpu-t4",
    )

    assert result["status"] == "running"
    assert result["modal_function"] == GPU_T4_FUNCTION_NAME
    assert requested["function_name"] == GPU_T4_FUNCTION_NAME


def test_modal_job_cancel_persists_cancelled_job_record(monkeypatch, tmp_path):
    record = JobRecord(
        job_id="job_test",
        run_id="run_test",
        stage="train",
        backend="modal",
        backend_id="fc-test",
        status="running",
    )
    record.write(tmp_path)
    cancelled = {"called": False}

    class FakeFunctionCall:
        @classmethod
        def from_id(cls, backend_id):
            assert backend_id == "fc-test"
            return cls()

        def cancel(self):
            cancelled["called"] = True

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(FunctionCall=FakeFunctionCall))

    result = modal_backend.cancel_modal_job("fc-test", run_dir=str(tmp_path))

    saved = json.loads((tmp_path / "jobs" / "job_test.json").read_text(encoding="utf-8"))
    assert cancelled["called"] is True
    assert result["status"] == "cancelled"
    assert result["job_id"] == "job_test"
    assert saved["status"] == "cancelled"
    assert saved["error"] is None


def test_modal_job_status_persists_succeeded_job_record(monkeypatch, tmp_path):
    record = JobRecord(
        job_id="job_done",
        run_id="run_test",
        stage="smoke_test",
        backend="modal",
        backend_id="fc-done",
        status="running",
    )
    record.write(tmp_path)

    class FakeFunctionCall:
        @classmethod
        def from_id(cls, backend_id):
            assert backend_id == "fc-done"
            return cls()

        def get(self, timeout=0):
            return {"status": "succeeded", "stdout": "ok", "stderr": "", "artifacts": {}}

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(FunctionCall=FakeFunctionCall))

    result = modal_backend.get_modal_job_status("fc-done", run_dir=str(tmp_path))

    saved = json.loads((tmp_path / "jobs" / "job_done.json").read_text(encoding="utf-8"))
    assert result["status"] == "succeeded"
    assert result["job_id"] == "job_done"
    assert saved["status"] == "succeeded"


def test_modal_job_status_keeps_cancelled_record_cancelled(monkeypatch, tmp_path):
    record = JobRecord(
        job_id="job_cancelled",
        run_id="run_test",
        stage="test",
        backend="modal",
        backend_id="fc-cancelled",
        status="cancelled",
    )
    record.write(tmp_path)

    class FakeFunctionCall:
        @classmethod
        def from_id(cls, backend_id):
            assert backend_id == "fc-cancelled"
            return cls()

        def get(self, timeout=0):
            raise RuntimeError("Function call was cancelled by user.")

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(FunctionCall=FakeFunctionCall))

    result = modal_backend.get_modal_job_status("fc-cancelled", run_dir=str(tmp_path))
    logs = modal_backend.get_modal_job_logs("fc-cancelled", run_dir=str(tmp_path))
    artifacts = modal_backend.fetch_modal_job_artifacts("fc-cancelled", run_dir=str(tmp_path))

    saved = json.loads((tmp_path / "jobs" / "job_cancelled.json").read_text(encoding="utf-8"))
    assert result["status"] == "cancelled"
    assert logs["status"] == "cancelled"
    assert artifacts["status"] == "cancelled"
    assert saved["status"] == "cancelled"
