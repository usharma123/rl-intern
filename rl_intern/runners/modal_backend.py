from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path
from typing import Any

from rl_intern.modal_jobs.generic import APP_NAME, function_name_for_hardware
from rl_intern.orchestrator.manifest import append_manifest_item
from rl_intern.orchestrator.models import JobRecord, utc_now_iso

_LOG_SNIPPET_CHARS = 12_000


def _modal_import_error() -> str | None:
    try:
        import modal  # noqa: F401
    except Exception as exc:
        return f"Modal is not installed or importable. Install with `uv sync --extra modal`. Import error: {exc}"
    return None


def _call_id(function_call: Any) -> str | None:
    for attr in ("object_id", "id", "call_id"):
        value = getattr(function_call, attr, None)
        if value:
            return str(value)
    return None


def _read_script(script_path: str | None, script: str | None) -> str | None:
    if script is not None:
        return script
    if not script_path:
        return None
    return Path(script_path).read_text(encoding="utf-8")


def run_modal_job(
    *,
    run_id: str,
    stage: str,
    run_dir: str | None = None,
    script_path: str | None = None,
    script: str | None = None,
    command: list[str] | str | None = None,
    script_args: list[str] | None = None,
    dependencies: list[str] | None = None,
    hardware: str = "cpu-basic",
    timeout: str = "30m",
    env: dict[str, str] | None = None,
    secrets: dict[str, str] | None = None,
) -> dict[str, Any]:
    import_error = _modal_import_error()
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    record = JobRecord(
        job_id=job_id,
        run_id=run_id,
        stage=stage,
        backend="modal",
        status="pending",
        hardware=hardware,
    )
    if run_dir:
        record.write(run_dir)
    if import_error:
        record.status = "failed"
        record.error = import_error
        if run_dir:
            record.write(run_dir)
        return {**record.model_dump(), "error": import_error}

    try:
        import modal

        payload = {
            "run_id": run_id,
            "stage": stage,
            "script": _read_script(script_path, script),
            "command": command,
            "script_args": script_args or [],
            "dependencies": dependencies or [],
            "hardware": hardware,
            "timeout": timeout,
            "env": env or {},
            "secrets": secrets or {},
        }
        function_name = function_name_for_hardware(hardware)
        fn = modal.Function.from_name(APP_NAME, function_name)
        call = fn.spawn(payload)
        backend_id = _call_id(call)
        if not backend_id:
            raise RuntimeError("Modal did not return a call id")
        record.backend_id = backend_id
        record.status = "running"
        if run_dir:
            record.write(run_dir)
        return {
            **record.model_dump(),
            "modal_app": APP_NAME,
            "modal_function": function_name,
            "message": "Modal job launched. Use modal_job_status/logs/artifacts to poll.",
        }
    except Exception as exc:
        record.status = "failed"
        record.error = str(exc)
        if run_dir:
            record.write(run_dir)
        return {**record.model_dump(), "error": f"Failed to launch Modal job: {exc}"}


def get_modal_job_status(backend_id: str, *, run_dir: str | None = None, timeout: float = 0) -> dict[str, Any]:
    record = _find_job_record(run_dir, backend_id) if run_dir else None
    import_error = _modal_import_error()
    if import_error:
        output = {"status": "failed", "error": import_error, "backend_id": backend_id}
        _persist_status_result(run_dir, record, output)
        return output
    try:
        import modal

        call = modal.FunctionCall.from_id(backend_id)
        try:
            result = call.get(timeout=timeout)
        except TimeoutError:
            output = {"status": "running", "backend_id": backend_id}
            _persist_status_result(run_dir, record, output)
            return output
        except Exception as exc:
            if exc.__class__.__name__.lower().endswith("timeout"):
                output = {"status": "running", "backend_id": backend_id}
                _persist_status_result(run_dir, record, output)
                return output
            raise
        status = result.get("status", "succeeded")
        output = {"status": status, "backend_id": backend_id, "result": _strip_artifacts(result)}
        if run_dir and isinstance(result.get("artifacts"), dict):
            output["artifacts"] = _write_artifacts(run_dir, result["artifacts"])
        _persist_status_result(run_dir, record, output)
        return output
    except Exception as exc:
        if _is_cancelled_error(exc) or (record and record.status == "cancelled"):
            output = {"status": "cancelled", "backend_id": backend_id, "error": str(exc)}
        else:
            output = {"status": "failed", "backend_id": backend_id, "error": str(exc)}
        _persist_status_result(run_dir, record, output)
        return output


def get_modal_job_logs(backend_id: str, *, run_dir: str | None = None, timeout: float = 0) -> dict[str, Any]:
    status = get_modal_job_status(backend_id, run_dir=run_dir, timeout=timeout)
    result = status.get("result") or {}
    return {
        "status": status.get("status"),
        "backend_id": backend_id,
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
    }


def cancel_modal_job(backend_id: str, *, run_dir: str | None = None) -> dict[str, Any]:
    record = _find_job_record(run_dir, backend_id) if run_dir else None
    import_error = _modal_import_error()
    if import_error:
        result = {"status": "failed", "error": import_error, "backend_id": backend_id}
        _persist_cancel_result(run_dir, record, result)
        return result
    try:
        import modal

        call = modal.FunctionCall.from_id(backend_id)
        call.cancel()
        result = {"status": "cancelled", "backend_id": backend_id}
        _persist_cancel_result(run_dir, record, result)
        return result
    except Exception as exc:
        result = {"status": "failed", "backend_id": backend_id, "error": str(exc)}
        _persist_cancel_result(run_dir, record, result)
        return result


def fetch_modal_job_artifacts(backend_id: str, *, run_dir: str, timeout: float = 0) -> dict[str, Any]:
    status = get_modal_job_status(backend_id, run_dir=run_dir, timeout=timeout)
    if status.get("status") == "running":
        return status
    artifacts = status.get("artifacts", [])
    return {"status": status.get("status"), "backend_id": backend_id, "artifacts": artifacts}


def _strip_artifacts(result: dict[str, Any]) -> dict[str, Any]:
    stripped = {key: value for key, value in result.items() if key != "artifacts"}
    for key in ("stdout", "stderr"):
        value = stripped.get(key)
        if isinstance(value, str) and len(value) > _LOG_SNIPPET_CHARS:
            stripped[key] = value[-_LOG_SNIPPET_CHARS:]
            stripped[f"{key}_truncated"] = True
    return stripped


def _write_artifacts(run_dir: str | Path, artifacts: dict[str, str]) -> list[str]:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for relative_path, encoded in artifacts.items():
        safe = Path(relative_path)
        if safe.is_absolute() or ".." in safe.parts:
            continue
        target = root / "modal_artifacts" / safe
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(encoded.encode("ascii")))
        written.append(str(target))
        bucket = _bucket_for(target)
        append_manifest_item(root, bucket, target, kind=f"modal_{bucket.rstrip('s')}")
    return written


def _find_job_record(run_dir: str | Path | None, backend_id: str) -> JobRecord | None:
    if not run_dir:
        return None
    jobs_dir = Path(run_dir) / "jobs"
    if not jobs_dir.exists():
        return None
    for path in sorted(jobs_dir.glob("*.json")):
        try:
            record = JobRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        if record.backend_id == backend_id:
            return record
    return None


def _persist_cancel_result(
    run_dir: str | Path | None,
    record: JobRecord | None,
    result: dict[str, Any],
) -> None:
    if not run_dir or record is None:
        return
    record.status = "cancelled" if result.get("status") == "cancelled" else "failed"
    record.error = result.get("error")
    record.updated_at = utc_now_iso()
    record.write(run_dir)
    result.setdefault("job_id", record.job_id)
    result.setdefault("run_id", record.run_id)
    result.setdefault("stage", record.stage)


def _persist_status_result(
    run_dir: str | Path | None,
    record: JobRecord | None,
    result: dict[str, Any],
) -> None:
    if not run_dir or record is None:
        return
    status = result.get("status")
    if record.status == "cancelled" and status == "failed":
        status = "cancelled"
        result["status"] = "cancelled"
    if status in {"running", "succeeded", "failed", "cancelled"}:
        record.status = status
    record.error = result.get("error")
    artifacts = result.get("artifacts")
    if isinstance(artifacts, list):
        record.artifact_paths = [str(path) for path in artifacts]
    record.updated_at = utc_now_iso()
    record.write(run_dir)
    result.setdefault("job_id", record.job_id)
    result.setdefault("run_id", record.run_id)
    result.setdefault("stage", record.stage)


def _is_cancelled_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "cancelled by user" in text or "canceled by user" in text or "cancelled" in text


def _bucket_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".log", ".txt"}:
        return "logs"
    if suffix in {".json", ".jsonl", ".csv"}:
        return "metrics"
    if suffix in {".md", ".html"}:
        return "reports"
    if suffix in {".mp4", ".gif"}:
        return "videos"
    if path.name.endswith(".zip") or "checkpoint" in path.parts:
        return "checkpoints"
    return "configs"
