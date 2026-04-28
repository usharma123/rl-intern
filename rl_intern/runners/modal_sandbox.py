from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any


DEFAULT_APP_NAME = "rl-intern-sandbox"
DEFAULT_VOLUME_NAME = "rl-intern-runs"


def _modal_import_error() -> str | None:
    try:
        import modal  # noqa: F401
    except Exception as exc:
        return f"Modal is not installed or importable. Install with `uv sync --extra modal`. Import error: {exc}"
    return None


def create_sandbox(
    *,
    run_id: str,
    hardware: str = "cpu-basic",
    image: str | None = None,
    timeout: int = 60 * 60 * 6,
) -> dict[str, Any]:
    import_error = _modal_import_error()
    if import_error:
        return {"status": "error", "error": import_error}
    try:
        import modal

        app = modal.App.lookup(DEFAULT_APP_NAME, create_if_missing=True)
        volume = modal.Volume.from_name(DEFAULT_VOLUME_NAME, create_if_missing=True)
        modal_image = modal.Image.from_registry(image) if image else modal.Image.debian_slim()
        name = f"rl-intern-{run_id}-{uuid.uuid4().hex[:8]}"
        sandbox = modal.Sandbox.create(
            "sleep",
            str(timeout),
            app=app,
            image=modal_image,
            volumes={"/runs": volume},
            name=name,
            timeout=timeout,
        )
        sandbox.set_tags({"run_id": run_id, "kind": "rl-intern"})
        return {
            "status": "running",
            "sandbox_id": sandbox.object_id,
            "name": name,
            "run_id": run_id,
            "hardware": hardware,
            "message": "Modal sandbox created.",
        }
    except Exception as exc:
        return {"status": "error", "error": f"Failed to create Modal sandbox: {exc}"}


def exec_sandbox(sandbox_id: str, command: str, *, timeout: int = 120) -> dict[str, Any]:
    sandbox = _sandbox_from_id(sandbox_id)
    if isinstance(sandbox, dict):
        return sandbox
    try:
        proc = sandbox.exec("bash", "-lc", command, timeout=timeout)
        stdout = proc.stdout.read()
        stderr = proc.stderr.read()
        return {
            "status": "succeeded" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    except Exception as exc:
        return {"status": "error", "error": f"Sandbox exec failed: {exc}"}


def read_sandbox(sandbox_id: str, path: str) -> dict[str, Any]:
    sandbox = _sandbox_from_id(sandbox_id)
    if isinstance(sandbox, dict):
        return sandbox
    try:
        content = sandbox.filesystem.read_text(path)
        return {"status": "succeeded", "path": path, "content": content}
    except Exception as exc:
        return {"status": "error", "error": f"Sandbox read failed: {exc}", "path": path}


def write_sandbox(sandbox_id: str, path: str, content: str) -> dict[str, Any]:
    sandbox = _sandbox_from_id(sandbox_id)
    if isinstance(sandbox, dict):
        return sandbox
    try:
        remote = Path(path)
        if str(remote.parent) not in {"", "."}:
            sandbox.filesystem.mkdir(str(remote.parent), parents=True, exist_ok=True)
        sandbox.filesystem.write_text(path, content)
        return {"status": "succeeded", "path": path, "bytes": len(content.encode("utf-8"))}
    except Exception as exc:
        return {"status": "error", "error": f"Sandbox write failed: {exc}", "path": path}


def edit_sandbox(
    sandbox_id: str,
    path: str,
    old_str: str,
    new_str: str,
    *,
    replace_all: bool = False,
) -> dict[str, Any]:
    read = read_sandbox(sandbox_id, path)
    if read.get("status") != "succeeded":
        return read
    content = str(read.get("content", ""))
    count = content.count(old_str)
    if count == 0:
        return {"status": "error", "error": "old_str not found", "path": path}
    if count > 1 and not replace_all:
        return {"status": "error", "error": "old_str appears multiple times; set replace_all=true"}
    next_content = content.replace(old_str, new_str, -1 if replace_all else 1)
    result = write_sandbox(sandbox_id, path, next_content)
    result["replacements"] = count if replace_all else 1
    return result


def terminate_sandbox(sandbox_id: str) -> dict[str, Any]:
    sandbox = _sandbox_from_id(sandbox_id)
    if isinstance(sandbox, dict):
        return sandbox
    try:
        sandbox.terminate()
        return {"status": "terminated", "sandbox_id": sandbox_id}
    except Exception as exc:
        return {"status": "error", "error": f"Sandbox terminate failed: {exc}"}


def _sandbox_from_id(sandbox_id: str) -> Any:
    import_error = _modal_import_error()
    if import_error:
        return {"status": "error", "error": import_error}
    try:
        import modal

        return modal.Sandbox.from_id(sandbox_id)
    except Exception as exc:
        return {"status": "error", "error": f"Could not reconnect to sandbox: {exc}"}
