from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

try:
    import modal
except Exception:  # pragma: no cover - optional dependency
    modal = None


APP_NAME = "rl-intern-generic"
FUNCTION_NAME = "run_job"
GPU_T4_FUNCTION_NAME = "run_job_gpu_t4"
VOLUME_NAME = "rl-intern-runs"


def function_name_for_hardware(hardware: str | None) -> str:
    normalized = str(hardware or "").strip().lower().replace("_", "-")
    if normalized in {"gpu-t4", "t4"}:
        return GPU_T4_FUNCTION_NAME
    return FUNCTION_NAME


def _build_app():
    if modal is None:
        return None, None, None
    try:
        volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
        image = modal.Image.debian_slim(python_version="3.11").pip_install("uv", "hf-transfer")
        return modal.App(APP_NAME), image, volume
    except Exception:
        # Local imports should keep working without Modal credentials. The deploy
        # command runs this module in an authenticated Modal CLI process, where
        # app/image/volume will be constructed normally.
        return None, None, None


app, _image, _volume = _build_app()


def _collect_artifacts(root: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_file():
            artifacts[str(path.relative_to(root))] = base64.b64encode(path.read_bytes()).decode(
                "ascii"
            )
    return artifacts


def _write_status(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "job_status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _run_job_locally(request: dict[str, Any]) -> dict[str, Any]:
    with TemporaryDirectory(prefix="rl-intern-modal-job-") as tmp:
        root = Path(tmp)
        run_dir = root / request.get("run_id", "run")
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        _write_status(run_dir, {"status": "running", "request": request})

        script = request.get("script")
        command = request.get("command")
        if script and command:
            result = {"status": "failed", "error": "script and command are mutually exclusive"}
            _write_status(run_dir, result)
            result["artifacts"] = _collect_artifacts(run_dir)
            return result
        if not script and not command:
            result = {"status": "failed", "error": "script or command is required"}
            _write_status(run_dir, result)
            result["artifacts"] = _collect_artifacts(run_dir)
            return result

        env = os.environ.copy()
        env.update(request.get("env") or {})
        env.update(request.get("secrets") or {})
        deps = request.get("dependencies") or []
        main_cmd: list[str] | str
        use_shell = False
        if script:
            script_path = run_dir / "job_script.py"
            script_path.write_text(script, encoding="utf-8")
            main_cmd = ["python", "-u", str(script_path), *(request.get("script_args") or [])]
        else:
            if isinstance(command, list):
                main_cmd = _python_unbuffered([str(part) for part in command])
            else:
                main_cmd = str(command)
                use_shell = True

        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            if deps:
                pip_returncode = _run_process_with_tee(
                    ["python", "-m", "pip", "install", *[str(dep) for dep in deps]],
                    cwd=run_dir,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=_timeout_seconds(request.get("timeout", "30m")),
                )
                if pip_returncode != 0:
                    result = {
                        "status": "failed",
                        "returncode": pip_returncode,
                        "stdout": stdout_path.read_text(encoding="utf-8", errors="replace")[-20_000:],
                        "stderr": stderr_path.read_text(encoding="utf-8", errors="replace")[-20_000:],
                        "command": ["python", "-m", "pip", "install", *deps],
                    }
                    _write_status(run_dir, result)
                    result["artifacts"] = _collect_artifacts(run_dir)
                    return result
            returncode = _run_process_with_tee(
                main_cmd,
                cwd=run_dir,
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=_timeout_seconds(request.get("timeout", "30m")),
                shell=use_shell,
            )
        status = "succeeded" if returncode == 0 else "failed"
        result = {
            "status": status,
            "returncode": returncode,
            "stdout": stdout_path.read_text(encoding="utf-8", errors="replace")[-20_000:],
            "stderr": stderr_path.read_text(encoding="utf-8", errors="replace")[-20_000:],
            "command": main_cmd if isinstance(main_cmd, list) else shlex.split(main_cmd),
        }
        _write_status(run_dir, result)
        result["artifacts"] = _collect_artifacts(run_dir)
        return result


def _timeout_seconds(value: str | int | float) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text.endswith("h"):
        return int(float(text[:-1]) * 3600)
    if text.endswith("m"):
        return int(float(text[:-1]) * 60)
    if text.endswith("s"):
        return int(float(text[:-1]))
    return int(float(text))


def _python_unbuffered(command: list[str]) -> list[str]:
    if not command:
        return command
    executable = Path(command[0]).name
    if not executable.startswith("python"):
        return command
    if "-u" in command[1:]:
        return command
    return [command[0], "-u", *command[1:]]


def _run_process_with_tee(
    command: list[str] | str,
    *,
    cwd: Path,
    env: dict[str, str],
    stdout,
    stderr,
    timeout: int,
    shell: bool = False,
) -> int:
    print(f"[rl-intern] running command: {command}", flush=True)
    proc = subprocess.Popen(
        command,
        shell=shell,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def pump(src, file_target, console_target) -> None:
        try:
            for line in iter(src.readline, ""):
                file_target.write(line)
                file_target.flush()
                console_target.write(line)
                console_target.flush()
        finally:
            src.close()

    stdout_thread = threading.Thread(target=pump, args=(proc.stdout, stdout, sys.stdout), daemon=True)
    stderr_thread = threading.Thread(target=pump, args=(proc.stderr, stderr, sys.stderr), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        returncode = proc.wait()
        stderr.write(f"\n[rl-intern] command timed out after {timeout}s\n")
        stderr.flush()
        print(f"[rl-intern] command timed out after {timeout}s", file=sys.stderr, flush=True)
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    print(f"[rl-intern] command exited with code {returncode}", flush=True)
    return returncode


if modal is not None and app is not None and _image is not None and _volume is not None:

    @app.function(image=_image, volumes={"/runs": _volume}, timeout=60 * 60 * 24)
    def run_job(request: dict[str, Any]) -> dict[str, Any]:
        result = _run_job_locally(request)
        _volume.commit()
        return result

    @app.function(image=_image, volumes={"/runs": _volume}, gpu="T4", timeout=60 * 60 * 24)
    def run_job_gpu_t4(request: dict[str, Any]) -> dict[str, Any]:
        result = _run_job_locally(request)
        _volume.commit()
        return result

else:
    run_job = None
    run_job_gpu_t4 = None
