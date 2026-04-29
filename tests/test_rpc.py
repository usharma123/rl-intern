import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


def _read_json_line(proc, timeout=10):
    result_queue: queue.Queue[str] = queue.Queue()

    def read_line():
        result_queue.put(proc.stdout.readline())

    thread = threading.Thread(target=read_line, daemon=True)
    thread.start()
    try:
        line = result_queue.get(timeout=timeout)
    except queue.Empty as exc:
        raise TimeoutError("Timed out waiting for RPC output") from exc
    if not line:
        stderr = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"RPC process closed stdout. stderr={stderr}")
    return json.loads(line)


def test_rpc_start_user_input_and_shutdown(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    proc = subprocess.Popen(
        [sys.executable, "-m", "rl_intern.rpc"],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        proc.stdin.write(
            json.dumps(
                {
                    "type": "start_run",
                    "id": "start",
                    "model": "openrouter/anthropic/claude-sonnet-4.5",
                    "runner": "modal",
                    "max_iterations": 0,
                }
            )
            + "\n"
        )
        proc.stdin.flush()
        ready = _read_json_line(proc)
        assert ready["type"] == "ready"
        assert ready["run_id"]
        metadata_path = tmp_path / "artifacts" / "runs" / ready["run_id"] / "metadata.json"
        assert json.loads(metadata_path.read_text(encoding="utf-8"))["runner"] == "modal"

        proc.stdin.write(json.dumps({"type": "user_input", "id": "u1", "text": "hello"}) + "\n")
        proc.stdin.flush()
        seen = []
        for _ in range(5):
            event = _read_json_line(proc)
            seen.append(event["type"])
            if event["type"] == "turn_complete":
                break
        assert "turn_complete" in seen

        proc.stdin.write(json.dumps({"type": "shutdown", "id": "s1"}) + "\n")
        proc.stdin.flush()
        shutdown = _read_json_line(proc)
        assert shutdown["type"] in {"shutdown", "shutdown_complete"}
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_rpc_normalizes_openrouter_model_alias(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    proc = subprocess.Popen(
        [sys.executable, "-m", "rl_intern.rpc"],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        proc.stdin.write(
            json.dumps(
                {
                    "type": "start_run",
                    "id": "start",
                    "model": "openai/gpt-oss-120b:free",
                    "max_iterations": 0,
                }
            )
            + "\n"
        )
        proc.stdin.flush()
        ready = _read_json_line(proc)
        assert ready["type"] == "ready"
        metadata_path = tmp_path / "artifacts" / "runs" / ready["run_id"] / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["model"] == "openrouter/openai/gpt-oss-120b:free"

        proc.stdin.write(json.dumps({"type": "shutdown", "id": "s1"}) + "\n")
        proc.stdin.flush()
        shutdown = _read_json_line(proc)
        assert shutdown["type"] in {"shutdown", "shutdown_complete"}
    finally:
        proc.terminate()
        proc.wait(timeout=5)
