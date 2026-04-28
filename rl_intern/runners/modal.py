import base64
import json
from pathlib import Path
from typing import Any

from rl_intern.modal_jobs.sb3 import APP_NAME, FUNCTION_NAME
from rl_intern.run_store import RunStore
from rl_intern.schemas.modal import ModalExperimentRequest, ModalJobReference


def _modal_import_error() -> str | None:
    try:
        import modal  # noqa: F401
    except Exception as exc:
        return (
            "Modal is not installed or importable. Install it with "
            "`uv sync --extra modal`, then run `modal setup`."
            f" Import error: {exc}"
        )
    return None


def _call_id(function_call: Any) -> str | None:
    for attr in ("object_id", "id", "call_id"):
        value = getattr(function_call, attr, None)
        if value:
            return str(value)
    return None


def _write_artifacts(run_dir: str | Path, artifacts: dict[str, str]) -> list[str]:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for relative_path, encoded in artifacts.items():
        safe_relative = Path(relative_path)
        if safe_relative.is_absolute() or ".." in safe_relative.parts:
            continue
        target = root / safe_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(encoded.encode("ascii")))
        written.append(str(target))
    return written


def _metadata_run_id(run_dir: str | Path | None, fallback: str | None = None) -> str | None:
    if fallback:
        return fallback
    if not run_dir:
        return None
    return Path(run_dir).name


def launch_modal_experiment(
    env_id: str,
    algorithm: str = "PPO",
    total_timesteps: int = 100_000,
    seed: int = 0,
    eval_episodes: int = 20,
    max_steps: int = 1000,
    run_id: str | None = None,
    run_dir: str | None = None,
) -> dict[str, Any]:
    import_error = _modal_import_error()
    effective_run_id = _metadata_run_id(run_dir, run_id)
    if import_error:
        return {
            "runner": "modal",
            "status": "error",
            "error": import_error,
            "run_id": effective_run_id,
        }

    try:
        import modal

        request = ModalExperimentRequest(
            run_id=effective_run_id or "run_modal",
            env_id=env_id,
            algorithm=algorithm,
            total_timesteps=total_timesteps,
            seed=seed,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
        )
        fn = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
        call = fn.spawn(request.model_dump())
        modal_call_id = _call_id(call)
        if not modal_call_id:
            return {
                "runner": "modal",
                "status": "error",
                "error": "Modal did not return a function call id.",
            }
        ref = ModalJobReference(
            modal_app=APP_NAME,
            modal_function=FUNCTION_NAME,
            modal_call_id=modal_call_id,
            status="running",
        ).model_dump()
        ref.update(
            {
                "run_id": request.run_id,
                "env_id": env_id,
                "algorithm": algorithm.upper(),
                "total_timesteps": total_timesteps,
                "seed": seed,
                "eval_episodes": eval_episodes,
            }
        )
        if run_dir and effective_run_id:
            RunStore(Path(run_dir).parent).update_metadata(effective_run_id, ref)
        return ref
    except Exception as exc:
        return {
            "runner": "modal",
            "status": "error",
            "run_id": effective_run_id,
            "modal_app": APP_NAME,
            "modal_function": FUNCTION_NAME,
            "error": (
                f"Failed to launch Modal job: {exc}. "
                "Ensure you ran `modal setup` and deployed the job with "
                "`uv run modal deploy rl_intern/modal_jobs/sb3.py`."
            ),
        }


def get_modal_run_status(
    modal_call_id: str,
    run_dir: str | None = None,
    timeout: float = 0,
) -> dict[str, Any]:
    import_error = _modal_import_error()
    if import_error:
        return {"runner": "modal", "status": "error", "error": import_error}

    try:
        import modal

        call = modal.FunctionCall.from_id(modal_call_id)
        try:
            result = call.get(timeout=timeout)
        except TimeoutError:
            return {
                "runner": "modal",
                "status": "running",
                "modal_call_id": modal_call_id,
                "message": "Modal job is still running.",
            }
        except Exception as exc:
            if exc.__class__.__name__.lower().endswith("timeout"):
                return {
                    "runner": "modal",
                    "status": "running",
                    "modal_call_id": modal_call_id,
                    "message": "Modal job is still running.",
                }
            raise

        output = {
            "runner": "modal",
            "status": result.get("status", "succeeded"),
            "modal_call_id": modal_call_id,
            "result": {k: v for k, v in result.items() if k != "artifacts"},
        }
        if run_dir and isinstance(result.get("artifacts"), dict):
            output["synced_artifacts"] = _write_artifacts(run_dir, result["artifacts"])
        return output
    except Exception as exc:
        return {
            "runner": "modal",
            "status": "error",
            "modal_call_id": modal_call_id,
            "error": f"Failed to read Modal job status: {exc}",
        }


def fetch_modal_artifacts(
    modal_call_id: str,
    run_dir: str,
    timeout: float = 0,
) -> dict[str, Any]:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    status = get_modal_run_status(modal_call_id, run_dir=run_dir, timeout=timeout)
    if status.get("status") == "running":
        return {
            "runner": "modal",
            "status": "running",
            "modal_call_id": modal_call_id,
            "message": "Modal job is still running; artifacts are not ready.",
        }
    if status.get("status") == "error":
        return status

    artifacts = status.get("synced_artifacts", [])
    result_path = Path(run_dir) / "modal_status.json"
    result_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "runner": "modal",
        "status": status.get("status", "succeeded"),
        "modal_call_id": modal_call_id,
        "run_dir": run_dir,
        "artifacts": artifacts,
        "status_path": str(result_path),
        "result": status.get("result", {}),
    }
