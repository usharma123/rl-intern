import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

try:
    import modal
except Exception:  # pragma: no cover - exercised when optional dependency is absent
    modal = None


APP_NAME = "rl-intern-sb3"
FUNCTION_NAME = "run_sb3_experiment"


def _build_app():
    if modal is None:
        return None, None
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "swig")
        .pip_install(
            "gymnasium[classic-control,box2d]>=1.0.0",
            "stable-baselines3>=2.7.0",
            "sb3-contrib>=2.7.0",
            "torch",
            "numpy",
            "pandas",
            "matplotlib",
            "jinja2",
            "imageio",
            "moviepy",
            "tensorboard",
            "pygame",
        )
    )
    return modal.App(APP_NAME), image


app, _image = _build_app()


def _collect_artifacts(run_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for path in run_dir.rglob("*"):
        if path.is_file():
            artifacts[str(path.relative_to(run_dir))] = base64.b64encode(path.read_bytes()).decode(
                "ascii"
            )
    return artifacts


def _run_locally(request: dict[str, Any]) -> dict[str, Any]:
    from agent.tools.algorithm_select import choose_algorithm
    from agent.tools.env_inspect import inspect_env
    from agent.tools.env_smoke_test import smoke_test_env
    from agent.tools.evaluate_policy import evaluate_policy
    from agent.tools.random_baseline import run_random_baseline
    from agent.tools.record_rollout import record_rollout
    from agent.tools.report import generate_report
    from agent.tools.train_sb3 import train_sb3

    with TemporaryDirectory(prefix="rl-intern-modal-") as tmp:
        run_dir = Path(tmp) / request["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)

        env_id = request["env_id"]
        algorithm = request.get("algorithm", "PPO")
        seed = int(request.get("seed", 0))
        env_result = inspect_env(env_id)
        smoke_result = smoke_test_env(env_id, seed=seed)
        algorithm_result = choose_algorithm(env_id, algorithm)
        baseline_result = run_random_baseline(env_id, seed=seed)
        training_result = train_sb3(
            env_id,
            algorithm=algorithm,
            total_timesteps=int(request.get("total_timesteps", 100_000)),
            seed=seed,
            run_dir=str(run_dir),
        )
        if training_result.get("error"):
            result = {
                "status": "failed",
                "env_result": env_result,
                "smoke_test_result": smoke_result,
                "algorithm_result": algorithm_result,
                "random_baseline_result": baseline_result,
                "training_result": training_result,
            }
            (run_dir / "modal_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            result["artifacts"] = _collect_artifacts(run_dir)
            return result

        evaluation_result = evaluate_policy(
            env_id,
            algorithm,
            training_result["model_path"],
            episodes=int(request.get("eval_episodes", 20)),
            seed=seed,
            run_dir=str(run_dir),
        )
        rollout_result = record_rollout(
            env_id,
            algorithm,
            training_result["model_path"],
            seed=seed,
            max_steps=int(request.get("max_steps", 1000)),
            run_dir=str(run_dir),
        )
        report_result = generate_report(
            env_result,
            smoke_result,
            baseline_result,
            training_result,
            evaluation_result,
            rollout_result,
            run_dir=str(run_dir),
        )
        result = {
            "status": "succeeded",
            "env_result": env_result,
            "smoke_test_result": smoke_result,
            "algorithm_result": algorithm_result,
            "random_baseline_result": baseline_result,
            "training_result": training_result,
            "evaluation_result": evaluation_result,
            "rollout_result": rollout_result,
            "report_result": report_result,
        }
        (run_dir / "modal_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["artifacts"] = _collect_artifacts(run_dir)
        return result


if modal is not None:

    @app.function(image=_image, timeout=60 * 60, include_source=True)
    def run_sb3_experiment(request: dict[str, Any]) -> dict[str, Any]:
        return _run_locally(request)

else:
    run_sb3_experiment = None
