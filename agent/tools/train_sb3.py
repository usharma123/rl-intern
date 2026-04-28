import json
from pathlib import Path

from agent.tools.algorithm_select import choose_algorithm
from agent.tools.common import artifact_dir, load_sb3_class
from agent.tools.env_inspect import inspect_env
from rl_intern.schemas.training import TrainingConfig


def train_sb3(
    env_id: str,
    algorithm: str = "PPO",
    total_timesteps: int = 100_000,
    seed: int = 0,
    output_dir: str = "artifacts",
    log_dir: str = "runs",
    run_dir: str | None = None,
) -> dict:
    env = None
    algo = algorithm.upper()
    try:
        import gymnasium as gym

        env_result = inspect_env(env_id)
        if env_result.get("error"):
            return {"env_id": env_id, "algorithm": algo, "error": env_result["error"]}

        selection = choose_algorithm(env_id, algo)
        warnings = list(selection.get("warnings", []))
        if algo not in {"PPO", "DQN"}:
            return {
                "env_id": env_id,
                "algorithm": algo,
                "error": f"Unsupported algorithm for v0.1: {algo}",
                "warnings": warnings,
            }
        if algo not in selection.get("compatible_algorithms", []):
            return {
                "env_id": env_id,
                "algorithm": algo,
                "error": f"{algo} is not compatible with {env_result.get('action_space_type')} action spaces.",
                "warnings": warnings,
            }

        artifact_path = Path(run_dir) if run_dir else artifact_dir(env_id, algo, seed, output_dir)
        run_log_dir = (
            artifact_path / "logs"
            if run_dir
            else artifact_dir(env_id, algo, seed, log_dir)
        )
        artifact_path.mkdir(parents=True, exist_ok=True)
        run_log_dir.mkdir(parents=True, exist_ok=True)

        config = TrainingConfig(
            env_id=env_id,
            algorithm=algo,
            total_timesteps=total_timesteps,
            seed=seed,
            log_dir=log_dir,
            output_dir=output_dir,
        )
        config_path = artifact_path / "config.json"
        config_path.write_text(
            json.dumps(config.model_dump(), indent=2), encoding="utf-8"
        )

        env = gym.make(env_id)
        env.reset(seed=seed)
        model_cls = load_sb3_class(algo)
        model = model_cls(
            "MlpPolicy",
            env,
            seed=seed,
            tensorboard_log=str(run_log_dir),
            verbose=0,
        )
        model.learn(total_timesteps=total_timesteps)
        model_path = artifact_path / "model.zip"
        model.save(str(model_path))

        return {
            "env_id": env_id,
            "algorithm": algo,
            "total_timesteps": total_timesteps,
            "seed": seed,
            "model_path": str(model_path),
            "config_path": str(config_path),
            "log_dir": str(run_log_dir),
            "warnings": warnings,
        }
    except Exception as exc:
        return {"env_id": env_id, "algorithm": algo, "error": str(exc)}
    finally:
        if env is not None:
            env.close()
