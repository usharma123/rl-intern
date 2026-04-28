from typing import Any

from rl_intern.runners.modal import (
    fetch_modal_artifacts as _fetch_modal_artifacts,
    get_modal_run_status as _get_modal_run_status,
    launch_modal_experiment as _launch_modal_experiment,
)


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
    return _launch_modal_experiment(
        env_id=env_id,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        seed=seed,
        eval_episodes=eval_episodes,
        max_steps=max_steps,
        run_id=run_id,
        run_dir=run_dir,
    )


def get_modal_run_status(
    modal_call_id: str,
    run_dir: str | None = None,
    timeout: float = 0,
) -> dict[str, Any]:
    return _get_modal_run_status(modal_call_id=modal_call_id, run_dir=run_dir, timeout=timeout)


def fetch_modal_artifacts(
    modal_call_id: str,
    run_dir: str,
    timeout: float = 0,
) -> dict[str, Any]:
    return _fetch_modal_artifacts(modal_call_id=modal_call_id, run_dir=run_dir, timeout=timeout)
