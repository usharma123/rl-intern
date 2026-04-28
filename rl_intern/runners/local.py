from agent.tools.algorithm_select import choose_algorithm
from agent.tools.env_inspect import inspect_env
from agent.tools.env_smoke_test import smoke_test_env
from agent.tools.evaluate_policy import evaluate_policy
from agent.tools.random_baseline import run_random_baseline
from agent.tools.record_rollout import record_rollout
from agent.tools.report import generate_report
from agent.tools.train_sb3 import train_sb3
from rl_intern.run_store import RunStore


def run_local_experiment(
    env_id: str,
    algorithm: str = "PPO",
    total_timesteps: int = 100_000,
    seed: int = 0,
    eval_episodes: int = 20,
    run_id: str | None = None,
) -> dict:
    run_record = RunStore().create_run(run_id=run_id, prompt=f"{algorithm} {env_id}")
    run_dir = str(run_record.run_dir)
    env_result = inspect_env(env_id)
    smoke_result = smoke_test_env(env_id, seed=seed)
    algorithm_result = choose_algorithm(env_id, algorithm)
    baseline_result = run_random_baseline(env_id, seed=seed)
    training_result = train_sb3(
        env_id,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        seed=seed,
        run_dir=run_dir,
    )
    if training_result.get("error"):
        return {
            "env_result": env_result,
            "smoke_test_result": smoke_result,
            "algorithm_result": algorithm_result,
            "random_baseline_result": baseline_result,
            "training_result": training_result,
            "run_id": run_record.run_id,
            "run_dir": run_dir,
        }
    evaluation_result = evaluate_policy(
        env_id,
        algorithm,
        training_result["model_path"],
        episodes=eval_episodes,
        seed=seed,
        run_dir=run_dir,
    )
    rollout_result = record_rollout(
        env_id, algorithm, training_result["model_path"], seed=seed, run_dir=run_dir
    )
    report_result = generate_report(
        env_result,
        smoke_result,
        baseline_result,
        training_result,
        evaluation_result,
        rollout_result,
        run_dir=run_dir,
    )
    return {
        "env_result": env_result,
        "smoke_test_result": smoke_result,
        "algorithm_result": algorithm_result,
        "random_baseline_result": baseline_result,
        "training_result": training_result,
        "evaluation_result": evaluation_result,
        "rollout_result": rollout_result,
        "report_result": report_result,
        "run_id": run_record.run_id,
        "run_dir": run_dir,
    }
