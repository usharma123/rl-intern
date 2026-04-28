from pathlib import Path

from agent.tools.evaluate_policy import evaluate_policy
from agent.tools.report import generate_report
from agent.tools.train_sb3 import train_sb3


def test_run_scoped_training_eval_and_report_paths(tmp_path):
    run_dir = tmp_path / "artifacts" / "runs" / "run_scoped"
    training = train_sb3(
        "CartPole-v1",
        algorithm="PPO",
        total_timesteps=1000,
        seed=0,
        run_dir=str(run_dir),
    )
    evaluation = evaluate_policy(
        "CartPole-v1",
        "PPO",
        training["model_path"],
        episodes=2,
        seed=0,
        run_dir=str(run_dir),
    )
    report = generate_report(
        env_result={"env_id": "CartPole-v1"},
        smoke_test_result={"passed": True},
        random_baseline_result={"mean_reward": 1.0},
        training_result=training,
        evaluation_result=evaluation,
        rollout_result=None,
        run_dir=str(run_dir),
    )

    assert Path(training["model_path"]) == run_dir / "model.zip"
    assert Path(training["config_path"]) == run_dir / "config.json"
    assert Path(evaluation["results_path"]) == run_dir / "eval.json"
    assert Path(report["report_path"]) == run_dir / "report.md"
    assert Path(report["report_path"]).exists()
