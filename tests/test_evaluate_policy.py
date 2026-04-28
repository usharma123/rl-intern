from pathlib import Path

from agent.tools.evaluate_policy import evaluate_policy
from agent.tools.train_sb3 import train_sb3


def test_evaluate_saved_ppo_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    training = train_sb3(
        "CartPole-v1",
        algorithm="PPO",
        total_timesteps=1000,
        seed=0,
        output_dir="artifacts",
        log_dir="runs",
    )

    result = evaluate_policy(
        "CartPole-v1",
        "PPO",
        training["model_path"],
        episodes=2,
        seed=0,
    )

    assert "error" not in result
    assert result["episodes"] == 2
    assert isinstance(result["mean_reward"], float)
    assert Path(result["results_path"]).exists()
