from pathlib import Path

from agent.tools.train_sb3 import train_sb3


def test_train_ppo_cartpole_saves_model_and_config(tmp_path):
    result = train_sb3(
        "CartPole-v1",
        algorithm="PPO",
        total_timesteps=1000,
        seed=0,
        output_dir=str(tmp_path / "artifacts"),
        log_dir=str(tmp_path / "runs"),
    )

    assert "error" not in result
    assert Path(result["model_path"]).exists()
    assert Path(result["config_path"]).exists()


def test_train_unsupported_algorithm_returns_clean_error(tmp_path):
    result = train_sb3(
        "CartPole-v1",
        algorithm="SAC",
        total_timesteps=100,
        output_dir=str(tmp_path / "artifacts"),
        log_dir=str(tmp_path / "runs"),
    )

    assert "error" in result
    assert "Unsupported algorithm" in result["error"]
