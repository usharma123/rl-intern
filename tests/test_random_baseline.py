from agent.tools.random_baseline import run_random_baseline


def test_random_baseline_cartpole_returns_numeric_rewards():
    result = run_random_baseline("CartPole-v1", episodes=2, seed=0)

    assert "error" not in result
    assert result["episodes"] == 2
    assert isinstance(result["mean_reward"], float)
    assert isinstance(result["std_reward"], float)
    assert len(result["episode_rewards"]) == 2
