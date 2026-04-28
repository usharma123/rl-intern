from agent.tools.algorithm_select import choose_algorithm


def test_cartpole_recommends_ppo_and_dqn():
    result = choose_algorithm("CartPole-v1")

    assert result["recommended_algorithm"] == "PPO"
    assert "PPO" in result["compatible_algorithms"]
    assert "DQN" in result["compatible_algorithms"]


def test_pendulum_recommends_continuous_algorithms():
    result = choose_algorithm("Pendulum-v1")

    assert result["recommended_algorithm"] == "PPO"
    assert result["compatible_algorithms"] == ["PPO", "SAC", "TD3"]


def test_incompatible_preference_warns():
    result = choose_algorithm("CartPole-v1", user_preference="SAC")

    assert result["recommended_algorithm"] == "SAC"
    assert result["warnings"]
