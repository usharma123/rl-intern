from agent.tools.env_inspect import inspect_env
from rl_intern.schemas.env import EnvInspectionResult
from rl_intern.schemas.evaluation import EvaluationResult
from rl_intern.schemas.training import TrainingConfig


def test_inspect_cartpole_reports_discrete_action_space():
    result = inspect_env("CartPole-v1")

    assert "error" not in result
    assert result["action_space_type"] == "Discrete"
    assert result["is_discrete_action"] is True
    assert result["max_episode_steps"] == 500


def test_inspect_invalid_env_returns_clean_error():
    result = inspect_env("DefinitelyMissingEnv-v0")

    assert result["env_id"] == "DefinitelyMissingEnv-v0"
    assert "error" in result


def test_basic_schemas_construct():
    EnvInspectionResult(
        env_id="CartPole-v1",
        observation_space="Box",
        action_space="Discrete(2)",
        action_space_type="Discrete",
        observation_space_type="Box",
        is_discrete_action=True,
        is_continuous_action=False,
    )
    TrainingConfig(env_id="CartPole-v1", algorithm="PPO")
    EvaluationResult(
        env_id="CartPole-v1",
        algorithm="PPO",
        episodes=1,
        mean_reward=1.0,
        std_reward=0.0,
        min_reward=1.0,
        max_reward=1.0,
        seed=0,
    )
