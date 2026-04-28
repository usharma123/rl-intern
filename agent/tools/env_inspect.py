from typing import Any

from rl_intern.schemas.env import EnvInspectionResult


def _space_type(space: Any) -> str:
    return type(space).__name__ if space is not None else "unknown"


def inspect_env(env_id: str) -> dict:
    env = None
    try:
        import gymnasium as gym
        from gymnasium import spaces

        env = gym.make(env_id)
        action_space_type = _space_type(env.action_space)
        observation_space_type = _space_type(env.observation_space)
        spec = getattr(env, "spec", None)
        metadata = getattr(env.unwrapped, "metadata", {}) or {}
        result = EnvInspectionResult(
            env_id=env_id,
            observation_space=str(env.observation_space),
            action_space=str(env.action_space),
            action_space_type=action_space_type,
            observation_space_type=observation_space_type,
            is_discrete_action=isinstance(env.action_space, spaces.Discrete),
            is_continuous_action=isinstance(env.action_space, spaces.Box),
            max_episode_steps=getattr(spec, "max_episode_steps", None),
            reward_range=str(getattr(env, "reward_range", None)),
            render_modes=list(metadata.get("render_modes", []) or []),
            warnings=[],
        )
        if action_space_type not in {"Discrete", "Box", "MultiDiscrete", "MultiBinary"}:
            result.warnings.append(f"Unknown action space type: {action_space_type}")
        return result.model_dump()
    except Exception as exc:
        return {
            "env_id": env_id,
            "error": f"Could not inspect environment '{env_id}': {exc}",
            "warnings": [],
        }
    finally:
        if env is not None:
            env.close()
