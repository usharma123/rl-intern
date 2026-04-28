import math
from typing import Any


def _contains_observation(space: Any, observation: Any) -> bool:
    try:
        return bool(space.contains(observation))
    except Exception:
        return False


def smoke_test_env(
    env_id: str,
    episodes: int = 3,
    max_steps: int = 1000,
    seed: int = 0,
) -> dict:
    env = None
    errors: list[str] = []
    warnings: list[str] = []
    steps_total = 0
    try:
        import gymnasium as gym

        env = gym.make(env_id)
        for episode in range(episodes):
            try:
                observation, _info = env.reset(seed=seed + episode)
            except Exception as exc:
                errors.append(f"reset failed in episode {episode}: {exc}")
                break

            if not _contains_observation(env.observation_space, observation):
                errors.append(f"reset returned invalid observation in episode {episode}")
                break

            terminated = False
            truncated = False
            for step in range(max_steps):
                try:
                    action = env.action_space.sample()
                    observation, reward, terminated, truncated, _info = env.step(action)
                except Exception as exc:
                    errors.append(f"step failed in episode {episode}, step {step}: {exc}")
                    break

                steps_total += 1
                if not _contains_observation(env.observation_space, observation):
                    errors.append(
                        f"step returned invalid observation in episode {episode}, step {step}"
                    )
                    break
                if isinstance(reward, float) and math.isnan(reward):
                    errors.append(f"NaN reward in episode {episode}, step {step}")
                    break
                if terminated or truncated:
                    break
            else:
                warnings.append(
                    f"episode {episode} did not terminate or truncate within {max_steps} steps"
                )

            if errors:
                break

        return {
            "env_id": env_id,
            "passed": not errors,
            "episodes": episodes,
            "steps_total": steps_total,
            "errors": errors,
            "warnings": warnings,
        }
    except Exception as exc:
        return {
            "env_id": env_id,
            "passed": False,
            "episodes": episodes,
            "steps_total": steps_total,
            "errors": [f"Could not create environment '{env_id}': {exc}"],
            "warnings": warnings,
        }
    finally:
        if env is not None:
            env.close()
