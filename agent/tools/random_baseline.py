import numpy as np


def run_random_baseline(env_id: str, episodes: int = 10, seed: int = 0) -> dict:
    env = None
    rewards: list[float] = []
    try:
        import gymnasium as gym

        env = gym.make(env_id)
        for episode in range(episodes):
            _observation, _info = env.reset(seed=seed + episode)
            env.action_space.seed(seed + episode)
            terminated = False
            truncated = False
            total_reward = 0.0
            while not (terminated or truncated):
                action = env.action_space.sample()
                _observation, reward, terminated, truncated, _info = env.step(action)
                total_reward += float(reward)
            rewards.append(total_reward)

        values = np.asarray(rewards, dtype=float)
        return {
            "env_id": env_id,
            "episodes": episodes,
            "mean_reward": float(values.mean()),
            "std_reward": float(values.std()),
            "min_reward": float(values.min()),
            "max_reward": float(values.max()),
            "seed": seed,
            "episode_rewards": rewards,
        }
    except Exception as exc:
        return {"env_id": env_id, "episodes": episodes, "seed": seed, "error": str(exc)}
    finally:
        if env is not None:
            env.close()
