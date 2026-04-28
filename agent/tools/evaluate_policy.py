import json
from pathlib import Path

import numpy as np

from agent.tools.common import artifact_dir, load_sb3_class
from rl_intern.schemas.evaluation import EvaluationResult


def evaluate_policy(
    env_id: str,
    algorithm: str,
    model_path: str,
    episodes: int = 20,
    seed: int = 0,
    run_dir: str | None = None,
) -> dict:
    env = None
    algo = algorithm.upper()
    try:
        import gymnasium as gym

        model_file = Path(model_path)
        if not model_file.exists():
            return {
                "env_id": env_id,
                "algorithm": algo,
                "model_path": model_path,
                "error": "Model file does not exist.",
            }

        env = gym.make(env_id)
        model_cls = load_sb3_class(algo)
        model = model_cls.load(str(model_file), env=env)
        rewards: list[float] = []

        for episode in range(episodes):
            observation, _info = env.reset(seed=seed + episode)
            terminated = False
            truncated = False
            total_reward = 0.0
            while not (terminated or truncated):
                action, _state = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _info = env.step(action)
                total_reward += float(reward)
            rewards.append(total_reward)

        values = np.asarray(rewards, dtype=float)
        result = EvaluationResult(
            env_id=env_id,
            algorithm=algo,
            episodes=episodes,
            mean_reward=float(values.mean()),
            std_reward=float(values.std()),
            min_reward=float(values.min()),
            max_reward=float(values.max()),
            seed=seed,
        ).model_dump()
        result["episode_rewards"] = rewards
        results_path = (
            Path(run_dir) / "eval.json" if run_dir else artifact_dir(env_id, algo, seed) / "eval.json"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["results_path"] = str(results_path)
        return result
    except Exception as exc:
        return {"env_id": env_id, "algorithm": algo, "error": str(exc)}
    finally:
        if env is not None:
            env.close()
