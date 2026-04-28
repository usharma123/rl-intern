from pathlib import Path

import numpy as np

from agent.tools.common import artifact_dir, load_sb3_class


def record_rollout(
    env_id: str,
    algorithm: str,
    model_path: str,
    seed: int = 0,
    max_steps: int = 1000,
    output_dir: str = "artifacts",
    run_dir: str | None = None,
) -> dict:
    env = None
    algo = algorithm.upper()
    try:
        import gymnasium as gym
        import imageio.v2 as imageio

        model_file = Path(model_path)
        if not model_file.exists():
            return {
                "env_id": env_id,
                "algorithm": algo,
                "model_path": model_path,
                "error": "Model file does not exist.",
            }

        env = gym.make(env_id, render_mode="rgb_array")
        model_cls = load_sb3_class(algo)
        model = model_cls.load(str(model_file), env=env)

        observation, _info = env.reset(seed=seed)
        frames = []
        episode_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < max_steps:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
            action, _state = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _info = env.step(action)
            episode_reward += float(reward)
            steps += 1

        if not frames:
            return {
                "env_id": env_id,
                "algorithm": algo,
                "error": "No frames were captured. The environment may not support rgb_array rendering.",
                "episode_reward": episode_reward,
                "steps": steps,
            }

        video_path = (
            Path(run_dir) / "rollout.mp4"
            if run_dir
            else artifact_dir(env_id, algo, seed, output_dir) / "rollout.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            imageio.mimsave(str(video_path), frames, fps=30)
        except Exception as exc:
            gif_path = video_path.with_suffix(".gif")
            try:
                imageio.mimsave(str(gif_path), frames, fps=30)
                video_path = gif_path
            except Exception as gif_exc:
                return {
                    "env_id": env_id,
                    "algorithm": algo,
                    "error": f"Could not write video ({exc}); gif fallback also failed ({gif_exc}).",
                    "episode_reward": episode_reward,
                    "steps": steps,
                }

        return {
            "env_id": env_id,
            "algorithm": algo,
            "video_path": str(video_path),
            "episode_reward": episode_reward,
            "steps": steps,
        }
    except Exception as exc:
        return {
            "env_id": env_id,
            "algorithm": algo,
            "error": f"Could not record rollout: {exc}",
        }
    finally:
        if env is not None:
            env.close()
