import json
from functools import wraps
from pathlib import Path
from typing import Any, Callable


def json_ready(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=str)


def artifact_dir(
    env_id: str,
    algorithm: str,
    seed: int,
    output_dir: str = "artifacts",
) -> Path:
    safe_env = env_id.replace("/", "_")
    safe_algo = algorithm.upper()
    return Path(output_dir) / safe_env / safe_algo / f"seed_{seed}"


def tool_handler(func: Callable[..., dict[str, Any]]):
    @wraps(func)
    async def wrapper(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
        try:
            result = func(**args)
            ok = not result.get("error") and not result.get("is_error", False)
            return json_ready(result), ok
        except TypeError as exc:
            return json_ready({"error": f"Invalid tool arguments: {exc}"}), False
        except Exception as exc:
            return json_ready({"error": f"{type(exc).__name__}: {exc}"}), False

    return wrapper


def load_sb3_class(algorithm: str):
    algo = algorithm.upper()
    if algo == "PPO":
        from stable_baselines3 import PPO

        return PPO
    if algo == "DQN":
        from stable_baselines3 import DQN

        return DQN
    raise ValueError(f"Unsupported algorithm for v0.1: {algorithm}")
