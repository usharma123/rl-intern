import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config(BaseModel):
    model_name: str = "anthropic/claude-sonnet-4-5-20250929"
    save_sessions: bool = False
    yolo_mode: bool = False
    max_iterations: int = 50
    runner: str = "local"


def substitute_env_vars(obj: Any) -> Any:
    if isinstance(obj, str):
        pattern = r"\$\{([^}:]+)(?::(-)?([^}]*))?\}"

        def replacer(match):
            var_name = match.group(1)
            has_default = match.group(2) is not None
            default_value = match.group(3) if has_default else None
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if has_default:
                return default_value or ""
            raise ValueError(
                f"Environment variable '{var_name}' is not set. Add it to .env."
            )

        return re.sub(pattern, replacer, obj)
    if isinstance(obj, dict):
        return {key: substitute_env_vars(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [substitute_env_vars(item) for item in obj]
    return obj


def load_config(config_path: str | Path | None = None) -> Config:
    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv(override=False)

    if config_path is None:
        config_path = _PROJECT_ROOT / "configs" / "main_agent_config.json"

    path = Path(config_path)
    if not path.exists():
        return Config()

    with path.open("r", encoding="utf-8") as f:
        raw_config = json.load(f)
    return Config.model_validate(substitute_env_vars(raw_config))
