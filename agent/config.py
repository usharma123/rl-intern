import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_ALIASES = {
    "nvidia/nemotron-3-nano-30b-a3b:free": "openrouter/nvidia/nemotron-3-nano-30b-a3b:free",
    "google/gemma-4-26b-a4b-it:free": "openrouter/google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-120b:free": "openrouter/openai/gpt-oss-120b:free",
}


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip()
    return _MODEL_ALIASES.get(normalized, normalized)


class Config(BaseModel):
    model_name: str = "openrouter/openai/gpt-oss-120b:free"
    save_sessions: bool = False
    yolo_mode: bool = False
    max_iterations: int = 50
    runner: str = "local"

    @field_validator("model_name")
    @classmethod
    def _normalize_model_name(cls, model_name: str) -> str:
        return normalize_model_name(model_name)


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
