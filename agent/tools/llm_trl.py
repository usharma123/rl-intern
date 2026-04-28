from __future__ import annotations

from agent.tools.common import json_ready
from rl_intern.domains.llm_trl.dataset import inspect_llm_dataset
from rl_intern.domains.llm_trl.scripts import build_trl_training_script
from rl_intern.domains.llm_trl.verifier import validate_grpo_verifier


async def inspect_llm_dataset_handler(args: dict, **_) -> tuple[str, bool]:
    result = inspect_llm_dataset(**args)
    return json_ready(result), bool(result.get("valid")) and "error" not in result


async def validate_grpo_verifier_handler(args: dict, **_) -> tuple[str, bool]:
    result = validate_grpo_verifier(**args)
    return json_ready(result), bool(result.get("valid"))


async def generate_trl_script_handler(args: dict, **_) -> tuple[str, bool]:
    try:
        method = args.get("method", "sft")
        script = build_trl_training_script(method)
        return json_ready({"method": method, "script": script}), True
    except Exception as exc:
        return json_ready({"error": str(exc)}), False
