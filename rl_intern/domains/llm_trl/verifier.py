from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path
from typing import Any, Callable


def validate_grpo_verifier(
    *,
    verifier_path: str | None = None,
    verifier_source: str | None = None,
    example: dict[str, Any] | None = None,
    completion: str = "test completion",
) -> dict[str, Any]:
    try:
        score_fn = _load_score(verifier_path=verifier_path, verifier_source=verifier_source)
        signature = inspect.signature(score_fn)
        if len(signature.parameters) != 2:
            return {"valid": False, "error": "score must accept exactly (example, completion)"}
        value = score_fn(example or {"prompt": "test", "answer": "test"}, completion)
        numeric = _coerce_score(value)
        if numeric is None or not math.isfinite(numeric):
            return {"valid": False, "error": "score must return a finite number or dict with numeric score"}
        return {"valid": True, "score": numeric, "raw_return_type": type(value).__name__}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def _load_score(
    *,
    verifier_path: str | None,
    verifier_source: str | None,
) -> Callable[[dict[str, Any], str], float | dict[str, Any]]:
    if verifier_source:
        scope: dict[str, Any] = {}
        exec(verifier_source, scope)  # noqa: S102 - explicit user-provided verifier sandbox check
        score = scope.get("score")
    elif verifier_path:
        path = Path(verifier_path)
        spec = importlib.util.spec_from_file_location("rl_intern_grpo_verifier", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not import verifier: {verifier_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        score = getattr(module, "score", None)
    else:
        raise ValueError("verifier_path or verifier_source is required")
    if not callable(score):
        raise ValueError("Verifier must define callable score(example, completion)")
    return score


def _coerce_score(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict) and isinstance(value.get("score"), (int, float)):
        return float(value["score"])
    return None
