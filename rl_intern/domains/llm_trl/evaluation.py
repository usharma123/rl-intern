from __future__ import annotations

from typing import Any


DEFAULT_EVAL_CATEGORIES = [
    "reasoning",
    "coding",
    "explanation",
    "multilingual",
    "format_following",
]


def smoke_only(max_steps: Any, max_samples: Any, thresholds: dict[str, Any] | None = None) -> bool:
    thresholds = thresholds or {}
    max_smoke_steps = int(thresholds.get("max_steps", 5))
    max_smoke_samples = int(thresholds.get("max_samples", 20))
    try:
        return int(max_steps or 0) <= max_smoke_steps and int(max_samples or 0) <= max_smoke_samples
    except (TypeError, ValueError):
        return False


def sft_evidence(
    eval_metrics: dict[str, Any],
    *,
    threshold_pct: float = 1.0,
    max_steps: Any = None,
    max_samples: Any = None,
    smoke_thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = eval_metrics.get("base") or {}
    adapter = eval_metrics.get("adapter") or {}
    base_loss = _number(base.get("loss"))
    adapter_loss = _number(adapter.get("loss"))
    warnings: list[str] = []
    is_smoke = smoke_only(max_steps, max_samples, smoke_thresholds)
    if is_smoke:
        warnings.append("Tiny training budget; treat this run as smoke-only unless improvement is above threshold.")
    if base_loss is None or adapter_loss is None:
        return _inconclusive("Held-out loss was not available for both base and adapter models.", warnings, is_smoke)
    examples = min(int(base.get("examples") or 0), int(adapter.get("examples") or 0))
    if examples < 3:
        warnings.append("Very small eval sample count; treat this as a smoke signal, not proof.")
    pct = ((base_loss - adapter_loss) / base_loss * 100.0) if base_loss else 0.0
    evidence = _threshold_verdict(
        pct,
        threshold_pct,
        improved_reason=f"Adapter held-out loss improved by {pct:.2f}%.",
        regressed_reason=f"Adapter held-out loss worsened by {abs(pct):.2f}%.",
        inconclusive_reason=f"Loss delta {pct:.2f}% is below the {threshold_pct:.2f}% threshold.",
    )
    evidence.update(
        {
            "method": "sft",
            "base_loss": base_loss,
            "adapter_loss": adapter_loss,
            "delta_loss": adapter_loss - base_loss,
            "delta_loss_pct": pct,
            "threshold_pct": threshold_pct,
            "run_class": "smoke_only" if is_smoke else "standard",
            "warnings": warnings,
        }
    )
    return evidence


def dpo_evidence(
    preference_metrics: dict[str, Any],
    *,
    threshold: float = 0.01,
    max_steps: Any = None,
    max_samples: Any = None,
    smoke_thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_margin = _number((preference_metrics.get("base") or {}).get("mean_margin"))
    adapter_margin = _number((preference_metrics.get("adapter") or {}).get("mean_margin"))
    warnings: list[str] = []
    is_smoke = smoke_only(max_steps, max_samples, smoke_thresholds)
    if is_smoke:
        warnings.append("Tiny training budget; treat this run as smoke-only unless preference margin improves clearly.")
    if base_margin is None or adapter_margin is None:
        return _inconclusive("Preference margin was not available for both base and adapter models.", warnings, is_smoke)
    delta = adapter_margin - base_margin
    evidence = _threshold_verdict(
        delta,
        threshold,
        improved_reason=f"Adapter chosen-vs-rejected margin improved by {delta:.4f}.",
        regressed_reason=f"Adapter chosen-vs-rejected margin worsened by {abs(delta):.4f}.",
        inconclusive_reason=f"Preference margin delta {delta:.4f} is below the {threshold:.4f} threshold.",
    )
    evidence.update(
        {
            "method": "dpo",
            "base_margin": base_margin,
            "adapter_margin": adapter_margin,
            "delta_margin": delta,
            "threshold": threshold,
            "run_class": "smoke_only" if is_smoke else "standard",
            "warnings": warnings,
        }
    )
    return evidence


def grpo_evidence(
    reward_metrics: dict[str, Any],
    *,
    threshold: float = 0.01,
    max_steps: Any = None,
    max_samples: Any = None,
    smoke_thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_reward = _number((reward_metrics.get("base") or {}).get("mean_reward"))
    adapter_reward = _number((reward_metrics.get("adapter") or {}).get("mean_reward"))
    base_failures = int((reward_metrics.get("base") or {}).get("failure_count") or 0)
    adapter_failures = int((reward_metrics.get("adapter") or {}).get("failure_count") or 0)
    warnings: list[str] = []
    is_smoke = smoke_only(max_steps, max_samples, smoke_thresholds)
    if is_smoke:
        warnings.append("Tiny training budget; treat this run as smoke-only unless verifier reward improves clearly.")
    if base_reward is None or adapter_reward is None:
        return _inconclusive("Verifier reward was not available for both base and adapter models.", warnings, is_smoke)
    delta = adapter_reward - base_reward
    if adapter_failures > base_failures:
        evidence = {
            "verdict": "regressed",
            "reason": "Adapter verifier failure count increased.",
        }
    else:
        evidence = _threshold_verdict(
            delta,
            threshold,
            improved_reason=f"Adapter verifier reward improved by {delta:.4f}.",
            regressed_reason=f"Adapter verifier reward worsened by {abs(delta):.4f}.",
            inconclusive_reason=f"Verifier reward delta {delta:.4f} is below the {threshold:.4f} threshold.",
        )
    evidence.update(
        {
            "method": "grpo",
            "base_mean_reward": base_reward,
            "adapter_mean_reward": adapter_reward,
            "delta_mean_reward": delta,
            "base_failure_count": base_failures,
            "adapter_failure_count": adapter_failures,
            "threshold": threshold,
            "run_class": "smoke_only" if is_smoke else "standard",
            "warnings": warnings,
        }
    )
    return evidence


def evaluation_summary(
    *,
    method: str,
    evidence: dict[str, Any],
    eval_dataset: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    samples: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    merged_warnings = list(warnings or [])
    merged_warnings.extend(str(w) for w in evidence.get("warnings", []))
    return {
        "status": "completed",
        "method": method,
        "verdict": evidence.get("verdict", "inconclusive"),
        "run_class": evidence.get("run_class", "standard"),
        "reason": evidence.get("reason"),
        "eval_dataset": eval_dataset or {},
        "metrics": metrics or {},
        "sample_count": len(samples or []),
        "warnings": merged_warnings,
    }


def normalize_preference_row(row: dict[str, Any]) -> dict[str, str] | None:
    prompt = row.get("prompt")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if isinstance(prompt, str) and isinstance(chosen, str) and isinstance(rejected, str):
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    if isinstance(chosen, list) and isinstance(rejected, list):
        chosen_prompt = _first_user(chosen)
        rejected_prompt = _first_user(rejected)
        chosen_completion = _assistant_completion(chosen)
        rejected_completion = _assistant_completion(rejected)
        if chosen_prompt and chosen_prompt == rejected_prompt and chosen_completion and rejected_completion:
            return {
                "prompt": chosen_prompt,
                "chosen": chosen_completion,
                "rejected": rejected_completion,
            }
    return None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _inconclusive(reason: str, warnings: list[str], is_smoke: bool) -> dict[str, Any]:
    return {
        "verdict": "inconclusive",
        "reason": reason,
        "run_class": "smoke_only" if is_smoke else "standard",
        "warnings": warnings,
    }


def _threshold_verdict(
    delta: float,
    threshold: float,
    *,
    improved_reason: str,
    regressed_reason: str,
    inconclusive_reason: str,
) -> dict[str, Any]:
    if delta >= threshold:
        return {"verdict": "improved", "reason": improved_reason}
    if delta <= -threshold:
        return {"verdict": "regressed", "reason": regressed_reason}
    return {"verdict": "inconclusive", "reason": inconclusive_reason}


def _first_user(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user" and message.get("content"):
            return str(message["content"])
    return None


def _assistant_completion(messages: list[dict[str, Any]]) -> str | None:
    parts = [
        str(message["content"])
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant" and message.get("content")
    ]
    return "\n".join(parts) if parts else None
