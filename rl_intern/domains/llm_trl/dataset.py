from __future__ import annotations

from typing import Any


def inspect_llm_dataset(
    dataset_path: str | None = None,
    *,
    rows: list[dict[str, Any]] | None = None,
    method: str = "sft",
    sample_rows: int = 3,
) -> dict[str, Any]:
    method = method.lower()
    loaded_rows = rows
    source = "inline"
    warnings: list[str] = []
    if loaded_rows is None:
        if not dataset_path:
            return {"method": method, "error": "dataset_path or rows is required"}
        source = dataset_path
        try:
            loaded_rows = _load_dataset_sample(dataset_path, sample_rows)
        except Exception as exc:
            return {
                "method": method,
                "dataset_path": dataset_path,
                "error": f"Could not load dataset sample: {exc}",
            }
    sample = loaded_rows[:sample_rows]
    columns = sorted({key for row in sample for key in row})
    valid, reason = _validate_columns(columns, method)
    if method == "grpo" and "answer" not in columns and "target" not in columns:
        warnings.append("GRPO verifier may need an answer/target column depending on the task.")
    return {
        "method": method,
        "source": source,
        "sample_rows": sample,
        "columns": columns,
        "valid": valid,
        "format": _format_name(columns, method) if valid else None,
        "reason": reason,
        "warnings": warnings,
    }


def _load_dataset_sample(dataset_path: str, sample_rows: int) -> list[dict[str, Any]]:
    if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
        return _load_local_json(dataset_path, sample_rows)
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("Install `datasets` to inspect Hugging Face datasets") from exc
    ds = load_dataset(dataset_path, split=f"train[:{sample_rows}]")
    return [dict(row) for row in ds]


def _load_local_json(path: str, sample_rows: int) -> list[dict[str, Any]]:
    import json
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()][:sample_rows]
    payload = json.loads(text)
    if isinstance(payload, list):
        return [dict(row) for row in payload[:sample_rows]]
    if isinstance(payload, dict):
        for key in ("train", "rows", "data"):
            if isinstance(payload.get(key), list):
                return [dict(row) for row in payload[key][:sample_rows]]
    raise ValueError("JSON dataset must be a list or contain train/rows/data list")


def _validate_columns(columns: list[str], method: str) -> tuple[bool, str]:
    col = set(columns)
    if method == "sft":
        if "messages" in col or "text" in col or {"prompt", "completion"} <= col:
            return True, "SFT dataset has messages, text, or prompt/completion columns."
        return False, "SFT requires messages, text, or prompt/completion columns."
    if method == "dpo":
        if {"prompt", "chosen", "rejected"} <= col:
            return True, "DPO dataset has prompt/chosen/rejected columns."
        return False, "DPO requires prompt, chosen, and rejected columns."
    if method == "grpo":
        if "prompt" in col:
            return True, "GRPO dataset has prompt column."
        return False, "GRPO requires a prompt column."
    return False, f"Unsupported TRL method: {method}"


def _format_name(columns: list[str], method: str) -> str:
    col = set(columns)
    if method == "sft" and "messages" in col:
        return "chat_messages"
    if method == "sft" and "text" in col:
        return "text"
    if method == "sft":
        return "prompt_completion"
    if method == "dpo":
        return "preference_pairs"
    return "prompt_verifier"
