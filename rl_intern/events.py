import json
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_json_output(output: Any) -> Any:
    if not isinstance(output, str):
        return output
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return output


def normalize_event(
    event_type: str,
    data: dict[str, Any] | None,
    *,
    run_id: str | None,
    turn_id: str | None,
) -> dict[str, Any]:
    payload = data or {}
    normalized: dict[str, Any] = {
        "type": event_type,
        "run_id": run_id,
        "turn_id": turn_id,
        "timestamp": utc_now_iso(),
        "data": payload,
        "rl_intern": {"schema_version": "0.1"},
    }

    if event_type in {"tool_call", "tool_output", "tool_state_change"}:
        normalized["tool"] = payload.get("tool")
        normalized["tool_call_id"] = payload.get("tool_call_id")

    if event_type == "tool_call":
        normalized["input"] = payload.get("arguments", {})
        normalized["status"] = "started"
        normalized["role"] = "assistant"
        normalized["recipient"] = payload.get("tool")
    elif event_type == "tool_output":
        normalized["output"] = parse_json_output(payload.get("output"))
        normalized["status"] = "success" if payload.get("success") else "error"
        normalized["role"] = "tool"
        normalized["content"] = normalized["output"]
    elif event_type == "tool_state_change":
        normalized["status"] = payload.get("state")
    elif event_type == "approval_required":
        normalized["status"] = "waiting_for_approval"
    elif event_type == "error":
        normalized["status"] = "error"
    elif event_type == "turn_complete":
        normalized["status"] = "complete"
    elif event_type == "user_input":
        normalized["role"] = "user"
        normalized["content"] = payload.get("text", "")
    elif event_type in {"assistant_chunk", "assistant_message"}:
        normalized["role"] = "assistant"
        normalized["content"] = payload.get("content", "")
    elif event_type == "plan_update":
        normalized["status"] = payload.get("status", "updated")
        normalized["plan"] = payload.get("plan")
        normalized["stage"] = payload.get("stage")
    elif event_type == "job_update":
        normalized["status"] = payload.get("status", payload.get("state", "updated"))
        normalized["job_id"] = payload.get("job_id")
        normalized["backend_id"] = payload.get("backend_id")
    elif event_type == "artifact_manifest":
        normalized["status"] = "updated"
        normalized["manifest"] = payload.get("manifest")

    return normalized


def to_json_line(event: dict[str, Any]) -> str:
    return json.dumps(event, sort_keys=True, default=str) + "\n"
