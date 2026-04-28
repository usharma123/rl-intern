from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.tools.common import json_ready


def _resolve_run_file(run_dir: str, path: str) -> Path:
    root = Path(run_dir).resolve()
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("path must stay inside run_dir") from exc
    return target


async def read_run_file_handler(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
    try:
        path = _resolve_run_file(args["run_dir"], args["path"])
        content = path.read_text(encoding="utf-8")
        return json_ready({"status": "succeeded", "path": str(path), "content": content}), True
    except Exception as exc:
        return json_ready({"status": "failed", "error": str(exc)}), False


async def write_run_file_handler(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
    try:
        path = _resolve_run_file(args["run_dir"], args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return json_ready({"status": "succeeded", "path": str(path), "bytes": len(args["content"])}), True
    except Exception as exc:
        return json_ready({"status": "failed", "error": str(exc)}), False


async def edit_run_file_handler(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
    try:
        path = _resolve_run_file(args["run_dir"], args["path"])
        content = path.read_text(encoding="utf-8")
        old = args["old_str"]
        count = content.count(old)
        if count == 0:
            return json_ready({"status": "failed", "error": "old_str not found", "path": str(path)}), False
        replacements = -1 if args.get("replace_all") else 1
        next_content = content.replace(old, args["new_str"], replacements)
        path.write_text(next_content, encoding="utf-8")
        return (
            json_ready(
                {
                    "status": "succeeded",
                    "path": str(path),
                    "replacements": count if args.get("replace_all") else 1,
                }
            ),
            True,
        )
    except Exception as exc:
        return json_ready({"status": "failed", "error": str(exc)}), False
