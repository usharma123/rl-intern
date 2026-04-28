import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from agent.config import Config
from agent.context_manager.manager import ContextManager
from litellm import ChatCompletionMessageToolCall, Message
from rl_intern.events import normalize_event
from rl_intern.run_store import RunStore

logger = logging.getLogger(__name__)

_RESTORED_TOOL_OUTPUT_LIMIT = 20_000


_MAX_TOKENS_MAP: dict[str, int] = {
    "anthropic/claude-opus-4-6": 200_000,
    "anthropic/claude-sonnet-4-5-20250929": 200_000,
    "anthropic/claude-sonnet-4-20250514": 200_000,
}
_DEFAULT_MAX_TOKENS = 200_000


def _get_max_tokens_safe(model_name: str) -> int:
    return _MAX_TOKENS_MAP.get(model_name, _DEFAULT_MAX_TOKENS)


class OpType(Enum):
    USER_INPUT = "user_input"
    EXEC_APPROVAL = "exec_approval"
    INTERRUPT = "interrupt"
    UNDO = "undo"
    COMPACT = "compact"
    SHUTDOWN = "shutdown"


@dataclass
class Event:
    event_type: str
    data: Optional[dict[str, Any]] = None


class Session:
    """Maintains agent session state."""

    def __init__(
        self,
        event_queue: asyncio.Queue,
        config: Config | None = None,
        tool_router=None,
        context_manager: ContextManager | None = None,
        local_mode: bool = False,
        stream: bool = True,
        run_id: str | None = None,
        run_dir: str | None = None,
        run_store: RunStore | None = None,
        **_: Any,
    ):
        self.tool_router = tool_router
        self.stream = stream
        self.config = config or Config()
        tool_specs = tool_router.get_tool_specs_for_llm() if tool_router else []
        self.context_manager = context_manager or ContextManager(
            max_context=_get_max_tokens_safe(self.config.model_name),
            compact_size=0.1,
            untouched_messages=5,
            tool_specs=tool_specs,
            local_mode=local_mode,
        )
        self.event_queue = event_queue
        self.session_id = str(uuid.uuid4())
        self.is_running = True
        self._cancelled = asyncio.Event()
        self.pending_approval: Optional[dict[str, Any]] = None
        self.logged_events: list[dict[str, Any]] = []
        self.session_start_time = datetime.now().isoformat()
        self.turn_count = 0
        self.run_id = run_id
        self.run_dir = run_dir
        self.run_store = run_store
        self.current_turn_id: str | None = None
        self._restore_context_from_run_log()

    async def send_event(self, event: Event) -> None:
        await self.event_queue.put(event)
        normalized = normalize_event(
            event.event_type,
            event.data,
            run_id=self.run_id,
            turn_id=self.current_turn_id,
        )
        self.logged_events.append(normalized)
        if self.run_store and self.run_id:
            self.run_store.append_event(self.run_id, normalized)

    def cancel(self) -> None:
        self._cancelled.set()

    def reset_cancel(self) -> None:
        self._cancelled.clear()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def update_model(self, model_name: str) -> None:
        self.config.model_name = model_name
        self.context_manager.max_context = _get_max_tokens_safe(model_name)

    def increment_turn(self) -> None:
        self.turn_count += 1

    async def auto_save_if_needed(self) -> None:
        return None

    def save_trajectory_local(self, directory: str = "session_logs") -> str | None:
        try:
            log_dir = Path(directory)
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / f"session_{self.session_id}.json"
            path.write_text(
                json.dumps(
                    {
                        "session_id": self.session_id,
                        "started_at": self.session_start_time,
                        "ended_at": datetime.now().isoformat(),
                        "model_name": self.config.model_name,
                        "events": self.logged_events,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return str(path)
        except Exception as exc:
            logger.warning("Failed to save session locally: %s", exc)
            return None

    def _restore_context_from_run_log(self) -> None:
        if not self.run_store or not self.run_id:
            return
        try:
            events = self.run_store.read_events(self.run_id)
        except Exception as exc:
            logger.warning("Failed to restore session context for %s: %s", self.run_id, exc)
            return
        if not events:
            return

        pending_content: list[str] = []
        pending_tool_calls: list[ChatCompletionMessageToolCall] = []
        tool_calls_flushed = False

        def flush_assistant_with_tools() -> None:
            nonlocal pending_content, pending_tool_calls, tool_calls_flushed
            if not pending_tool_calls or tool_calls_flushed:
                return
            self.context_manager.add_message(
                Message(
                    role="assistant",
                    content="".join(pending_content) or None,
                    tool_calls=pending_tool_calls,
                )
            )
            pending_content = []
            tool_calls_flushed = True

        def flush_plain_assistant() -> None:
            nonlocal pending_content
            if pending_content:
                self.context_manager.add_message(
                    Message(role="assistant", content="".join(pending_content))
                )
                pending_content = []

        def reset_tool_group() -> None:
            nonlocal pending_tool_calls, tool_calls_flushed
            pending_tool_calls = []
            tool_calls_flushed = False

        for event in events:
            event_type = event.get("type")
            if event_type == "user_input":
                if pending_tool_calls:
                    flush_assistant_with_tools()
                    reset_tool_group()
                else:
                    flush_plain_assistant()
                content = event.get("content") or event.get("data", {}).get("text")
                if content:
                    self.context_manager.add_message(Message(role="user", content=str(content)))
                continue

            if event_type == "assistant_chunk":
                if pending_tool_calls and tool_calls_flushed:
                    reset_tool_group()
                pending_content.append(str(event.get("content") or event.get("data", {}).get("content") or ""))
                continue

            if event_type == "tool_call":
                if pending_tool_calls and tool_calls_flushed:
                    reset_tool_group()
                data = event.get("data") or {}
                arguments = data.get("arguments") if isinstance(data.get("arguments"), dict) else event.get("input", {})
                tool_call_id = str(event.get("tool_call_id") or data.get("tool_call_id") or uuid.uuid4())
                actual_tool = str(data.get("actual_tool") or data.get("tool") or event.get("tool") or "")
                pending_tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=tool_call_id,
                        type="function",
                        function={
                            "name": actual_tool,
                            "arguments": json.dumps(arguments or {}, sort_keys=True),
                        },
                    )
                )
                continue

            if event_type == "tool_output":
                flush_assistant_with_tools()
                data = event.get("data") or {}
                output = data.get("output", event.get("output", event.get("content", "")))
                if not isinstance(output, str):
                    output = json.dumps(output, sort_keys=True)
                output = _truncate_restored_tool_output(output)
                tool_name = str(data.get("actual_tool") or data.get("tool") or event.get("tool") or "")
                tool_call_id = str(event.get("tool_call_id") or data.get("tool_call_id") or "")
                if tool_call_id:
                    self.context_manager.add_message(
                        Message(
                            role="tool",
                            content=output,
                            tool_call_id=tool_call_id,
                            name=tool_name,
                        )
                    )
                continue

            if event_type == "turn_complete":
                if pending_tool_calls:
                    flush_assistant_with_tools()
                else:
                    flush_plain_assistant()
                reset_tool_group()

        if pending_tool_calls:
            flush_assistant_with_tools()
        else:
            flush_plain_assistant()


def _truncate_restored_tool_output(output: str) -> str:
    if len(output) <= _RESTORED_TOOL_OUTPUT_LIMIT:
        return output
    omitted = len(output) - _RESTORED_TOOL_OUTPUT_LIMIT
    return (
        output[: _RESTORED_TOOL_OUTPUT_LIMIT // 2]
        + f"\n\n[rl-intern: restored tool output truncated, omitted {omitted} chars]\n\n"
        + output[-_RESTORED_TOOL_OUTPUT_LIMIT // 2 :]
    )
