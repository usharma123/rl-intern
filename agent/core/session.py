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
from rl_intern.events import normalize_event
from rl_intern.run_store import RunStore

logger = logging.getLogger(__name__)


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
