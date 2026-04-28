import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from litellm import Message

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages the LLM conversation context for the RL agent."""

    def __init__(
        self,
        max_context: int = 200_000,
        compact_size: float = 0.1,
        untouched_messages: int = 5,
        tool_specs: list[dict[str, Any]] | None = None,
        prompt_file_suffix: str = "rl_system_prompt.md",
        local_mode: bool = False,
        **_: Any,
    ):
        self.system_prompt = self._load_system_prompt(
            tool_specs or [],
            prompt_file_suffix=prompt_file_suffix,
            local_mode=local_mode,
        )
        self.max_context = max_context - 10_000
        self.compact_size = int(max_context * compact_size)
        self.context_length = 0
        self.untouched_messages = untouched_messages
        self.items: list[Message] = [Message(role="system", content=self.system_prompt)]

    def _load_system_prompt(
        self,
        tool_specs: list[dict[str, Any]],
        prompt_file_suffix: str,
        local_mode: bool,
    ) -> str:
        prompt_file = Path(__file__).parent.parent / "prompts" / prompt_file_suffix
        prompt = prompt_file.read_text(encoding="utf-8")

        now = datetime.now(ZoneInfo("America/New_York"))
        tool_names = [
            spec.get("function", {}).get("name", "unknown") for spec in tool_specs
        ]
        local_context = ""
        if local_mode:
            local_context = (
                "\n\n# Local Execution Context\n\n"
                "You are running as a local CLI on the user's machine. "
                "All RL tools create local artifacts relative to the current working directory.\n"
                f"Working directory: {os.getcwd()}\n"
            )

        return (
            f"{prompt}{local_context}\n\n"
            f"# Available Built-in Tools\n\n{', '.join(tool_names)}\n\n"
            f"[Session context: Date={now.date().isoformat()}, "
            f"Time={now.strftime('%H:%M:%S')}, Timezone=America/New_York, "
            f"Tools={len(tool_specs)}]"
        )

    def add_message(self, message: Message, token_count: int | None = None) -> None:
        if token_count:
            self.context_length = token_count
        self.items.append(message)

    def get_messages(self) -> list[Message]:
        self._patch_dangling_tool_calls()
        return self.items

    async def compact(self, **_: Any) -> None:
        # v0.1 keeps the hook used by the agent loop. Real summarization can be
        # added later without changing Session/agent_loop contracts.
        return None

    def _patch_dangling_tool_calls(self) -> None:
        if not self.items:
            return
        assistant_msg = None
        for msg in reversed(self.items):
            if getattr(msg, "role", None) == "assistant" and getattr(
                msg, "tool_calls", None
            ):
                assistant_msg = msg
                break
            if getattr(msg, "role", None) == "user":
                break
        if not assistant_msg:
            return
        answered_ids = {
            getattr(m, "tool_call_id", None)
            for m in self.items
            if getattr(m, "role", None) == "tool"
        }
        for tc in assistant_msg.tool_calls:
            if tc.id not in answered_ids:
                self.items.append(
                    Message(
                        role="tool",
                        content="Tool was not executed.",
                        tool_call_id=tc.id,
                        name=tc.function.name,
                    )
                )

    def undo_last_turn(self) -> bool:
        if len(self.items) <= 1:
            return False
        while len(self.items) > 1:
            msg = self.items.pop()
            if getattr(msg, "role", None) == "user":
                return True
        return False
