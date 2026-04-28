import asyncio
import json
import logging
import sys
import uuid
from dataclasses import dataclass
from typing import Any

from agent.config import load_config
from agent.core.agent_loop import submission_loop
from agent.core.session import Event, OpType
from agent.core.tools import ToolRouter
from rl_intern.events import normalize_event, utc_now_iso
from rl_intern.run_store import RunStore


@dataclass
class Operation:
    op_type: OpType
    data: dict[str, Any] | None = None


@dataclass
class Submission:
    id: str
    operation: Operation


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    sys.stdout.flush()


class RpcRuntime:
    def __init__(self) -> None:
        self.run_store = RunStore()
        self.submission_queue: asyncio.Queue | None = None
        self.event_queue: asyncio.Queue | None = None
        self.agent_task: asyncio.Task | None = None
        self.forward_task: asyncio.Task | None = None
        self.session_holder: list[Any] = [None]
        self.run_id: str | None = None
        self.run_dir: str | None = None
        self.turn_id: str | None = None

    async def handle(self, message: dict[str, Any]) -> None:
        msg_type = message.get("type")
        if msg_type == "start_run":
            await self.start_run(message)
        elif msg_type == "user_input":
            await self.submit(OpType.USER_INPUT, {"text": message.get("text", "")})
        elif msg_type == "exec_approval":
            await self.submit(
                OpType.EXEC_APPROVAL, {"approvals": message.get("approvals", [])}
            )
        elif msg_type == "interrupt":
            session = self.session_holder[0]
            if session:
                session.cancel()
            _emit({"type": "interrupt_ack", "run_id": self.run_id, "timestamp": utc_now_iso()})
        elif msg_type == "list_runs":
            _emit(
                {
                    "type": "runs",
                    "id": message.get("id"),
                    "runs": self.run_store.list_runs(),
                    "timestamp": utc_now_iso(),
                }
            )
        elif msg_type == "get_run":
            run_id = message.get("run_id")
            _emit(
                {
                    "type": "run",
                    "id": message.get("id"),
                    "run": self.run_store.load_metadata(run_id),
                    "events": self.run_store.read_events(run_id),
                    "artifacts": self.run_store.list_artifacts(run_id),
                    "timestamp": utc_now_iso(),
                }
            )
        elif msg_type == "shutdown":
            await self.shutdown()
            _emit({"type": "shutdown_complete", "timestamp": utc_now_iso()})
        else:
            _emit(
                {
                    "type": "error",
                    "id": message.get("id"),
                    "error": f"Unknown RPC message type: {msg_type}",
                    "timestamp": utc_now_iso(),
                }
            )

    async def start_run(self, message: dict[str, Any]) -> None:
        if self.agent_task and not self.agent_task.done():
            _emit(
                {
                    "type": "error",
                    "id": message.get("id"),
                    "error": "A run is already active in this RPC process.",
                    "timestamp": utc_now_iso(),
                }
            )
            return

        config = load_config()
        if message.get("model"):
            config.model_name = message["model"]
        if message.get("max_iterations") is not None:
            config.max_iterations = int(message["max_iterations"])
        if message.get("yolo") is not None:
            config.yolo_mode = bool(message["yolo"])

        record = self.run_store.create_run(
            run_id=message.get("run_id"),
            model=config.model_name,
            prompt=message.get("prompt"),
        )
        self.run_id = record.run_id
        self.run_dir = str(record.run_dir)
        self.submission_queue = asyncio.Queue()
        self.event_queue = asyncio.Queue()
        self.session_holder = [None]
        self.agent_task = asyncio.create_task(
            submission_loop(
                self.submission_queue,
                self.event_queue,
                config=config,
                tool_router=ToolRouter(local_mode=True),
                session_holder=self.session_holder,
                local_mode=True,
                stream=not bool(message.get("no_stream", False)),
                run_id=record.run_id,
                run_dir=str(record.run_dir),
                run_store=self.run_store,
            )
        )
        self.forward_task = asyncio.create_task(self.forward_events())

    async def submit(self, op_type: OpType, data: dict[str, Any]) -> None:
        if not self.submission_queue:
            _emit({"type": "error", "error": "No active run. Send start_run first."})
            return
        await self.submission_queue.put(
            Submission(
                id=f"rpc_{uuid.uuid4().hex}",
                operation=Operation(op_type=op_type, data=data),
            )
        )

    async def forward_events(self) -> None:
        assert self.event_queue is not None
        while True:
            event: Event = await self.event_queue.get()
            session = self.session_holder[0]
            turn_id = getattr(session, "current_turn_id", None) if session else None
            payload = normalize_event(
                event.event_type, event.data, run_id=self.run_id, turn_id=turn_id
            )
            if event.event_type == "ready":
                payload["run_id"] = self.run_id
                payload["run_dir"] = self.run_dir
            _emit(payload)
            if event.event_type == "shutdown":
                break

    async def shutdown(self) -> None:
        if self.submission_queue:
            await self.submission_queue.put(
                Submission(id="shutdown", operation=Operation(op_type=OpType.SHUTDOWN))
            )
        if self.agent_task:
            try:
                await asyncio.wait_for(self.agent_task, timeout=10)
            except asyncio.TimeoutError:
                self.agent_task.cancel()
        if self.forward_task:
            try:
                await asyncio.wait_for(self.forward_task, timeout=2)
            except asyncio.TimeoutError:
                self.forward_task.cancel()


async def amain() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    runtime = RpcRuntime()
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"type": "error", "error": f"Invalid JSON: {exc}"})
            continue
        await runtime.handle(message)
        if message.get("type") == "shutdown":
            break


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
