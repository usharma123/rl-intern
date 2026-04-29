import argparse
import asyncio
import json
import logging
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import litellm

from agent.config import load_config, normalize_model_name
from agent.core.agent_loop import submission_loop
from agent.core.session import OpType
from agent.core.tools import ToolRouter
from rl_intern.run_store import RunStore

litellm.drop_params = True


@dataclass
class Operation:
    op_type: OpType
    data: Optional[dict[str, Any]] = None


@dataclass
class Submission:
    id: str
    operation: Operation


def _print_tool_event(name: str, arguments: dict[str, Any]) -> None:
    args = json.dumps(arguments, default=str)
    if len(args) > 160:
        args = args[:157] + "..."
    print(f"\n[tool] {name} {args}", file=sys.stderr)


async def _run_agent_prompt(
    prompt: str,
    model: str | None,
    max_iterations: int | None,
    stream: bool,
    headless: bool,
    runner: str,
) -> None:
    config = load_config()
    if model:
        config.model_name = normalize_model_name(model)
    config.runner = runner
    if max_iterations is not None:
        config.max_iterations = max_iterations
    if headless:
        config.yolo_mode = True

    submission_queue: asyncio.Queue = asyncio.Queue()
    event_queue: asyncio.Queue = asyncio.Queue()
    tool_router = ToolRouter(local_mode=True)
    session_holder: list = [None]
    run_store = RunStore()
    run_record = run_store.create_run(
        model=config.model_name,
        prompt=prompt,
        runner=runner,
    )
    print(f"Run: {run_record.run_id}", file=sys.stderr)

    agent_task = asyncio.create_task(
        submission_loop(
            submission_queue,
            event_queue,
            config=config,
            tool_router=tool_router,
            session_holder=session_holder,
            local_mode=True,
            stream=stream,
            run_id=run_record.run_id,
            run_dir=str(run_record.run_dir),
            run_store=run_store,
        )
    )

    while True:
        event = await event_queue.get()
        if event.event_type == "ready":
            break

    await submission_queue.put(
        Submission(
            id="sub_1",
            operation=Operation(op_type=OpType.USER_INPUT, data={"text": prompt}),
        )
    )

    while True:
        event = await event_queue.get()
        data = event.data or {}
        if event.event_type == "assistant_chunk":
            print(data.get("content", ""), end="", flush=True)
        elif event.event_type == "assistant_stream_end":
            print()
        elif event.event_type == "assistant_message":
            if data.get("content"):
                print(data["content"])
        elif event.event_type == "tool_call":
            _print_tool_event(data.get("tool", ""), data.get("arguments", {}))
        elif event.event_type == "tool_output":
            success = "ok" if data.get("success") else "failed"
            print(f"[tool:{success}] {data.get('tool')}", file=sys.stderr)
        elif event.event_type == "approval_required":
            tools = data.get("tools", [])
            approvals = []
            for tool in tools:
                if headless:
                    approved = True
                else:
                    print(
                        f"\nApproval required for {tool.get('tool')} with arguments:",
                        file=sys.stderr,
                    )
                    print(json.dumps(tool.get("arguments", {}), indent=2), file=sys.stderr)
                    response = input("Approve? [y/N]: ").strip().lower()
                    approved = response in {"y", "yes"}
                approvals.append(
                    {
                        "tool_call_id": tool.get("tool_call_id", ""),
                        "approved": approved,
                        "feedback": None,
                    }
                )
            await submission_queue.put(
                Submission(
                    id=f"approval_{uuid.uuid4().hex}",
                    operation=Operation(
                        op_type=OpType.EXEC_APPROVAL,
                        data={"approvals": approvals},
                    ),
                )
            )
        elif event.event_type == "error":
            print(f"ERROR: {data.get('error', 'unknown error')}", file=sys.stderr)
            break
        elif event.event_type in {"turn_complete", "interrupted"}:
            break

    await submission_queue.put(
        Submission(id="shutdown", operation=Operation(op_type=OpType.SHUTDOWN))
    )
    try:
        await asyncio.wait_for(agent_task, timeout=10)
    except asyncio.TimeoutError:
        agent_task.cancel()


async def _interactive(
    model: str | None,
    max_iterations: int | None,
    stream: bool,
    runner: str,
) -> None:
    print("rl-intern")
    print("Type /exit to quit.")
    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
            return
        await _run_agent_prompt(
            prompt,
            model=model,
            max_iterations=max_iterations,
            stream=stream,
            headless=False,
            runner=runner,
        )


def cli() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(
        prog="rl-intern",
        description=(
            "Deprecated prompt CLI for RL Intern. Use the web app with "
            "`cd frontend && bun run dev`."
        ),
    )
    parser.add_argument("prompt", nargs="?", default=None, help="Run one prompt headlessly.")
    parser.add_argument("--model", "-m", default=None, help="LiteLLM model name.")
    parser.add_argument(
        "--runner",
        choices=["local", "modal"],
        default="local",
        help="Execution backend for RL experiments.",
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--no-stream", action="store_true", help="Disable token streaming.")
    args = parser.parse_args()

    try:
        print(
            "Warning: `rl-intern` is deprecated for normal use. "
            "Use `cd frontend && bun run dev`.",
            file=sys.stderr,
        )
        if args.prompt:
            asyncio.run(
                _run_agent_prompt(
                    args.prompt,
                    model=args.model,
                    max_iterations=args.max_iterations,
                    stream=not args.no_stream,
                    headless=True,
                    runner=args.runner,
                )
            )
        else:
            asyncio.run(
                _interactive(
                    model=args.model,
                    max_iterations=args.max_iterations,
                    stream=not args.no_stream,
                    runner=args.runner,
                )
            )
    except KeyboardInterrupt:
        print("\nGoodbye.")


if __name__ == "__main__":
    cli()
