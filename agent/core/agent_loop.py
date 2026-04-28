import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from litellm import ChatCompletionMessageToolCall, Message, acompletion

from agent.config import Config
from agent.core.doom_loop import check_for_doom_loop
from agent.core.session import Event, OpType, Session
from agent.core.tools import ToolRouter

logger = logging.getLogger(__name__)

ToolCall = ChatCompletionMessageToolCall


def _resolve_llm_params(model_name: str) -> dict[str, Any]:
    return {"model": model_name}


def _validate_tool_args(tool_args: dict[str, Any]) -> tuple[bool, str | None]:
    args = tool_args if tool_args is not None else {}
    if isinstance(args, str):
        return False, "'arguments' must be a JSON object, not a string."
    if not isinstance(args, dict):
        return False, f"'arguments' must be a JSON object, got {type(args).__name__}."
    return True, None


def _needs_approval(
    tool_name: str,
    config: Config | None = None,
    args: dict[str, Any] | None = None,
) -> bool:
    if config and config.yolo_mode:
        return False
    args = args or {}
    if tool_name == "run_experiment_stage":
        return args.get("stage") in {"train", "publish_optional"}
    return tool_name in {
        "train_sb3",
        "launch_modal_experiment",
        "modal_job_run",
        "modal_job_cancel",
        "modal_sandbox_create",
        "modal_sandbox_exec",
        "modal_sandbox_write",
        "modal_sandbox_edit",
        "modal_sandbox_terminate",
    }


def _runner_user_text(session: Session, text: str) -> str:
    if getattr(session.config, "runner", "local") != "modal":
        return text
    return (
        "Use the Modal remote runner for trusted SB3 execution. "
        "For training/evaluation/report requests, call launch_modal_experiment, "
        "then get_modal_run_status, then fetch_modal_artifacts before reporting success. "
        "Do not use local train_sb3/evaluate_policy/record_rollout/generate_report unless "
        "the Modal tool reports an error that requires local fallback. "
        f"User request: {text}"
    )


def _friendly_error_message(error: Exception) -> str | None:
    err_str = str(error).lower()
    if "authentication" in err_str or "unauthorized" in err_str or "api key" in err_str:
        return (
            "Authentication failed. Set an API key for your selected LiteLLM provider "
            "or use --model with a configured provider."
        )
    if "rate limit" in err_str or "429" in err_str:
        return "The model provider rate limited this request. Retry later or switch models."
    return None


def _display_tool_name(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name == "run_experiment_stage" and args.get("stage"):
        return f"stage:{args['stage']}"
    return tool_name


async def _compact_and_notify(session: Session) -> None:
    old_length = session.context_manager.context_length
    await session.context_manager.compact(
        model_name=session.config.model_name,
        tool_specs=session.tool_router.get_tool_specs_for_llm(),
    )
    new_length = session.context_manager.context_length
    if new_length != old_length:
        await session.send_event(
            Event(
                event_type="compacted",
                data={"old_tokens": old_length, "new_tokens": new_length},
            )
        )


async def _cleanup_on_cancel(session: Session) -> None:
    return None


@dataclass
class LLMResult:
    content: str | None
    tool_calls_acc: dict[int, dict[str, Any]]
    token_count: int
    finish_reason: str | None


async def _call_llm_streaming(session: Session, messages, tools, llm_params) -> LLMResult:
    response = await acompletion(
        **llm_params,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=True,
    )
    content_parts: list[str] = []
    tool_calls_acc: dict[int, dict[str, Any]] = {}
    token_count = session.context_manager.context_length
    finish_reason = None

    async for chunk in response:
        choice = chunk.choices[0]
        delta = choice.delta
        finish_reason = choice.finish_reason or finish_reason
        if delta.content:
            content_parts.append(delta.content)
            await session.send_event(
                Event(event_type="assistant_chunk", data={"content": delta.content})
            )
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc_delta.id:
                    tool_calls_acc[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_calls_acc[idx]["function"]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_calls_acc[idx]["function"]["arguments"] += (
                            tc_delta.function.arguments
                        )
        usage = getattr(chunk, "usage", None)
        if usage and getattr(usage, "total_tokens", None):
            token_count = usage.total_tokens

    return LLMResult(
        content="".join(content_parts) or None,
        tool_calls_acc=tool_calls_acc,
        token_count=token_count,
        finish_reason=finish_reason,
    )


async def _call_llm_non_streaming(session: Session, messages, tools, llm_params) -> LLMResult:
    response = await acompletion(
        **llm_params,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=False,
    )
    choice = response.choices[0]
    message = choice.message
    tool_calls_acc: dict[int, dict[str, Any]] = {}
    if message.tool_calls:
        for idx, tc in enumerate(message.tool_calls):
            tool_calls_acc[idx] = {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
    usage = getattr(response, "usage", None)
    token_count = getattr(usage, "total_tokens", session.context_manager.context_length)
    return LLMResult(
        content=message.content,
        tool_calls_acc=tool_calls_acc,
        token_count=token_count,
        finish_reason=choice.finish_reason,
    )


class Handlers:
    @staticmethod
    async def _abandon_pending_approval(session: Session) -> None:
        tool_calls = session.pending_approval.get("tool_calls", [])
        for tc in tool_calls:
            tool_msg = Message(
                role="tool",
                content="Task abandoned; user continued without approving.",
                tool_call_id=tc.id,
                name=tc.function.name,
            )
            session.context_manager.add_message(tool_msg)
            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": tc.function.name,
                        "state": "abandoned",
                    },
                )
            )
        session.pending_approval = None

    @staticmethod
    async def run_agent(session: Session, text: str) -> str | None:
        session.reset_cancel()
        if text:
            session.current_turn_id = f"turn_{session.turn_count + 1:03d}_{uuid.uuid4().hex[:6]}"
        if text and session.pending_approval:
            await Handlers._abandon_pending_approval(session)
        if text:
            session.context_manager.add_message(Message(role="user", content=text))
            await session.send_event(Event(event_type="user_input", data={"text": text}))

        await session.send_event(
            Event(event_type="processing", data={"message": "Processing user input"})
        )

        iteration = 0
        final_response = None
        errored = False
        max_iterations = session.config.max_iterations

        while max_iterations == -1 or iteration < max_iterations:
            if session.is_cancelled:
                break
            await _compact_and_notify(session)
            doom_prompt = check_for_doom_loop(session.context_manager.items)
            if doom_prompt:
                session.context_manager.add_message(Message(role="user", content=doom_prompt))
                await session.send_event(
                    Event(
                        event_type="tool_log",
                        data={"tool": "system", "log": "Repeated tool pattern detected."},
                    )
                )

            try:
                llm_params = _resolve_llm_params(session.config.model_name)
                if session.stream:
                    llm_result = await _call_llm_streaming(
                        session,
                        session.context_manager.get_messages(),
                        session.tool_router.get_tool_specs_for_llm(),
                        llm_params,
                    )
                else:
                    llm_result = await _call_llm_non_streaming(
                        session,
                        session.context_manager.get_messages(),
                        session.tool_router.get_tool_specs_for_llm(),
                        llm_params,
                    )

                if session.stream:
                    await session.send_event(Event(event_type="assistant_stream_end", data={}))

                tool_calls = [
                    ToolCall(
                        id=tc_data["id"],
                        type="function",
                        function={
                            "name": tc_data["function"]["name"],
                            "arguments": tc_data["function"]["arguments"],
                        },
                    )
                    for _idx, tc_data in sorted(llm_result.tool_calls_acc.items())
                    if tc_data["function"]["name"]
                ]

                if not tool_calls:
                    if llm_result.content:
                        session.context_manager.add_message(
                            Message(role="assistant", content=llm_result.content),
                            llm_result.token_count,
                        )
                        final_response = llm_result.content
                    break

                session.context_manager.add_message(
                    Message(
                        role="assistant",
                        content=llm_result.content,
                        tool_calls=tool_calls,
                    ),
                    llm_result.token_count,
                )

                good_tools: list[tuple[ToolCall, str, dict[str, Any]]] = []
                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except (json.JSONDecodeError, TypeError) as exc:
                        msg = f"Malformed arguments for {tc.function.name}: {exc}"
                        session.context_manager.add_message(
                            Message(
                                role="tool",
                                content=msg,
                                tool_call_id=tc.id,
                                name=tc.function.name,
                            )
                        )
                        await session.send_event(
                            Event(
                                event_type="tool_output",
                                data={
                                    "tool": tc.function.name,
                                    "tool_call_id": tc.id,
                                    "output": msg,
                                    "success": False,
                                },
                            )
                        )
                        continue
                    valid, err = _validate_tool_args(args)
                    if not valid:
                        session.context_manager.add_message(
                            Message(role="tool", content=err, tool_call_id=tc.id, name=tc.function.name)
                        )
                        continue
                    good_tools.append((tc, tc.function.name, args))

                approval_required = [
                    item
                    for item in good_tools
                    if _needs_approval(item[1], session.config, item[2])
                ]
                executable = [
                    item
                    for item in good_tools
                    if not _needs_approval(item[1], session.config, item[2])
                ]

                await _execute_tools(session, executable)

                if approval_required:
                    await session.send_event(
                        Event(
                            event_type="approval_required",
                            data={
                                "tools": [
                                    {
                                        "tool": name,
                                        "arguments": args,
                                        "tool_call_id": tc.id,
                                    }
                                    for tc, name, args in approval_required
                                ],
                                "count": len(approval_required),
                            },
                        )
                    )
                    session.pending_approval = {
                        "tool_calls": [tc for tc, _name, _args in approval_required],
                    }
                    return None

                iteration += 1
            except Exception as exc:
                error_msg = _friendly_error_message(exc) or str(exc)
                await session.send_event(
                    Event(event_type="error", data={"error": error_msg})
                )
                errored = True
                break

        if session.is_cancelled:
            await _cleanup_on_cancel(session)
            await session.send_event(Event(event_type="interrupted"))
        elif not errored:
            await session.send_event(
                Event(
                    event_type="turn_complete",
                    data={"history_size": len(session.context_manager.items)},
                )
            )

        session.increment_turn()
        await session.auto_save_if_needed()
        return final_response

    @staticmethod
    async def undo(session: Session) -> None:
        session.context_manager.undo_last_turn()
        await session.send_event(Event(event_type="undo_complete"))

    @staticmethod
    async def exec_approval(session: Session, approvals: list[dict[str, Any]]) -> None:
        if not session.pending_approval:
            await session.send_event(
                Event(event_type="error", data={"error": "No pending approval."})
            )
            return

        approval_map = {item["tool_call_id"]: item for item in approvals}
        approved = []
        rejected = []
        for tc in session.pending_approval.get("tool_calls", []):
            decision = approval_map.get(tc.id, {"approved": False})
            if decision.get("approved", False):
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    args = {"error": f"Malformed approved arguments: {exc}"}
                approved.append((tc, tc.function.name, args))
            else:
                rejected.append((tc, tc.function.name, decision))

        session.pending_approval = None

        for tc, name, _args in approved:
            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={"tool_call_id": tc.id, "tool": name, "state": "approved"},
                )
            )
        for tc, name, decision in rejected:
            msg = "Tool execution cancelled by user"
            if decision.get("feedback"):
                msg += f". User feedback: {decision['feedback']}"
            session.context_manager.add_message(
                Message(role="tool", content=msg, tool_call_id=tc.id, name=name)
            )
            await session.send_event(
                Event(
                    event_type="tool_output",
                    data={"tool": name, "tool_call_id": tc.id, "output": msg, "success": False},
                )
            )

        await _execute_tools(session, approved, state_events=True)
        await Handlers.run_agent(session, "")

    @staticmethod
    async def shutdown(session: Session) -> bool:
        if session.config.save_sessions:
            session.save_trajectory_local()
        session.is_running = False
        await session.send_event(Event(event_type="shutdown"))
        return True


async def _execute_tools(
    session: Session,
    tools: list[tuple[ToolCall, str, dict[str, Any]]],
    state_events: bool = False,
) -> None:
    if not tools:
        return

    prepared_tools = []
    for tc, name, args in tools:
        if session.run_dir and name in {
            "train_sb3",
            "evaluate_policy",
            "record_rollout",
            "generate_report",
            "launch_modal_experiment",
            "get_modal_run_status",
            "fetch_modal_artifacts",
            "create_experiment_plan",
            "run_experiment_stage",
            "get_artifact_manifest",
            "modal_job_run",
            "modal_job_status",
            "modal_job_logs",
            "modal_job_artifacts",
            "modal_job_cancel",
        }:
            args = {**args, "run_dir": session.run_dir}
        prepared_tools.append((tc, name, args))
        if state_events:
            await session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": tc.id,
                        "tool": _display_tool_name(name, args),
                        "actual_tool": name,
                        "state": "running",
                    },
                )
            )
        await session.send_event(
            Event(
                event_type="tool_call",
                data={
                    "tool": _display_tool_name(name, args),
                    "actual_tool": name,
                    "arguments": args,
                    "tool_call_id": tc.id,
                },
            )
        )

    async def run_one(tc: ToolCall, name: str, args: dict[str, Any]):
        output, success = await session.tool_router.call_tool(
            name,
            args,
            session=session,
            tool_call_id=tc.id,
        )
        return tc, name, args, output, success

    results = await asyncio.gather(
        *[run_one(tc, name, args) for tc, name, args in prepared_tools],
        return_exceptions=True,
    )

    for result in results:
        if isinstance(result, Exception):
            logger.exception("Tool execution failed", exc_info=result)
            continue
        tc, name, args, output, success = result
        session.context_manager.add_message(
            Message(role="tool", content=output, tool_call_id=tc.id, name=name)
        )
        await session.send_event(
            Event(
                event_type="tool_output",
                data={
                    "tool": _display_tool_name(name, args),
                    "actual_tool": name,
                    "tool_call_id": tc.id,
                    "output": output,
                    "success": success,
                },
            )
        )


async def process_submission(session: Session, submission) -> bool:
    op = submission.operation
    if op.op_type == OpType.USER_INPUT:
        text = op.data.get("text", "") if op.data else ""
        await Handlers.run_agent(session, _runner_user_text(session, text))
        return True
    if op.op_type == OpType.COMPACT:
        await _compact_and_notify(session)
        return True
    if op.op_type == OpType.UNDO:
        await Handlers.undo(session)
        return True
    if op.op_type == OpType.EXEC_APPROVAL:
        approvals = op.data.get("approvals", []) if op.data else []
        await Handlers.exec_approval(session, approvals)
        return True
    if op.op_type == OpType.SHUTDOWN:
        return not await Handlers.shutdown(session)
    logger.warning("Unknown operation: %s", op.op_type)
    return True


async def submission_loop(
    submission_queue: asyncio.Queue,
    event_queue: asyncio.Queue,
    config: Config | None = None,
    tool_router: ToolRouter | None = None,
    session_holder: list | None = None,
    local_mode: bool = False,
    stream: bool = True,
    run_id: str | None = None,
    run_dir: str | None = None,
    run_store: Any = None,
    **_: Any,
) -> None:
    tool_router = tool_router or ToolRouter(local_mode=local_mode)
    session = Session(
        event_queue,
        config=config,
        tool_router=tool_router,
        local_mode=local_mode,
        stream=stream,
        run_id=run_id,
        run_dir=run_dir,
        run_store=run_store,
    )
    if session_holder is not None:
        session_holder[0] = session

    async with tool_router:
        await session.send_event(
            Event(
                event_type="ready",
                data={"message": "Agent initialized", "tool_count": len(tool_router.tools)},
            )
        )
        while session.is_running:
            submission = await submission_queue.get()
            try:
                should_continue = await process_submission(session, submission)
                if not should_continue:
                    break
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Error in agent loop")
                await session.send_event(Event(event_type="error", data={"error": str(exc)}))
