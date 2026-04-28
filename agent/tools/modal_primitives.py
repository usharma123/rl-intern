from __future__ import annotations

import asyncio
from typing import Any

from agent.core.session import Event
from agent.tools.common import json_ready
from rl_intern.runners.modal_backend import (
    cancel_modal_job,
    fetch_modal_job_artifacts,
    get_modal_job_logs,
    get_modal_job_status,
    run_modal_job,
)
from rl_intern.runners.modal_sandbox import (
    create_sandbox,
    edit_sandbox,
    exec_sandbox,
    read_sandbox,
    terminate_sandbox,
    write_sandbox,
)


async def modal_sandbox_create_handler(args: dict[str, Any], session: Any = None, **_: Any) -> tuple[str, bool]:
    result = await asyncio.to_thread(create_sandbox, **args)
    if session:
        await session.send_event(Event(event_type="job_update", data={"kind": "sandbox", **result}))
    return json_ready(result), result.get("status") not in {"error", "failed"}


async def modal_sandbox_exec_handler(args: dict[str, Any], session: Any = None, **_: Any) -> tuple[str, bool]:
    result = await asyncio.to_thread(exec_sandbox, **args)
    if session:
        await session.send_event(Event(event_type="job_update", data={"kind": "sandbox_exec", **result}))
    return json_ready(result), result.get("status") == "succeeded"


async def modal_sandbox_read_handler(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
    result = await asyncio.to_thread(read_sandbox, **args)
    return json_ready(result), result.get("status") == "succeeded"


async def modal_sandbox_write_handler(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
    result = await asyncio.to_thread(write_sandbox, **args)
    return json_ready(result), result.get("status") == "succeeded"


async def modal_sandbox_edit_handler(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
    result = await asyncio.to_thread(edit_sandbox, **args)
    return json_ready(result), result.get("status") == "succeeded"


async def modal_sandbox_terminate_handler(args: dict[str, Any], session: Any = None, **_: Any) -> tuple[str, bool]:
    result = await asyncio.to_thread(terminate_sandbox, **args)
    if session:
        await session.send_event(Event(event_type="job_update", data={"kind": "sandbox", **result}))
    return json_ready(result), result.get("status") == "terminated"


async def modal_job_run_handler(args: dict[str, Any], session: Any = None, **_: Any) -> tuple[str, bool]:
    result = await asyncio.to_thread(run_modal_job, **args)
    if session:
        await session.send_event(Event(event_type="job_update", data=result))
    return json_ready(result), result.get("error") is None and result.get("status") != "failed"


async def modal_job_status_handler(args: dict[str, Any], session: Any = None, **_: Any) -> tuple[str, bool]:
    result = await asyncio.to_thread(get_modal_job_status, **args)
    if session:
        await session.send_event(Event(event_type="job_update", data=result))
    return json_ready(result), result.get("status") != "failed"


async def modal_job_logs_handler(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
    result = await asyncio.to_thread(get_modal_job_logs, **args)
    return json_ready(result), result.get("status") != "failed"


async def modal_job_cancel_handler(args: dict[str, Any], session: Any = None, **_: Any) -> tuple[str, bool]:
    result = await asyncio.to_thread(cancel_modal_job, **args)
    if session:
        await session.send_event(Event(event_type="job_update", data=result))
    return json_ready(result), result.get("status") == "cancelled"


async def modal_job_artifacts_handler(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
    result = await asyncio.to_thread(fetch_modal_job_artifacts, **args)
    return json_ready(result), result.get("status") != "failed"
