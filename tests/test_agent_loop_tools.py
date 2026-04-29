import asyncio
from types import SimpleNamespace

from agent.core.agent_loop import _execute_tools
from agent.core.session import Session
from agent.core.tools import ToolRouter, ToolSpec


def test_execute_tools_records_failed_tool_output_when_handler_raises():
    async def raising_handler(args):
        raise ValueError("missing required field")

    async def run():
        event_queue = asyncio.Queue()
        router = ToolRouter()
        router.register_tool(
            ToolSpec(
                name="raising_tool",
                description="raises",
                parameters={"type": "object", "properties": {}},
                handler=raising_handler,
            )
        )
        session = Session(event_queue, tool_router=router)

        await _execute_tools(session, [(SimpleNamespace(id="call_1"), "raising_tool", {})])

        events = []
        while not event_queue.empty():
            events.append(await event_queue.get())
        return session, events

    session, events = asyncio.run(run())

    tool_messages = [
        message for message in session.context_manager.items if getattr(message, "role", None) == "tool"
    ]
    tool_outputs = [event for event in events if event.event_type == "tool_output"]

    assert tool_messages
    assert tool_messages[-1].tool_call_id == "call_1"
    assert "missing required field" in tool_messages[-1].content
    assert tool_outputs[-1].data["success"] is False
