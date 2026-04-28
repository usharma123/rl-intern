import asyncio

from agent.core.session import Session
from rl_intern.events import normalize_event
from rl_intern.run_store import RunStore


def test_session_restores_context_from_jsonl(tmp_path):
    store = RunStore(tmp_path / "runs")
    store.create_run(run_id="run_restore")
    store.append_event(
        "run_restore",
        normalize_event(
            "user_input",
            {"text": "inspect CartPole"},
            run_id="run_restore",
            turn_id="turn_001",
        ),
    )
    store.append_event(
        "run_restore",
        normalize_event(
            "assistant_chunk",
            {"content": "I will inspect it."},
            run_id="run_restore",
            turn_id="turn_001",
        ),
    )
    store.append_event(
        "run_restore",
        normalize_event(
            "tool_call",
            {
                "tool": "stage:inspect",
                "actual_tool": "run_experiment_stage",
                "tool_call_id": "call_1",
                "arguments": {"stage": "inspect"},
            },
            run_id="run_restore",
            turn_id="turn_001",
        ),
    )
    store.append_event(
        "run_restore",
        normalize_event(
            "tool_output",
            {
                "tool": "stage:inspect",
                "actual_tool": "run_experiment_stage",
                "tool_call_id": "call_1",
                "output": '{"ok": true}',
                "success": True,
            },
            run_id="run_restore",
            turn_id="turn_001",
        ),
    )
    store.append_event(
        "run_restore",
        normalize_event(
            "assistant_chunk",
            {"content": "Now I will summarize."},
            run_id="run_restore",
            turn_id="turn_001",
        ),
    )
    store.append_event(
        "run_restore",
        normalize_event(
            "tool_call",
            {
                "tool": "get_artifact_manifest",
                "actual_tool": "get_artifact_manifest",
                "tool_call_id": "call_2",
                "arguments": {"run_dir": "x"},
            },
            run_id="run_restore",
            turn_id="turn_001",
        ),
    )
    store.append_event(
        "run_restore",
        normalize_event(
            "tool_output",
            {
                "tool": "get_artifact_manifest",
                "actual_tool": "get_artifact_manifest",
                "tool_call_id": "call_2",
                "output": '{"artifacts": []}',
                "success": True,
            },
            run_id="run_restore",
            turn_id="turn_001",
        ),
    )
    store.append_event(
        "run_restore",
        normalize_event("turn_complete", {}, run_id="run_restore", turn_id="turn_001"),
    )

    session = Session(asyncio.Queue(), run_id="run_restore", run_store=store)
    messages = session.context_manager.items

    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert messages[1].content == "inspect CartPole"
    assert messages[2].tool_calls[0].function.name == "run_experiment_stage"
    assert messages[3].tool_call_id == "call_1"
    assert messages[4].tool_calls[0].function.name == "get_artifact_manifest"
    assert messages[5].tool_call_id == "call_2"
