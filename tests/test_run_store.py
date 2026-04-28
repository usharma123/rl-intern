import json

from rl_intern.events import normalize_event
from rl_intern.run_store import RunStore


def test_run_store_creates_and_appends_jsonl(tmp_path):
    store = RunStore(tmp_path / "runs")
    record = store.create_run(run_id="run_test", model="test-model", prompt="hello")

    event = normalize_event(
        "tool_call",
        {"tool": "inspect_env", "tool_call_id": "call_1", "arguments": {"env_id": "CartPole-v1"}},
        run_id=record.run_id,
        turn_id="turn_001",
    )
    store.append_event(record.run_id, event)

    assert record.session_path.exists()
    lines = record.session_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["type"] == "tool_call"
    assert store.list_runs()[0]["run_id"] == "run_test"


def test_run_store_handles_malformed_jsonl(tmp_path):
    store = RunStore(tmp_path / "runs")
    record = store.create_run(run_id="run_bad")
    record.session_path.write_text("{bad json}\n", encoding="utf-8")

    events = store.read_events("run_bad")

    assert events[0]["type"] == "malformed_jsonl"
