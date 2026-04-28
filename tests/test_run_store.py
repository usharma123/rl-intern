import json
from concurrent.futures import ThreadPoolExecutor

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


def test_run_store_create_run_is_safe_for_same_run_id_race(tmp_path):
    store = RunStore(tmp_path / "runs")

    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(
            executor.map(
                lambda _: store.create_run(run_id="run_race", model="test-model"),
                range(32),
            )
        )

    assert {record.run_id for record in records} == {"run_race"}
    metadata = store.load_metadata("run_race")
    assert metadata["run_id"] == "run_race"
    assert metadata["model"] == "test-model"
    assert not list((tmp_path / "runs" / "run_race").glob("*.tmp"))


def test_create_run_preserves_existing_metadata_and_session_log(tmp_path):
    store = RunStore(tmp_path / "runs")
    first = store.create_run(run_id="run_persist", model="model-a", prompt="first")
    store.append_event("run_persist", {"type": "user_input", "content": "hello"})

    second = store.create_run(run_id="run_persist", model="model-b", prompt=None)

    metadata = store.load_metadata("run_persist")
    assert second.session_path == first.session_path
    assert metadata["created_at"]
    assert metadata["model"] == "model-b"
    assert metadata["prompt"] == "first"
    assert len(store.read_events("run_persist")) == 1


def test_delete_run_removes_local_directory(tmp_path):
    store = RunStore(tmp_path / "runs")
    record = store.create_run(run_id="run_delete")
    (record.run_dir / "artifact.txt").write_text("x", encoding="utf-8")

    assert store.delete_run("run_delete") is True
    assert not record.run_dir.exists()
    assert store.delete_run("run_delete") is False


def test_deleted_run_ignores_late_events_until_recreated(tmp_path):
    store = RunStore(tmp_path / "runs")
    record = store.create_run(run_id="run_late_delete")

    assert store.delete_run(record.run_id) is True
    store.append_event(record.run_id, {"type": "shutdown"})

    assert not record.run_dir.exists()
    assert store.list_runs() == []
    assert store.read_events(record.run_id) == []

    recreated = store.create_run(run_id=record.run_id)
    store.append_event(recreated.run_id, {"type": "ready"})

    assert store.list_runs()[0]["run_id"] == record.run_id
    assert store.read_events(record.run_id)[0]["type"] == "ready"
