from rl_intern.run_store import RunStore
from rl_intern.server.app import create_app


def test_run_server_endpoints(tmp_path):
    store = RunStore(tmp_path / "runs")
    record = store.create_run(run_id="run_server")
    store.append_event(record.run_id, {"type": "ready", "run_id": record.run_id})
    (record.run_dir / "report.md").write_text("# Report\n", encoding="utf-8")

    from fastapi.testclient import TestClient

    client = TestClient(create_app(store))

    assert client.get("/runs").status_code == 200
    detail = client.get("/runs/run_server")
    assert detail.status_code == 200
    assert detail.json()["metadata"]["run_id"] == "run_server"
    assert client.get("/runs/run_server/events.jsonl").text.strip()
    assert client.get("/runs/run_server/artifacts").status_code == 200
    assert client.get("/runs/run_server/report.md").text == "# Report\n"
    assert "Euphony" in client.get("/runs/run_server/viewer").text
