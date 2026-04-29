from rl_intern.run_store import RunStore
from rl_intern.server.app import create_app


def test_run_server_endpoints(tmp_path):
    store = RunStore(tmp_path / "runs")
    record = store.create_run(run_id="run_server")
    store.append_event(record.run_id, {"type": "ready", "run_id": record.run_id})
    (record.run_dir / "report.md").write_text("# Report\n", encoding="utf-8")

    from fastapi.testclient import TestClient

    client = TestClient(create_app(store))

    setup = client.get("/api/setup/status")
    assert setup.status_code == 200
    assert "openrouter" in setup.json()
    assert client.get("/runs").status_code == 200
    session = client.post("/api/session")
    assert session.status_code == 200
    session_id = session.json()["session_id"]
    assert client.get(f"/runs/{session_id}/events.jsonl").status_code == 200
    created = client.post("/runs", json={"run_id": "run_modal_server", "runner": "modal"})
    assert created.status_code == 200
    assert created.json()["runner"] == "modal"
    detail = client.get("/runs/run_server")
    assert detail.status_code == 200
    assert detail.json()["metadata"]["run_id"] == "run_server"
    assert client.get("/runs/run_server/events.jsonl").text.strip()
    assert client.get("/runs/run_server/artifacts").status_code == 200
    assert client.get("/runs/run_server/report.md").text == "# Report\n"
    assert "Euphony" in client.get("/runs/run_server/viewer").text
    deleted = client.delete("/runs/run_server")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert client.get("/runs/run_server").status_code == 404
