from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from rl_intern.run_store import RunStore


class CreateRunRequest(BaseModel):
    run_id: str | None = None
    model: str | None = None
    prompt: str | None = None
    runner: str = "local"


def create_app(run_store: RunStore | None = None) -> FastAPI:
    store = run_store or RunStore()
    app = FastAPI(title="RL Intern Run Server")

    @app.post("/runs")
    def create_run(request: CreateRunRequest) -> dict[str, Any]:
        record = store.create_run(
            run_id=request.run_id,
            model=request.model,
            prompt=request.prompt,
            runner=request.runner,
        )
        return store.load_metadata(record.run_id)

    @app.get("/runs")
    def list_runs() -> list[dict[str, Any]]:
        return store.list_runs()

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        metadata = store.load_metadata(run_id)
        if not Path(metadata["run_dir"]).exists():
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "metadata": metadata,
            "artifacts": store.list_artifacts(run_id),
            "event_count": len(store.read_events(run_id)),
        }

    @app.get("/runs/{run_id}/events.jsonl")
    def events_jsonl(run_id: str) -> PlainTextResponse:
        record = store.get_run(run_id)
        if not record.session_path.exists():
            raise HTTPException(status_code=404, detail="Run log not found")
        return PlainTextResponse(
            record.session_path.read_text(encoding="utf-8"),
            media_type="application/x-ndjson",
        )

    @app.get("/runs/{run_id}/artifacts")
    def artifacts(run_id: str) -> list[dict[str, Any]]:
        record = store.get_run(run_id)
        if not record.run_dir.exists():
            raise HTTPException(status_code=404, detail="Run not found")
        return store.list_artifacts(run_id)

    @app.get("/runs/{run_id}/report.md")
    def report(run_id: str):
        report_path = store.get_run(run_id).run_dir / "report.md"
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")
        return FileResponse(report_path, media_type="text/markdown")

    @app.get("/runs/{run_id}/viewer")
    def viewer(run_id: str) -> HTMLResponse:
        record = store.get_run(run_id)
        if not record.session_path.exists():
            raise HTTPException(status_code=404, detail="Run log not found")
        template_path = Path(__file__).resolve().parents[1] / "viewer" / "euphony_embed.html"
        html = template_path.read_text(encoding="utf-8").replace("__RUN_ID__", run_id)
        return HTMLResponse(html)

    return app


app = create_app()
