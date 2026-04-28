import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from rl_intern.run_store import RunStore

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_NOISY_PATTERNS = (
    "UserWarning",
    "DeprecationWarning",
    "FutureWarning",
    "pkg_resources",
    "IMAGEIO FFMPEG",
    "WARNING:imageio_ffmpeg",
)


def _is_noisy(line: str) -> bool:
    if line.startswith("WARNING:"):
        return True
    return any(p in line for p in _NOISY_PATTERNS)


def _looks_like_error(line: str) -> bool:
    lower = line.lower()
    return (
        line.startswith("ERROR")
        or line.startswith("CRITICAL")
        or "traceback" in lower
        or "error:" in lower
        or "exception" in lower
    )


class CreateRunRequest(BaseModel):
    run_id: str | None = None
    model: str | None = None
    prompt: str | None = None
    runner: str = "local"


def create_app(run_store: RunStore | None = None) -> FastAPI:
    store = run_store or RunStore()
    app = FastAPI(title="RL Intern Run Server")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/session")
    def create_session() -> dict[str, Any]:
        """Allocate a session id without spawning the agent yet.

        The frontend calls this from the welcome screen and then opens the
        chat WebSocket with `?session_id=<id>` — the WS spawns a fresh
        rl_intern.rpc subprocess and asks it to start a run with that id, so
        the browser's session id and the on-disk run id stay aligned.
        """
        return {
            "session_id": f"run_{uuid.uuid4().hex[:12]}",
            "created_at": _utc_now_iso(),
        }

    @app.websocket("/api/ws/chat")
    async def chat_ws(websocket: WebSocket) -> None:
        """Bridge a single browser session to one rl_intern.rpc subprocess.

        The browser speaks the same newline-delimited JSON protocol the
        existing rpc module already uses, so events flow through unchanged.
        Stderr from the subprocess is filtered for noise and forwarded as
        synthetic ``bridge_log`` events so the UI can surface real failures
        instead of hanging silently.
        """
        await websocket.accept()

        session_id = websocket.query_params.get("session_id")
        model = websocket.query_params.get("model")
        runner = websocket.query_params.get("runner")

        async def send_bridge(event_type: str, **data: Any) -> None:
            try:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": event_type,
                            "timestamp": _utc_now_iso(),
                            "data": data,
                        }
                    )
                )
            except Exception:
                pass

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "rl_intern.rpc",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:  # noqa: BLE001
            await send_bridge(
                "bridge_log",
                level="error",
                source="spawn",
                line=f"could not spawn rl_intern.rpc: {exc}",
            )
            try:
                await websocket.close()
            except Exception:
                pass
            return
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None

        await send_bridge("bridge_open", pid=proc.pid)

        async def pump_rpc_to_client() -> None:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    return
                try:
                    await websocket.send_text(line.decode("utf-8").rstrip("\n"))
                except Exception:
                    return

        async def pump_stderr() -> None:
            assert proc.stderr is not None
            while True:
                raw = await proc.stderr.readline()
                if not raw:
                    return
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n").strip()
                if not line or _is_noisy(line):
                    continue
                level = "error" if _looks_like_error(line) else "warn"
                await send_bridge("bridge_log", level=level, source="stderr", line=line)

        async def pump_client_to_rpc() -> None:
            try:
                while True:
                    raw = await websocket.receive_text()
                    payload = (raw + "\n").encode("utf-8")
                    proc.stdin.write(payload)
                    await proc.stdin.drain()
            except WebSocketDisconnect:
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("client→rpc pump error: %s", exc)

        start_payload: dict[str, Any] = {"type": "start_run", "id": "start"}
        if session_id:
            start_payload["run_id"] = session_id
        if model:
            start_payload["model"] = model
        if runner:
            start_payload["runner"] = runner
        proc.stdin.write((json.dumps(start_payload) + "\n").encode("utf-8"))
        await proc.stdin.drain()

        reader_task = asyncio.create_task(pump_rpc_to_client())
        stderr_task = asyncio.create_task(pump_stderr())
        writer_task = asyncio.create_task(pump_client_to_rpc())
        try:
            done, pending = await asyncio.wait(
                {reader_task, writer_task, stderr_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc and not isinstance(exc, (asyncio.CancelledError, WebSocketDisconnect)):
                    logger.exception("ws bridge task crashed: %s", exc)
        finally:
            try:
                proc.stdin.write(b'{"type":"shutdown","id":"shutdown"}\n')
                await proc.stdin.drain()
            except Exception:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3)
                except asyncio.TimeoutError:
                    proc.kill()
            await send_bridge(
                "bridge_exit",
                code=proc.returncode if proc.returncode is not None else -1,
            )
            try:
                await websocket.close()
            except Exception:
                pass

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
