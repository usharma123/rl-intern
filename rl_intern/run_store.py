import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_intern.events import to_json_line, utc_now_iso


DEFAULT_RUNS_ROOT = Path("artifacts") / "runs"


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_dir: Path
    session_path: Path
    metadata_path: Path


def make_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


class RunStore:
    def __init__(self, root: str | Path = DEFAULT_RUNS_ROOT):
        self.root = Path(root)

    def create_run(
        self,
        *,
        run_id: str | None = None,
        model: str | None = None,
        prompt: str | None = None,
        runner: str = "local",
    ) -> RunRecord:
        run_id = run_id or make_run_id()
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        record = RunRecord(
            run_id=run_id,
            run_dir=run_dir,
            session_path=run_dir / "session.jsonl",
            metadata_path=run_dir / "metadata.json",
        )
        existing = self.load_metadata(run_id)
        metadata = {
            **existing,
            "run_id": run_id,
            "created_at": existing.get("created_at", utc_now_iso()),
            "updated_at": utc_now_iso(),
            "model": model or existing.get("model"),
            "prompt": prompt if prompt is not None else existing.get("prompt"),
            "runner": runner or existing.get("runner", "local"),
            "run_dir": str(run_dir),
            "session_path": str(record.session_path),
        }
        self._atomic_json(record.metadata_path, metadata)
        record.session_path.touch(exist_ok=True)
        return record

    def update_metadata(self, run_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        record = self.get_run(run_id)
        metadata = self.load_metadata(run_id)
        metadata.update(updates)
        metadata["updated_at"] = utc_now_iso()
        self._atomic_json(record.metadata_path, metadata)
        return metadata

    def get_run(self, run_id: str) -> RunRecord:
        run_dir = self.root / run_id
        return RunRecord(
            run_id=run_id,
            run_dir=run_dir,
            session_path=run_dir / "session.jsonl",
            metadata_path=run_dir / "metadata.json",
        )

    def append_event(self, run_id: str, event: dict[str, Any]) -> None:
        record = self.get_run(run_id)
        record.run_dir.mkdir(parents=True, exist_ok=True)
        with record.session_path.open("a", encoding="utf-8") as f:
            f.write(to_json_line(event))
            f.flush()
            os.fsync(f.fileno())
        metadata = self.load_metadata(run_id)
        metadata["updated_at"] = utc_now_iso()
        metadata["last_event_type"] = event.get("type")
        self._atomic_json(record.metadata_path, metadata)

    def load_metadata(self, run_id: str) -> dict[str, Any]:
        record = self.get_run(run_id)
        if not record.metadata_path.exists():
            return {
                "run_id": run_id,
                "run_dir": str(record.run_dir),
                "session_path": str(record.session_path),
            }
        return json.loads(record.metadata_path.read_text(encoding="utf-8"))

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        runs = []
        for path in sorted(self.root.iterdir(), reverse=True):
            if not path.is_dir():
                continue
            metadata_path = path / "metadata.json"
            if metadata_path.exists():
                try:
                    runs.append(json.loads(metadata_path.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    runs.append({"run_id": path.name, "run_dir": str(path)})
            else:
                runs.append({"run_id": path.name, "run_dir": str(path)})
        return runs

    def read_events(self, run_id: str) -> list[dict[str, Any]]:
        record = self.get_run(run_id)
        if not record.session_path.exists():
            return []
        events = []
        for line in record.session_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"type": "malformed_jsonl", "raw": line})
        return events

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        record = self.get_run(run_id)
        if not record.run_dir.exists():
            return []
        artifacts = []
        for path in sorted(record.run_dir.rglob("*")):
            if path.is_file():
                artifacts.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "size_bytes": path.stat().st_size,
                        "relative_path": str(path.relative_to(record.run_dir)),
                    }
                )
        return artifacts

    def delete_run(self, run_id: str) -> bool:
        record = self.get_run(run_id)
        root = self.root.resolve()
        run_dir = record.run_dir.resolve()
        if not run_dir.exists():
            return False
        if root not in run_dir.parents:
            raise ValueError(f"Refusing to delete run outside root: {run_dir}")
        shutil.rmtree(run_dir)
        return True

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
