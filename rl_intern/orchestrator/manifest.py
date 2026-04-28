from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rl_intern.orchestrator.models import ArtifactItem, ArtifactManifest, utc_now_iso


def manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "artifact_manifest.json"


def load_manifest(run_dir: str | Path, run_id: str | None = None) -> ArtifactManifest:
    path = manifest_path(run_dir)
    if path.exists():
        return ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))
    return ArtifactManifest(run_id=run_id or Path(run_dir).name)


def write_manifest(run_dir: str | Path, manifest: ArtifactManifest) -> Path:
    manifest.updated_at = utc_now_iso()
    path = manifest_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def append_manifest_item(
    run_dir: str | Path,
    bucket: str,
    path: str | Path,
    *,
    kind: str | None = None,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
    run_id: str | None = None,
    domain: str | None = None,
    plan_id: str | None = None,
) -> ArtifactManifest:
    manifest = load_manifest(run_dir, run_id=run_id)
    if domain:
        manifest.domain = domain
    if plan_id:
        manifest.plan_id = plan_id
    item = ArtifactItem(
        kind=kind or bucket.rstrip("s"),
        path=str(path),
        name=name or Path(path).name,
        metadata=metadata or {},
    )
    manifest.add(bucket, item)
    write_manifest(run_dir, manifest)
    return manifest
