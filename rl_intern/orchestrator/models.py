from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


StageName = Literal[
    "inspect",
    "prepare",
    "smoke_test",
    "train",
    "evaluate",
    "report",
    "publish_optional",
]
DomainName = Literal["gym_sb3", "llm_trl"]
RunnerBackend = Literal["local", "modal"]
JobStatus = Literal["pending", "running", "succeeded", "failed", "cancelled", "unknown"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class RewardSpec(BaseModel):
    type: Literal["environment", "python_verifier", "preference", "none"] = "none"
    description: str | None = None
    verifier_path: str | None = None
    verifier_source: str | None = None

    @model_validator(mode="after")
    def validate_verifier(self) -> "RewardSpec":
        if self.type == "python_verifier" and not (self.verifier_path or self.verifier_source):
            raise ValueError("python_verifier rewards require verifier_path or verifier_source")
        return self


class RunnerSpec(BaseModel):
    backend: RunnerBackend = "local"
    hardware: str = "cpu"
    image: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    timeout: str = "30m"
    env: dict[str, str] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)
    volume_name: str = "rl-intern-runs"
    remote_run_root: str = "/runs"

    @field_validator("timeout")
    @classmethod
    def timeout_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("timeout must not be empty")
        return value


class StageSpec(BaseModel):
    name: StageName
    status: Literal["pending", "in_progress", "completed", "failed", "skipped"] = "pending"
    requires_approval: bool = False
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ExperimentPlan(BaseModel):
    plan_id: str
    domain: DomainName
    objective: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    reward: RewardSpec = Field(default_factory=RewardSpec)
    runner: RunnerSpec = Field(default_factory=RunnerSpec)
    stages: list[StageSpec]
    expected_artifacts: list[str] = Field(default_factory=list)
    research_required: bool = False
    research_completed: bool = False
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    @field_validator("stages")
    @classmethod
    def stages_not_empty(cls, value: list[StageSpec]) -> list[StageSpec]:
        if not value:
            raise ValueError("plan requires at least one stage")
        return value

    @model_validator(mode="after")
    def validate_train_requirements(self) -> "ExperimentPlan":
        stage_names = {stage.name for stage in self.stages}
        if "train" in stage_names:
            if "inspect" not in stage_names:
                raise ValueError("train stage requires an inspect stage")
            if not self.expected_artifacts:
                raise ValueError("train stage requires expected_artifacts")
            if self.domain == "gym_sb3" and self.reward.type not in {"environment", "none"}:
                raise ValueError("gym_sb3 plans use environment rewards")
            if self.domain == "llm_trl":
                method = str(self.inputs.get("method", "")).lower()
                if method == "grpo" and self.reward.type != "python_verifier":
                    raise ValueError("llm_trl GRPO requires a python_verifier reward")
        return self


class ArtifactItem(BaseModel):
    kind: str
    path: str
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)


class ArtifactManifest(BaseModel):
    run_id: str
    domain: str | None = None
    plan_id: str | None = None
    checkpoints: list[ArtifactItem] = Field(default_factory=list)
    adapters: list[ArtifactItem] = Field(default_factory=list)
    metrics: list[ArtifactItem] = Field(default_factory=list)
    logs: list[ArtifactItem] = Field(default_factory=list)
    videos: list[ArtifactItem] = Field(default_factory=list)
    reports: list[ArtifactItem] = Field(default_factory=list)
    samples: list[ArtifactItem] = Field(default_factory=list)
    configs: list[ArtifactItem] = Field(default_factory=list)
    errors: list[ArtifactItem] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utc_now_iso)

    def add(self, bucket: str, item: ArtifactItem) -> None:
        if not hasattr(self, bucket):
            raise ValueError(f"Unknown artifact bucket: {bucket}")
        getattr(self, bucket).append(item)
        self.updated_at = utc_now_iso()


class JobRecord(BaseModel):
    job_id: str
    run_id: str
    stage: str
    backend: RunnerBackend
    backend_id: str | None = None
    status: JobStatus = "pending"
    hardware: str | None = None
    log_paths: dict[str, str] = Field(default_factory=dict)
    artifact_paths: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    error: str | None = None

    def write(self, run_dir: str | Path) -> Path:
        import json

        path = Path(run_dir) / "jobs" / f"{self.job_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
        return path
