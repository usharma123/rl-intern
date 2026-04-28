from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from agent.tools.common import json_ready
from agent.core.session import Event
from rl_intern.orchestrator import ExperimentPlan, get_adapter
from rl_intern.orchestrator.adapters import get_registry
from rl_intern.orchestrator.manifest import load_manifest, write_manifest
from rl_intern.orchestrator.models import RewardSpec, RunnerSpec, StageSpec


def create_experiment_plan(
    domain: str,
    objective: str,
    inputs: dict[str, Any],
    stages: list[str] | None = None,
    reward: dict[str, Any] | None = None,
    runner: dict[str, Any] | None = None,
    expected_artifacts: list[str] | None = None,
    research_required: bool = False,
    research_completed: bool = False,
    run_dir: str | None = None,
) -> dict[str, Any]:
    inputs = _normalize_inputs(domain, inputs)
    plan = ExperimentPlan(
        plan_id=f"plan_{uuid.uuid4().hex[:12]}",
        domain=domain,
        objective=objective,
        inputs=inputs,
        reward=RewardSpec.model_validate(reward or _default_reward(domain, inputs)),
        runner=RunnerSpec.model_validate(_normalize_runner(runner, domain)),
        stages=[StageSpec(name=stage) for stage in (stages or _default_stages())],
        expected_artifacts=expected_artifacts or _default_artifacts(domain, inputs),
        research_required=research_required,
        research_completed=research_completed,
    )
    result = plan.model_dump()
    if run_dir:
        path = Path(run_dir) / "experiment_plan.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        result["plan_path"] = str(path)
        manifest = load_manifest(run_dir)
        manifest.domain = domain
        manifest.plan_id = plan.plan_id
        write_manifest(run_dir, manifest)
    return result


def validate_experiment_plan(plan: dict[str, Any], run_dir: str | None = None) -> dict[str, Any]:
    try:
        validated = ExperimentPlan.model_validate(_resolve_plan(plan, run_dir))
        adapter = get_adapter(validated.domain)
        return {
            "valid": True,
            "plan": validated.model_dump(),
            "domain": validated.domain,
            "artifact_schema": adapter.artifact_schema(),
        }
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def run_experiment_stage(
    plan: dict[str, Any],
    stage: str,
    run_dir: str | None = None,
) -> dict[str, Any]:
    resolved_plan = _resolve_plan(plan, run_dir)
    validated = ExperimentPlan.model_validate(resolved_plan)
    adapter = get_adapter(validated.domain)
    if not hasattr(adapter, stage):
        return {"error": f"Unknown adapter stage: {stage}"}
    method = getattr(adapter, stage)
    result = method(validated, run_dir=run_dir)
    return {"stage": stage, "domain": validated.domain, "result": result}


def get_artifact_manifest(run_dir: str) -> dict[str, Any]:
    return load_manifest(run_dir).model_dump()


def list_domain_adapters() -> dict[str, Any]:
    registry = get_registry()
    return {
        "domains": registry.list_domains(),
        "artifact_schemas": {
            domain: registry.get(domain).artifact_schema() for domain in registry.list_domains()
        },
    }


def _default_stages() -> list[str]:
    return ["inspect", "prepare", "smoke_test", "train", "evaluate", "report"]


def _resolve_plan(plan: dict[str, Any], run_dir: str | None = None) -> dict[str, Any]:
    candidate = _normalize_plan(plan)
    if _has_required_plan_fields(candidate):
        return candidate
    saved = _load_saved_plan(run_dir, candidate.get("plan_path"))
    if saved:
        merged = {**saved, **candidate}
        merged["inputs"] = {**saved.get("inputs", {}), **candidate.get("inputs", {})}
        return _normalize_plan(merged)
    return candidate


def _load_saved_plan(run_dir: str | None, plan_path: str | None = None) -> dict[str, Any] | None:
    paths = []
    if plan_path:
        paths.append(Path(plan_path))
    if run_dir:
        paths.append(Path(run_dir) / "experiment_plan.json")
    for path in paths:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _has_required_plan_fields(plan: dict[str, Any]) -> bool:
    return all(key in plan for key in ("plan_id", "domain", "objective", "stages", "expected_artifacts"))


def _normalize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        return plan
    normalized = dict(plan)
    domain = str(normalized.get("domain", ""))
    if isinstance(normalized.get("inputs"), dict):
        normalized["inputs"] = _normalize_inputs(domain, normalized["inputs"])
    if isinstance(normalized.get("stages"), list) and normalized["stages"] and isinstance(normalized["stages"][0], str):
        normalized["stages"] = [{"name": stage} for stage in normalized["stages"]]
    if "reward" not in normalized and domain:
        normalized["reward"] = _default_reward(domain, normalized.get("inputs", {}))
    normalized["runner"] = _normalize_runner(normalized.get("runner"), domain)
    if "expected_artifacts" not in normalized and domain:
        normalized["expected_artifacts"] = _default_artifacts(domain, normalized.get("inputs", {}))
    return normalized


def _normalize_runner(runner: Any, domain: str) -> dict[str, Any]:
    normalized = dict(runner or {}) if isinstance(runner, dict) else {}
    hardware = str(normalized.get("hardware", "")).lower()
    if "backend" not in normalized and (hardware.startswith("gpu") or "modal" in hardware):
        normalized["backend"] = "modal"
    if domain == "llm_trl" and hardware.startswith("gpu") and "backend" not in normalized:
        normalized["backend"] = "modal"
    return normalized


def _normalize_inputs(domain: str, inputs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(inputs)
    if domain == "llm_trl":
        if "dataset" not in normalized and "dataset_name" in normalized:
            normalized["dataset"] = normalized["dataset_name"]
        if "dataset" not in normalized and "dataset_path" in normalized:
            normalized["dataset"] = normalized["dataset_path"]
        if "model" not in normalized and "model_name" in normalized:
            normalized["model"] = normalized["model_name"]
        if "split" not in normalized and "dataset_split" in normalized:
            normalized["split"] = normalized["dataset_split"]
    if domain == "gym_sb3":
        if "env_id" not in normalized and "environment" in normalized:
            normalized["env_id"] = normalized["environment"]
    return normalized


def _default_reward(domain: str, inputs: dict[str, Any]) -> dict[str, Any]:
    if domain == "gym_sb3":
        return {"type": "environment"}
    if domain == "llm_trl" and str(inputs.get("method", "")).lower() == "grpo":
        return {"type": "python_verifier"}
    if domain == "llm_trl" and str(inputs.get("method", "")).lower() == "dpo":
        return {"type": "preference"}
    return {"type": "none"}


def _default_artifacts(domain: str, inputs: dict[str, Any]) -> list[str]:
    if domain == "gym_sb3":
        return ["model.zip", "eval.json", "rollout.mp4", "report.md", "artifact_manifest.json"]
    if domain == "llm_trl":
        method = str(inputs.get("method", "sft")).lower()
        artifacts = ["adapter", "metrics.json", "llm_eval.json", "llm_report.md"]
        if method == "grpo":
            artifacts.append("reward_distribution")
        return artifacts
    return ["artifact_manifest.json"]


async def create_experiment_plan_handler(args: dict[str, Any], session: Any = None, **_: Any) -> tuple[str, bool]:
    result = create_experiment_plan(**args)
    if session and "error" not in result:
        await session.send_event(Event(event_type="plan_update", data={"plan": result}))
    return json_ready(result), "error" not in result


async def validate_experiment_plan_handler(args: dict[str, Any], session: Any = None, **_: Any) -> tuple[str, bool]:
    plan = _resolve_plan(args.get("plan", {}), args.get("run_dir") or getattr(session, "run_dir", None))
    result = validate_experiment_plan(plan, args.get("run_dir") or getattr(session, "run_dir", None))
    return json_ready(result), bool(result.get("valid"))


async def run_experiment_stage_handler(args: dict[str, Any], session: Any = None, **_: Any) -> tuple[str, bool]:
    args = {
        **args,
        "plan": _resolve_plan(args.get("plan", {}), args.get("run_dir") or getattr(session, "run_dir", None)),
    }
    if session:
        await session.send_event(
            Event(
                event_type="plan_update",
                data={"stage": args.get("stage"), "status": "running", "plan": args.get("plan")},
            )
        )
    result = await asyncio.to_thread(run_experiment_stage, **args)
    if session:
        await session.send_event(
            Event(
                event_type="plan_update",
                data={
                    "stage": args.get("stage"),
                    "status": "failed" if "error" in result.get("result", result) else "completed",
                    "result": result,
                },
            )
        )
    return json_ready(result), "error" not in result.get("result", result)


async def get_artifact_manifest_handler(args: dict[str, Any], session: Any = None, **_: Any) -> tuple[str, bool]:
    result = get_artifact_manifest(args["run_dir"])
    if session:
        await session.send_event(Event(event_type="artifact_manifest", data={"manifest": result}))
    return json_ready(result), True


async def list_domain_adapters_handler(args: dict[str, Any], **_: Any) -> tuple[str, bool]:
    result = list_domain_adapters()
    return json_ready(result), True
