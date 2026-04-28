from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agent.tools.algorithm_select import choose_algorithm
from agent.tools.env_inspect import inspect_env
from agent.tools.env_smoke_test import smoke_test_env
from agent.tools.evaluate_policy import evaluate_policy
from agent.tools.random_baseline import run_random_baseline
from agent.tools.record_rollout import record_rollout
from agent.tools.report import generate_report
from agent.tools.train_sb3 import train_sb3
from rl_intern.orchestrator.adapters import DomainAdapter
from rl_intern.orchestrator.manifest import append_manifest_item
from rl_intern.orchestrator.models import ExperimentPlan


class GymSB3Adapter(DomainAdapter):
    domain = "gym_sb3"

    def inspect(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        env_id = _require(plan.inputs, "env_id")
        result = inspect_env(env_id)
        _write_json(run_dir, "gym_env_inspect.json", result, "metrics")
        return result

    def prepare(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        env_id = _require(plan.inputs, "env_id")
        algorithm = str(plan.inputs.get("algorithm", "PPO")).upper()
        result = choose_algorithm(env_id, algorithm)
        _write_json(run_dir, "gym_algorithm.json", result, "configs")
        return result

    def smoke_test(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        env_id = _require(plan.inputs, "env_id")
        seed = int(plan.inputs.get("seed", 0))
        result = smoke_test_env(env_id, seed=seed)
        _write_json(run_dir, "gym_smoke_test.json", result, "metrics")
        return result

    def train(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        env_id = _require(plan.inputs, "env_id")
        algorithm = str(plan.inputs.get("algorithm", "PPO")).upper()
        total_timesteps = int(plan.inputs.get("total_timesteps", 100_000))
        seeds = _seeds(plan)
        results = []
        baseline_results = []
        for seed in seeds:
            baseline_results.append(run_random_baseline(env_id, seed=seed))
            seed_run_dir = _seed_run_dir(run_dir, seed, multi=len(seeds) > 1)
            result = train_sb3(
                env_id,
                algorithm=algorithm,
                total_timesteps=total_timesteps,
                seed=seed,
                run_dir=seed_run_dir,
            )
            results.append(result)
            if run_dir and result.get("model_path"):
                append_manifest_item(
                    run_dir,
                    "checkpoints",
                    result["model_path"],
                    kind="sb3_model",
                    metadata={"seed": seed, "algorithm": algorithm, "env_id": env_id},
                    domain=self.domain,
                    plan_id=plan.plan_id,
                )
            if run_dir and result.get("config_path"):
                append_manifest_item(
                    run_dir,
                    "configs",
                    result["config_path"],
                    kind="training_config",
                    metadata={"seed": seed},
                    domain=self.domain,
                    plan_id=plan.plan_id,
                )
        aggregate = {
            "env_id": env_id,
            "algorithm": algorithm,
            "seeds": seeds,
            "random_baselines": baseline_results,
            "training_results": results,
        }
        _write_json(run_dir, "gym_training_summary.json", aggregate, "metrics")
        return aggregate

    def evaluate(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        env_id = _require(plan.inputs, "env_id")
        algorithm = str(plan.inputs.get("algorithm", "PPO")).upper()
        episodes = int(plan.inputs.get("eval_episodes", 20))
        train_summary = _read_json(run_dir, "gym_training_summary.json")
        training_results = train_summary.get("training_results", [])
        evaluations = []
        rollouts = []
        for idx, training in enumerate(training_results):
            if training.get("error"):
                evaluations.append({"error": training["error"], "seed": training.get("seed")})
                continue
            seed = int(training.get("seed", idx))
            seed_run_dir = _seed_run_dir(run_dir, seed, multi=len(training_results) > 1)
            evaluation = evaluate_policy(
                env_id,
                algorithm,
                training["model_path"],
                episodes=episodes,
                seed=seed,
                run_dir=seed_run_dir,
            )
            rollout = record_rollout(
                env_id,
                algorithm,
                training["model_path"],
                seed=seed,
                run_dir=seed_run_dir,
            )
            evaluations.append(evaluation)
            rollouts.append(rollout)
            if run_dir and evaluation.get("results_path"):
                append_manifest_item(
                    run_dir,
                    "metrics",
                    evaluation["results_path"],
                    kind="evaluation",
                    metadata={"seed": seed},
                    domain=self.domain,
                    plan_id=plan.plan_id,
                )
            if run_dir and rollout.get("video_path"):
                append_manifest_item(
                    run_dir,
                    "videos",
                    rollout["video_path"],
                    kind="rollout",
                    metadata={"seed": seed},
                    domain=self.domain,
                    plan_id=plan.plan_id,
                )
        rewards = [e["mean_reward"] for e in evaluations if "mean_reward" in e]
        summary = {
            "env_id": env_id,
            "algorithm": algorithm,
            "episodes": episodes,
            "evaluations": evaluations,
            "rollouts": rollouts,
            "aggregate_mean_reward": float(np.mean(rewards)) if rewards else None,
            "aggregate_std_reward": float(np.std(rewards)) if rewards else None,
        }
        _write_json(run_dir, "gym_evaluation_summary.json", summary, "metrics")
        return summary

    def report(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        train_summary = _read_json(run_dir, "gym_training_summary.json")
        eval_summary = _read_json(run_dir, "gym_evaluation_summary.json")
        env_result = _read_json(run_dir, "gym_env_inspect.json")
        smoke_result = _read_json(run_dir, "gym_smoke_test.json")
        first_training = _first_ok(train_summary.get("training_results", []))
        first_eval = _first_ok(eval_summary.get("evaluations", []))
        first_rollout = _first_ok(eval_summary.get("rollouts", []))
        first_baseline = _first_ok(train_summary.get("random_baselines", []))
        if first_training and first_eval:
            result = generate_report(
                env_result,
                smoke_result,
                first_baseline,
                first_training,
                first_eval,
                first_rollout,
                run_dir=run_dir,
            )
        else:
            result = {"error": "No successful Gym/SB3 training and evaluation pair to report."}
        summary_path = None
        if run_dir:
            summary_path = Path(run_dir) / "gym_comparison_report.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "training": train_summary,
                        "evaluation": eval_summary,
                        "report": result,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            append_manifest_item(
                run_dir,
                "reports",
                summary_path,
                kind="gym_comparison",
                domain=self.domain,
                plan_id=plan.plan_id,
            )
            if result.get("report_path"):
                append_manifest_item(
                    run_dir,
                    "reports",
                    result["report_path"],
                    kind="markdown_report",
                    domain=self.domain,
                    plan_id=plan.plan_id,
                )
        return {**result, "comparison_report_path": str(summary_path) if summary_path else None}

    def artifact_schema(self) -> dict[str, Any]:
        return {
            "checkpoints": ["model.zip"],
            "metrics": ["eval.json", "gym_*_summary.json"],
            "videos": ["rollout.mp4 or rollout.gif"],
            "reports": ["report.md", "gym_comparison_report.json"],
            "configs": ["config.json"],
        }


def _require(inputs: dict[str, Any], key: str) -> Any:
    value = inputs.get(key)
    if value in (None, ""):
        raise ValueError(f"gym_sb3 plan requires inputs.{key}")
    return value


def _seeds(plan: ExperimentPlan) -> list[int]:
    if isinstance(plan.inputs.get("seeds"), list):
        return [int(seed) for seed in plan.inputs["seeds"]]
    return [int(plan.inputs.get("seed", 0))]


def _seed_run_dir(run_dir: str | None, seed: int, *, multi: bool) -> str | None:
    if not run_dir:
        return None
    return str(Path(run_dir) / f"seed_{seed}") if multi else run_dir


def _write_json(
    run_dir: str | None,
    filename: str,
    payload: dict[str, Any],
    bucket: str,
) -> None:
    if not run_dir:
        return
    path = Path(run_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    append_manifest_item(run_dir, bucket, path, kind=filename.removesuffix(".json"))


def _read_json(run_dir: str | None, filename: str) -> dict[str, Any]:
    if not run_dir:
        return {}
    path = Path(run_dir) / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _first_ok(items: list[dict[str, Any]]) -> dict[str, Any]:
    for item in items:
        if item and not item.get("error"):
            return item
    return {}
