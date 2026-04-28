from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rl_intern.domains.llm_trl.dataset import inspect_llm_dataset
from rl_intern.domains.llm_trl.scripts import build_trl_training_script
from rl_intern.domains.llm_trl.verifier import validate_grpo_verifier
from rl_intern.orchestrator.adapters import DomainAdapter
from rl_intern.orchestrator.manifest import append_manifest_item
from rl_intern.orchestrator.models import ExperimentPlan


class LLMTRLAdapter(DomainAdapter):
    domain = "llm_trl"

    def inspect(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        method = _method(plan)
        result = inspect_llm_dataset(
            dataset_path=_input(plan, "dataset"),
            rows=plan.inputs.get("rows"),
            method=method,
        )
        _write_json(run_dir, "llm_dataset_inspect.json", result, "metrics")
        if method == "grpo":
            verifier = validate_grpo_verifier(
                verifier_path=plan.reward.verifier_path,
                verifier_source=plan.reward.verifier_source,
            )
            result["verifier"] = verifier
            _write_json(run_dir, "llm_grpo_verifier.json", verifier, "metrics")
        return result

    def prepare(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        method = _method(plan)
        script = build_trl_training_script(method)
        result = {
            "method": method,
            "model": _input(plan, "model"),
            "dataset": _input(plan, "dataset"),
            "dependencies": _dependencies(),
        }
        if run_dir:
            script_path = Path(run_dir) / "train_trl.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(script, encoding="utf-8")
            result["script_path"] = str(script_path)
            append_manifest_item(
                run_dir,
                "configs",
                script_path,
                kind="training_script",
                domain=self.domain,
                plan_id=plan.plan_id,
            )
            _write_json(run_dir, "llm_training_config.json", result, "configs")
        return result

    def smoke_test(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        inspect_result = self.inspect(plan, run_dir=run_dir)
        if not inspect_result.get("valid"):
            return {"passed": False, "error": inspect_result.get("reason"), "inspect": inspect_result}
        if _method(plan) == "grpo" and not inspect_result.get("verifier", {}).get("valid"):
            return {
                "passed": False,
                "error": inspect_result.get("verifier", {}).get("error"),
                "inspect": inspect_result,
            }
        result = {"passed": True, "message": "Dataset and verifier checks passed."}
        _write_json(run_dir, "llm_smoke_test.json", result, "metrics")
        return result

    def train(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        prepared = self.prepare(plan, run_dir=run_dir)
        if plan.runner.backend == "modal":
            from rl_intern.runners.modal_backend import run_modal_job

            return run_modal_job(
                run_id=Path(run_dir).name if run_dir else plan.plan_id,
                stage="train",
                script_path=prepared.get("script_path"),
                script_args=_script_args(plan, run_dir),
                dependencies=_dependencies(),
                hardware=plan.runner.hardware,
                timeout=plan.runner.timeout,
                run_dir=run_dir,
            )
        output_dir = Path(run_dir or ".") / "llm_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        marker = output_dir / "LOCAL_DRY_RUN.txt"
        marker.write_text(
            "Local LLM training is a dry run. Use runner.backend='modal' for heavy TRL jobs.\n",
            encoding="utf-8",
        )
        append_manifest_item(
            run_dir or ".",
            "logs",
            marker,
            kind="local_dry_run",
            domain=self.domain,
            plan_id=plan.plan_id,
        )
        return {"status": "dry_run", "output_dir": str(output_dir), "message": marker.read_text()}

    def evaluate(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        samples = [
            {
                "prompt": "Evaluation placeholder",
                "completion": "Evaluation requires a task-specific harness.",
                "score": None,
            }
        ]
        result = {
            "method": _method(plan),
            "model": _input(plan, "model"),
            "samples": samples,
            "metrics": {"status": "needs_task_specific_eval"},
        }
        _write_json(run_dir, "llm_eval.json", result, "metrics")
        if run_dir:
            append_manifest_item(
                run_dir,
                "samples",
                Path(run_dir) / "llm_eval.json",
                kind="eval_samples",
                domain=self.domain,
                plan_id=plan.plan_id,
            )
        return result

    def report(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        dataset = _read_json(run_dir, "llm_dataset_inspect.json")
        smoke = _read_json(run_dir, "llm_smoke_test.json")
        eval_result = _read_json(run_dir, "llm_eval.json")
        report = [
            "# LLM TRL Experiment Report",
            "",
            f"- Method: `{_method(plan)}`",
            f"- Base model: `{_input(plan, 'model') or 'unknown'}`",
            f"- Dataset: `{_input(plan, 'dataset') or 'inline'}`",
            f"- Dataset valid: `{dataset.get('valid')}`",
            f"- Smoke test: `{smoke.get('passed')}`",
            f"- Eval status: `{eval_result.get('metrics', {}).get('status', 'unknown')}`",
        ]
        if _method(plan) == "grpo":
            verifier = _read_json(run_dir, "llm_grpo_verifier.json")
            report.append(f"- GRPO verifier valid: `{verifier.get('valid')}`")
        if not run_dir:
            return {"content": "\n".join(report)}
        path = Path(run_dir) / "llm_report.md"
        path.write_text("\n".join(report) + "\n", encoding="utf-8")
        append_manifest_item(
            run_dir,
            "reports",
            path,
            kind="llm_markdown_report",
            domain=self.domain,
            plan_id=plan.plan_id,
        )
        return {"report_path": str(path)}

    def artifact_schema(self) -> dict[str, Any]:
        return {
            "adapters": ["llm_output/adapter"],
            "metrics": ["llm_eval.json", "metrics.json", "llm_dataset_inspect.json"],
            "samples": ["llm_eval.json"],
            "reports": ["llm_report.md"],
            "configs": ["train_trl.py", "llm_training_config.json"],
        }


def _method(plan: ExperimentPlan) -> str:
    return str(plan.inputs.get("method", "sft")).lower()


def _input(plan: ExperimentPlan, key: str) -> Any:
    aliases = {
        "model": ("model", "model_name"),
        "dataset": ("dataset", "dataset_name"),
        "split": ("split", "dataset_split"),
    }
    for candidate in aliases.get(key, (key,)):
        value = plan.inputs.get(candidate)
        if value not in (None, ""):
            return value
    return None


def _dependencies() -> list[str]:
    return [
        "transformers",
        "trl",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "torch",
    ]


def _script_args(plan: ExperimentPlan, run_dir: str | None) -> list[str]:
    output_dir = str(Path(run_dir or ".") / "llm_output")
    args = [
        "--model",
        str(_input(plan, "model")),
        "--dataset",
        str(_input(plan, "dataset")),
        "--output-dir",
        output_dir,
        "--max-steps",
        str(plan.inputs.get("max_steps", 20)),
    ]
    if _method(plan) == "grpo" and plan.reward.verifier_path:
        args.extend(["--verifier-path", plan.reward.verifier_path])
    return args


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
