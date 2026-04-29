from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rl_intern.domains.llm_trl.dataset import inspect_llm_dataset
from rl_intern.domains.llm_trl.evaluation import (
    dpo_evidence,
    evaluation_summary,
    grpo_evidence,
    sft_evidence,
)
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
        blocked = _blocked_by_failed_inspect(run_dir)
        if blocked:
            return blocked
        prepared = self._prepared_script(plan, run_dir)
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
        blocked = _blocked_by_failed_inspect(run_dir)
        if blocked:
            return blocked
        generation_result = _read_modal_sample_generations(run_dir)
        base_generations = _read_modal_json(run_dir, "base_generations.json")
        adapter_generations = _read_modal_json(run_dir, "adapter_generations.json")
        eval_metrics = _read_modal_json(run_dir, "eval_metrics.json")
        preference_metrics = _read_modal_json(run_dir, "preference_metrics.json")
        reward_metrics = _read_modal_json(run_dir, "reward_metrics.json")
        eval_dataset = _read_modal_json(run_dir, "eval_dataset_info.json")
        evidence = _read_modal_json(run_dir, "improvement_evidence.json")
        summary = _read_modal_json(run_dir, "evaluation_summary.json")
        samples = generation_result.get("samples") or _default_eval_prompts()
        status = generation_result.get("status", "pending_generation")
        if not evidence:
            evidence = _build_method_evidence(plan, eval_metrics, preference_metrics, reward_metrics)
        if not summary:
            summary = evaluation_summary(
                method=_method(plan),
                evidence=evidence or _missing_evidence(),
                eval_dataset=eval_dataset,
                metrics=_method_metrics(_method(plan), eval_metrics, preference_metrics, reward_metrics),
                samples=samples,
            )
        result = {
            "method": _method(plan),
            "model": _input(plan, "model"),
            "samples": samples,
            "base_samples": base_generations.get("samples", []),
            "adapter_samples": adapter_generations.get("samples", samples),
            "eval_metrics": eval_metrics,
            "preference_metrics": preference_metrics,
            "reward_metrics": reward_metrics,
            "eval_dataset": eval_dataset,
            "improvement_evidence": evidence or _missing_evidence(),
            "evaluation_summary": summary,
            "metrics": {
                "status": status,
                "sample_count": len(samples),
                "source": generation_result.get("path", "default_prompts"),
                "improvement_verdict": (evidence or {}).get("verdict", "inconclusive"),
                "run_class": (evidence or {}).get("run_class", "standard"),
            },
        }
        _write_json(run_dir, "llm_eval.json", result, "metrics")
        if evidence:
            _write_json(run_dir, "improvement_evidence.json", evidence, "metrics")
        if summary:
            _write_json(run_dir, "evaluation_summary.json", summary, "metrics")
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
        trainer_state = _read_modal_json(run_dir, "trainer_state.json")
        eval_metrics = eval_result.get("eval_metrics") or _read_modal_json(run_dir, "eval_metrics.json")
        preference_metrics = eval_result.get("preference_metrics") or _read_modal_json(run_dir, "preference_metrics.json")
        reward_metrics = eval_result.get("reward_metrics") or _read_modal_json(run_dir, "reward_metrics.json")
        summary = eval_result.get("evaluation_summary") or _read_json(run_dir, "evaluation_summary.json")
        evidence = eval_result.get("improvement_evidence") or _read_json(run_dir, "improvement_evidence.json")
        if not evidence:
            evidence = _build_method_evidence(plan, eval_metrics, preference_metrics, reward_metrics)
        evidence = evidence or _missing_evidence()
        verdict = evidence.get("verdict", "inconclusive")
        run_class = evidence.get("run_class", summary.get("run_class", "standard") if isinstance(summary, dict) else "standard")
        report = [
            "# LLM TRL Experiment Report",
            "",
            f"- Method: `{_method(plan)}`",
            f"- Base model: `{_input(plan, 'model') or 'unknown'}`",
            f"- Dataset: `{_input(plan, 'dataset') or 'inline'}`",
            f"- Dataset valid: `{dataset.get('valid')}`",
            f"- Smoke test: `{smoke.get('passed')}`",
            f"- Eval status: `{eval_result.get('metrics', {}).get('status', 'unknown')}`",
            f"- Eval samples: `{eval_result.get('metrics', {}).get('sample_count', 0)}`",
            f"- Improvement verdict: `{verdict}`",
            f"- Run class: `{run_class}`",
            f"- Improvement claim: {_claim_sentence(evidence)}",
        ]
        if evidence.get("reason"):
            report.append(f"- Verdict reason: {evidence['reason']}")
        warnings = evidence.get("warnings") or []
        if warnings:
            report.append(f"- Evidence warnings: {'; '.join(str(w) for w in warnings)}")
        report.extend(_metrics_table(_method(plan), eval_metrics, preference_metrics, reward_metrics))
        training_rows = _training_metrics_rows(trainer_state)
        if training_rows:
            report.extend(["", "## Training Metrics", "", "| Step | Loss | Learning Rate | Token Accuracy |", "| --- | ---: | ---: | ---: |"])
            report.extend(training_rows)
        samples = eval_result.get("samples") or []
        if samples:
            report.extend(["", "## Sample Generations", ""])
            for idx, sample in enumerate(samples[:3], start=1):
                prompt = str(sample.get("prompt", "")).replace("\n", " ")
                completion = str(sample.get("completion", "")).replace("\n", " ")
                report.extend(
                    [
                        f"### Sample {idx}",
                        "",
                        f"Prompt: {prompt}",
                        "",
                        f"Completion: {completion}",
                        "",
                    ]
                )
        base_samples = eval_result.get("base_samples") or []
        adapter_samples = eval_result.get("adapter_samples") or []
        if base_samples and adapter_samples:
            report.extend(["", "## Base vs Adapter Samples", ""])
            for idx, (base, adapter) in enumerate(zip(base_samples[:3], adapter_samples[:3]), start=1):
                prompt = str(adapter.get("prompt") or base.get("prompt") or "").replace("\n", " ")
                base_completion = str(base.get("completion", "")).replace("\n", " ")
                adapter_completion = str(adapter.get("completion", "")).replace("\n", " ")
                report.extend(
                    [
                        f"### Comparison {idx}",
                        "",
                        f"Prompt: {prompt}",
                        "",
                        f"Base: {base_completion}",
                        "",
                        f"Adapter: {adapter_completion}",
                        "",
                    ]
                )
        artifact_paths = _evidence_artifact_paths(run_dir)
        if artifact_paths:
            report.extend(["", "## Evidence Artifacts", ""])
            report.extend(f"- `{path}`" for path in artifact_paths)
        comparison_path = _write_comparison_report(plan, run_dir)
        if comparison_path:
            report.extend(["", "## Comparison", "", f"- `{comparison_path}`"])
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
            "metrics": [
                "llm_eval.json",
                "metrics.json",
                "eval_metrics.json",
                "sft_metrics.json",
                "preference_metrics.json",
                "reward_metrics.json",
                "reward_distribution.json",
                "improvement_evidence.json",
                "evaluation_summary.json",
                "eval_dataset_info.json",
                "trainer_state.json",
                "llm_dataset_inspect.json",
            ],
            "samples": ["llm_eval.json", "base_generations.json", "adapter_generations.json", "sample_generations.json"],
            "reports": ["llm_report.md"],
            "configs": ["train_trl.py", "llm_training_config.json"],
        }

    def _prepared_script(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        if run_dir:
            script_path = Path(run_dir) / "train_trl.py"
            if script_path.exists():
                return {
                    "method": _method(plan),
                    "model": _input(plan, "model"),
                    "dataset": _input(plan, "dataset"),
                    "dependencies": _dependencies(),
                    "script_path": str(script_path),
                    "script_source": "existing",
                }
        result = self.prepare(plan, run_dir=run_dir)
        result["script_source"] = "generated"
        return result


def _method(plan: ExperimentPlan) -> str:
    return str(plan.inputs.get("method", "sft")).lower()


def _input(plan: ExperimentPlan, key: str) -> Any:
    aliases = {
        "model": ("model", "model_name"),
        "dataset": ("dataset", "dataset_name", "dataset_path"),
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
    output_dir = str(plan.inputs.get("output_dir", "llm_output"))
    if plan.runner.backend != "modal":
        output_dir = str(Path(run_dir or ".") / output_dir)
    args = [
        "--model",
        str(_input(plan, "model")),
        "--dataset",
        str(_input(plan, "dataset")),
        "--output-dir",
        output_dir,
        "--split",
        str(_input(plan, "split") or "train"),
        "--max-steps",
        str(plan.inputs.get("max_steps", 20)),
        "--max-samples",
        str(plan.inputs.get("max_samples", 100)),
        "--per-device-train-batch-size",
        str(plan.inputs.get("per_device_train_batch_size", 1)),
        "--gradient-accumulation-steps",
        str(plan.inputs.get("gradient_accumulation_steps", 1)),
        "--learning-rate",
        str(plan.inputs.get("learning_rate", 5e-5)),
        "--logging-steps",
        str(plan.inputs.get("logging_steps", 1)),
        "--warmup-steps",
        str(plan.inputs.get("warmup_steps", 0)),
    ]
    if plan.inputs.get("save_steps") is not None:
        args.extend(["--save-steps", str(plan.inputs["save_steps"])])
    args.extend(
        [
            "--eval-split",
            str(plan.inputs.get("eval_split", "test")),
            "--eval-samples",
            str(plan.inputs.get("eval_samples", 20)),
            "--max-new-tokens",
            str(plan.inputs.get("max_new_tokens", 128)),
            "--improvement-threshold-pct",
            str(_improvement_threshold_pct(plan)),
            "--preference-margin-threshold",
            str(plan.inputs.get("preference_margin_threshold", 0.01)),
            "--reward-threshold",
            str(plan.inputs.get("reward_threshold", 0.01)),
        ]
    )
    if plan.inputs.get("eval_prompt_suite") is not None:
        args.extend(["--eval-prompt-suite", json.dumps(plan.inputs["eval_prompt_suite"])])
    if plan.inputs.get("eval_categories") is not None:
        args.extend(["--eval-categories", json.dumps(plan.inputs["eval_categories"])])
    if plan.inputs.get("judge_mode") is not None:
        args.extend(["--judge-mode", str(plan.inputs["judge_mode"])])
    if plan.inputs.get("smoke_only_thresholds") is not None:
        args.extend(["--smoke-only-thresholds", json.dumps(plan.inputs["smoke_only_thresholds"])])
    for stop in plan.inputs.get("stop_sequences", ["### Human:", "\n### Human:"]):
        args.extend(["--stop-sequence", str(stop)])
    if bool(plan.inputs.get("fp16")) or _default_fp16(plan):
        args.append("--fp16")
    if bool(plan.inputs.get("bf16")):
        args.append("--bf16")
    if _method(plan) == "grpo" and plan.reward.verifier_path:
        args.extend(["--verifier-path", plan.reward.verifier_path])
    return args


def _default_fp16(plan: ExperimentPlan) -> bool:
    if plan.inputs.get("bf16") is not None or plan.inputs.get("fp16") is not None:
        return False
    return plan.runner.backend == "modal" and "t4" in str(plan.runner.hardware).lower()


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
    metadata = {}
    if filename in {"evaluation_summary.json", "improvement_evidence.json"}:
        metadata = {
            key: payload[key]
            for key in ("verdict", "run_class", "reason")
            if key in payload
        }
    append_manifest_item(run_dir, bucket, path, kind=filename.removesuffix(".json"), metadata=metadata)


def _read_json(run_dir: str | None, filename: str) -> dict[str, Any]:
    if not run_dir:
        return {}
    path = Path(run_dir) / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_modal_json(run_dir: str | None, filename: str) -> dict[str, Any]:
    if not run_dir:
        return {}
    root = Path(run_dir) / "modal_artifacts"
    if not root.exists():
        return {}
    for path in sorted(root.glob(f"**/{filename}")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload.setdefault("path", str(path))
        return payload
    return {}


def _read_modal_sample_generations(run_dir: str | None) -> dict[str, Any]:
    if not run_dir:
        return {}
    root = Path(run_dir) / "modal_artifacts"
    if not root.exists():
        return {}
    for path in sorted(root.glob("*/sample_generations.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["path"] = str(path)
        return payload
    return {}


def _default_eval_prompts() -> list[dict[str, Any]]:
    prompts = [
        "### Human: Explain reinforcement learning in one paragraph.\n### Assistant:",
        "### Human: Give two practical tips for training small language models.\n### Assistant:",
        "### Human: What is overfitting?\n### Assistant:",
    ]
    return [{"prompt": prompt, "completion": None, "score": None} for prompt in prompts]


def _blocked_by_failed_inspect(run_dir: str | None) -> dict[str, Any] | None:
    inspect_result = _read_json(run_dir, "llm_dataset_inspect.json")
    if inspect_result and inspect_result.get("valid") is False:
        return {
            "status": "failed",
            "error": "Cannot continue after failed LLM dataset inspection.",
            "inspect": inspect_result,
        }
    return None


def _improvement_threshold_pct(plan: ExperimentPlan) -> float:
    try:
        return float(plan.inputs.get("improvement_threshold_pct", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _missing_evidence() -> dict[str, Any]:
    return {
        "verdict": "inconclusive",
        "reason": "No base-vs-adapter held-out evaluation artifacts were found.",
        "warnings": ["Sample generations alone are qualitative and do not prove improvement."],
    }


def _metric_value(payload: dict[str, Any], model_key: str, metric_key: str) -> float | None:
    value = (payload.get(model_key) or {}).get(metric_key)
    return value if isinstance(value, int | float) else None


def _build_improvement_evidence(eval_metrics: dict[str, Any], threshold_pct: float) -> dict[str, Any]:
    base_loss = _metric_value(eval_metrics, "base", "loss")
    adapter_loss = _metric_value(eval_metrics, "adapter", "loss")
    warnings: list[str] = []
    if base_loss is None or adapter_loss is None:
        return {
            "verdict": "inconclusive",
            "reason": "Held-out loss was not available for both base and adapter models.",
            "warnings": warnings,
        }
    base_examples = _metric_value(eval_metrics, "base", "examples") or 0
    adapter_examples = _metric_value(eval_metrics, "adapter", "examples") or 0
    if base_examples < 3 or adapter_examples < 3:
        warnings.append("Very small eval sample count; treat this as a smoke signal, not proof.")
    delta = adapter_loss - base_loss
    pct = ((base_loss - adapter_loss) / base_loss * 100.0) if base_loss else 0.0
    if pct >= threshold_pct:
        verdict = "improved"
        reason = f"Adapter held-out loss improved by {pct:.2f}%."
    elif pct <= -threshold_pct:
        verdict = "regressed"
        reason = f"Adapter held-out loss worsened by {abs(pct):.2f}%."
    else:
        verdict = "inconclusive"
        reason = f"Loss delta {pct:.2f}% is below the {threshold_pct:.2f}% threshold."
    return {
        "verdict": verdict,
        "reason": reason,
        "base_loss": base_loss,
        "adapter_loss": adapter_loss,
        "delta_loss": delta,
        "delta_loss_pct": pct,
        "threshold_pct": threshold_pct,
        "warnings": warnings,
    }


def _claim_sentence(evidence: dict[str, Any]) -> str:
    verdict = evidence.get("verdict")
    if verdict == "improved":
        return "Fine-tuning improved the model on the configured held-out metric."
    if verdict == "regressed":
        return "Fine-tuning regressed the model on the configured held-out metric."
    return "Training succeeded only if the train stage passed; model improvement is not established."


def _fmt_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return "n/a"


def _sft_metrics_table(eval_metrics: dict[str, Any]) -> list[str]:
    if not eval_metrics:
        return [
            "",
            "## Base vs Adapter Metrics",
            "",
            "No held-out base-vs-adapter metrics were found.",
        ]
    base = eval_metrics.get("base") or {}
    adapter = eval_metrics.get("adapter") or {}
    return [
        "",
        "## Base vs Adapter Metrics",
        "",
        "| Model | Loss | Perplexity | Examples |",
        "| --- | ---: | ---: | ---: |",
        f"| Base | {_fmt_metric(base.get('loss'))} | {_fmt_metric(base.get('perplexity'))} | {_fmt_metric(base.get('examples'))} |",
        f"| Adapter | {_fmt_metric(adapter.get('loss'))} | {_fmt_metric(adapter.get('perplexity'))} | {_fmt_metric(adapter.get('examples'))} |",
    ]


def _training_metrics_rows(trainer_state: dict[str, Any]) -> list[str]:
    rows = []
    for item in trainer_state.get("log_history", []):
        if "loss" not in item:
            continue
        token_accuracy = item.get("mean_token_accuracy", item.get("token_accuracy"))
        rows.append(
            "| {step} | {loss} | {lr} | {acc} |".format(
                step=_fmt_metric(item.get("step")),
                loss=_fmt_metric(item.get("loss")),
                lr=_fmt_metric(item.get("learning_rate")),
                acc=_fmt_metric(token_accuracy),
            )
        )
    return rows


def _evidence_artifact_paths(run_dir: str | None) -> list[str]:
    if not run_dir:
        return []
    names = [
        "eval_dataset_info.json",
        "eval_metrics.json",
        "sft_metrics.json",
        "preference_metrics.json",
        "reward_metrics.json",
        "reward_distribution.json",
        "improvement_evidence.json",
        "evaluation_summary.json",
        "base_generations.json",
        "adapter_generations.json",
        "sample_generations.json",
        "trainer_state.json",
    ]
    root = Path(run_dir)
    paths = []
    for name in names:
        local = root / name
        if local.exists():
            paths.append(str(local))
            continue
        modal = _find_modal_file(run_dir, name)
        if modal:
            paths.append(str(modal))
    return paths


def _find_modal_file(run_dir: str, filename: str) -> Path | None:
    root = Path(run_dir) / "modal_artifacts"
    if not root.exists():
        return None
    for path in sorted(root.glob(f"**/{filename}")):
        return path
    return None


def _build_method_evidence(
    plan: ExperimentPlan,
    eval_metrics: dict[str, Any],
    preference_metrics: dict[str, Any],
    reward_metrics: dict[str, Any],
) -> dict[str, Any]:
    method = _method(plan)
    smoke_thresholds = plan.inputs.get("smoke_only_thresholds")
    if not isinstance(smoke_thresholds, dict):
        smoke_thresholds = None
    if method == "dpo":
        return dpo_evidence(
            preference_metrics,
            threshold=float(plan.inputs.get("preference_margin_threshold", 0.01)),
            max_steps=plan.inputs.get("max_steps"),
            max_samples=plan.inputs.get("max_samples"),
            smoke_thresholds=smoke_thresholds,
        )
    if method == "grpo":
        return grpo_evidence(
            reward_metrics,
            threshold=float(plan.inputs.get("reward_threshold", 0.01)),
            max_steps=plan.inputs.get("max_steps"),
            max_samples=plan.inputs.get("max_samples"),
            smoke_thresholds=smoke_thresholds,
        )
    return sft_evidence(
        eval_metrics,
        threshold_pct=_improvement_threshold_pct(plan),
        max_steps=plan.inputs.get("max_steps"),
        max_samples=plan.inputs.get("max_samples"),
        smoke_thresholds=smoke_thresholds,
    )


def _method_metrics(
    method: str,
    eval_metrics: dict[str, Any],
    preference_metrics: dict[str, Any],
    reward_metrics: dict[str, Any],
) -> dict[str, Any]:
    if method == "dpo":
        return preference_metrics
    if method == "grpo":
        return reward_metrics
    return eval_metrics


def _metrics_table(
    method: str,
    eval_metrics: dict[str, Any],
    preference_metrics: dict[str, Any] | None = None,
    reward_metrics: dict[str, Any] | None = None,
) -> list[str]:
    if method == "dpo":
        metrics = preference_metrics or {}
        if not metrics:
            return ["", "## DPO Preference Metrics", "", "No held-out DPO preference metrics were found."]
        base = metrics.get("base") or {}
        adapter = metrics.get("adapter") or {}
        return [
            "",
            "## DPO Preference Metrics",
            "",
            "| Model | Mean Margin | Pair Count |",
            "| --- | ---: | ---: |",
            f"| Base | {_fmt_metric(base.get('mean_margin'))} | {_fmt_metric(base.get('pairs'))} |",
            f"| Adapter | {_fmt_metric(adapter.get('mean_margin'))} | {_fmt_metric(adapter.get('pairs'))} |",
        ]
    if method == "grpo":
        metrics = reward_metrics or {}
        if not metrics:
            return ["", "## GRPO Reward Metrics", "", "No held-out GRPO reward metrics were found."]
        base = metrics.get("base") or {}
        adapter = metrics.get("adapter") or {}
        return [
            "",
            "## GRPO Reward Metrics",
            "",
            "| Model | Mean Reward | Samples | Failures |",
            "| --- | ---: | ---: | ---: |",
            f"| Base | {_fmt_metric(base.get('mean_reward'))} | {_fmt_metric(base.get('samples'))} | {_fmt_metric(base.get('failure_count'))} |",
            f"| Adapter | {_fmt_metric(adapter.get('mean_reward'))} | {_fmt_metric(adapter.get('samples'))} | {_fmt_metric(adapter.get('failure_count'))} |",
        ]
    return _sft_metrics_table(eval_metrics)


def _write_comparison_report(plan: ExperimentPlan, run_dir: str | None) -> str | None:
    group = plan.inputs.get("comparison_group")
    if not group or not run_dir:
        return None
    root = Path(run_dir).parent
    rows = []
    for candidate in sorted(root.glob("run_*")):
        plan_path = candidate / "experiment_plan.json"
        eval_path = candidate / "llm_eval.json"
        if not plan_path.exists() or not eval_path.exists():
            continue
        try:
            candidate_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            candidate_eval = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if candidate_plan.get("inputs", {}).get("comparison_group") != group:
            continue
        evidence = candidate_eval.get("improvement_evidence") or {}
        rows.append(
            {
                "run": candidate.name,
                "max_samples": candidate_plan.get("inputs", {}).get("max_samples"),
                "max_steps": candidate_plan.get("inputs", {}).get("max_steps"),
                "verdict": evidence.get("verdict", "unknown"),
                "run_class": evidence.get("run_class", "standard"),
                "reason": evidence.get("reason", ""),
            }
        )
    if not rows:
        return None
    lines = [
        "# LLM Comparison Report",
        "",
        f"- Comparison group: `{group}`",
        "",
        "| Run | Samples | Steps | Verdict | Class | Reason |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {_fmt_metric(row['max_samples'])} | {_fmt_metric(row['max_steps'])} | "
            f"{row['verdict']} | {row['run_class']} | {row['reason']} |"
        )
    path = Path(run_dir) / "comparison_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_manifest_item(run_dir, "reports", path, kind="llm_comparison_report")
    return str(path)
