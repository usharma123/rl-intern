from rl_intern.domains.llm_trl.dataset import inspect_llm_dataset
from rl_intern.domains.llm_trl.adapter import LLMTRLAdapter
from rl_intern.domains.llm_trl.scripts import build_trl_training_script
from rl_intern.domains.llm_trl.verifier import validate_grpo_verifier
from agent.tools.orchestrator import create_experiment_plan, validate_experiment_plan
from rl_intern.orchestrator.models import ExperimentPlan, RewardSpec, StageSpec


def test_sft_dataset_accepts_messages_rows():
    result = inspect_llm_dataset(rows=[{"messages": [{"role": "user", "content": "hi"}]}], method="sft")

    assert result["valid"] is True
    assert result["format"] == "chat_messages"


def test_dpo_dataset_rejects_missing_rejected():
    result = inspect_llm_dataset(rows=[{"prompt": "p", "chosen": "c"}], method="dpo")

    assert result["valid"] is False
    assert "rejected" in result["reason"]


def test_dpo_dataset_accepts_chat_preference_pairs():
    rows = [
        {
            "chosen": [
                {"role": "user", "content": "Solve 2+2"},
                {"role": "assistant", "content": "4"},
            ],
            "rejected": [
                {"role": "user", "content": "Solve 2+2"},
                {"role": "assistant", "content": "5"},
            ],
        }
    ]

    result = inspect_llm_dataset(rows=rows, method="dpo")

    assert result["valid"] is True
    assert result["format"] == "chat_preference_pairs"
    assert "derived" in result["warnings"][0]


def test_dpo_dataset_rejects_ambiguous_chat_preference_pairs():
    rows = [
        {
            "chosen": [
                {"role": "user", "content": "Solve 2+2"},
                {"role": "assistant", "content": "4"},
            ],
            "rejected": [
                {"role": "user", "content": "Solve 3+3"},
                {"role": "assistant", "content": "5"},
            ],
        }
    ]

    result = inspect_llm_dataset(rows=rows, method="dpo")

    assert result["valid"] is False
    assert "shared first user prompt" in result["reason"]


def test_grpo_verifier_accepts_numeric_score():
    source = "def score(example, completion):\n    return 1.0\n"

    result = validate_grpo_verifier(verifier_source=source)

    assert result["valid"] is True
    assert result["score"] == 1.0


def test_grpo_verifier_rejects_non_numeric_score():
    source = "def score(example, completion):\n    return 'good'\n"

    result = validate_grpo_verifier(verifier_source=source)

    assert result["valid"] is False


def test_generate_grpo_script_contains_reward_func():
    script = build_trl_training_script("grpo")

    assert "GRPOTrainer" in script
    assert "reward_func" in script


def test_generate_sft_script_logs_phases_and_limits_samples():
    script = build_trl_training_script("sft")

    compile(script, "train_trl.py", "exec")
    assert "--max-samples" in script
    assert "--fp16" in script
    assert "bf16=args.bf16" in script
    assert "dtype=torch_dtype" in script
    assert "dataset.select(range(args.max_samples))" in script
    assert "[rl-intern] loading model" in script
    assert "low_cpu_mem_usage=True" in script
    assert "sample_generations.json" in script


def test_generate_sft_script_emits_defensible_eval_artifacts():
    script = build_trl_training_script("sft")

    assert "eval_dataset_info.json" in script
    assert "base_generations.json" in script
    assert "adapter_generations.json" in script
    assert "eval_metrics.json" in script
    assert "improvement_evidence.json" in script
    assert "stop_at_sequences" in script
    assert "### Human:" in script
    assert "overlap_with_train" in script


def test_failed_llm_inspect_blocks_train(tmp_path):
    plan = ExperimentPlan(
        plan_id="plan_test",
        domain="llm_trl",
        objective="bad dpo",
        inputs={"method": "dpo", "model": "tiny", "rows": [{"prompt": "p", "chosen": "c"}]},
        reward=RewardSpec(type="preference"),
        stages=[StageSpec(name="inspect"), StageSpec(name="train")],
        expected_artifacts=["adapter"],
    )
    adapter = LLMTRLAdapter()

    inspect = adapter.inspect(plan, run_dir=str(tmp_path))
    train = adapter.train(plan, run_dir=str(tmp_path))

    assert inspect["valid"] is False
    assert train["status"] == "failed"
    assert "failed LLM dataset inspection" in train["error"]


def test_llm_plan_normalizes_dataset_path_alias(tmp_path):
    plan = create_experiment_plan(
        domain="llm_trl",
        objective="dpo",
        inputs={"method": "dpo", "model": "tiny", "dataset_path": "org/dataset"},
        run_dir=str(tmp_path),
    )

    assert plan["inputs"]["dataset"] == "org/dataset"


def test_llm_gpu_runner_defaults_to_modal_backend(tmp_path):
    plan = create_experiment_plan(
        domain="llm_trl",
        objective="sft",
        inputs={"method": "sft", "model": "tiny", "dataset": "org/dataset"},
        runner={"hardware": "gpu-t4", "timeout": "30m"},
        run_dir=str(tmp_path),
    )

    validated = validate_experiment_plan(plan, run_dir=str(tmp_path))

    assert validated["valid"] is True
    assert validated["plan"]["runner"]["backend"] == "modal"


def test_llm_train_passes_script_args_to_modal(monkeypatch, tmp_path):
    plan = ExperimentPlan(
        plan_id="plan_test",
        domain="llm_trl",
        objective="tiny sft",
        inputs={
            "method": "sft",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "dataset": "timdettmers/openassistant-guanaco",
            "output_dir": "custom_output",
            "max_steps": 5,
            "max_samples": 20,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "logging_steps": 3,
            "warmup_steps": 1,
            "save_steps": 5,
            "fp16": True,
            "bf16": False,
        },
        stages=[StageSpec(name="inspect"), StageSpec(name="train")],
        expected_artifacts=["adapter"],
        runner={"backend": "modal", "hardware": "gpu-t4", "timeout": "30m"},
    )
    captured = {}

    def fake_run_modal_job(**kwargs):
        captured.update(kwargs)
        return {"status": "running", "backend_id": "fc-test"}

    monkeypatch.setattr("rl_intern.runners.modal_backend.run_modal_job", fake_run_modal_job)

    result = LLMTRLAdapter().train(plan, run_dir=str(tmp_path))

    assert result["status"] == "running"
    assert captured["script_args"][:6] == [
        "--model",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "--dataset",
        "timdettmers/openassistant-guanaco",
        "--output-dir",
        "custom_output",
    ]
    script_args = captured["script_args"]
    assert "--max-samples" in script_args
    assert script_args[script_args.index("--max-samples") + 1] == "20"
    assert script_args[script_args.index("--per-device-train-batch-size") + 1] == "2"
    assert script_args[script_args.index("--gradient-accumulation-steps") + 1] == "4"
    assert script_args[script_args.index("--learning-rate") + 1] == "0.0002"
    assert script_args[script_args.index("--logging-steps") + 1] == "3"
    assert script_args[script_args.index("--warmup-steps") + 1] == "1"
    assert script_args[script_args.index("--save-steps") + 1] == "5"
    assert "--fp16" in script_args
    assert "--bf16" not in script_args
    assert script_args[script_args.index("--eval-samples") + 1] == "20"
    assert script_args[script_args.index("--max-new-tokens") + 1] == "128"
    assert script_args[script_args.index("--improvement-threshold-pct") + 1] == "1.0"
    assert "--stop-sequence" in script_args


def test_llm_train_defaults_fp16_on_modal_t4(monkeypatch, tmp_path):
    plan = ExperimentPlan(
        plan_id="plan_test",
        domain="llm_trl",
        objective="tiny sft",
        inputs={
            "method": "sft",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "dataset": "timdettmers/openassistant-guanaco",
        },
        stages=[StageSpec(name="inspect"), StageSpec(name="train")],
        expected_artifacts=["adapter"],
        runner={"backend": "modal", "hardware": "gpu-t4", "timeout": "30m"},
    )
    captured = {}

    def fake_run_modal_job(**kwargs):
        captured.update(kwargs)
        return {"status": "running", "backend_id": "fc-test"}

    monkeypatch.setattr("rl_intern.runners.modal_backend.run_modal_job", fake_run_modal_job)

    LLMTRLAdapter().train(plan, run_dir=str(tmp_path))

    assert "--fp16" in captured["script_args"]


def test_llm_train_reuses_existing_edited_script(monkeypatch, tmp_path):
    edited_script = tmp_path / "train_trl.py"
    edited_script.write_text("print('edited script')\n", encoding="utf-8")
    plan = ExperimentPlan(
        plan_id="plan_test",
        domain="llm_trl",
        objective="tiny sft",
        inputs={
            "method": "sft",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "dataset": "timdettmers/openassistant-guanaco",
        },
        stages=[StageSpec(name="inspect"), StageSpec(name="train")],
        expected_artifacts=["adapter"],
        runner={"backend": "modal", "hardware": "gpu-t4", "timeout": "30m"},
    )
    captured = {}

    def fake_run_modal_job(**kwargs):
        captured.update(kwargs)
        return {"status": "running", "backend_id": "fc-test"}

    monkeypatch.setattr("rl_intern.runners.modal_backend.run_modal_job", fake_run_modal_job)

    result = LLMTRLAdapter().train(plan, run_dir=str(tmp_path))

    assert result["status"] == "running"
    assert captured["script_path"] == str(edited_script)
    assert edited_script.read_text(encoding="utf-8") == "print('edited script')\n"


def test_llm_evaluate_reads_modal_sample_generations(tmp_path):
    sample_path = tmp_path / "modal_artifacts" / "sft_output" / "sample_generations.json"
    sample_path.parent.mkdir(parents=True)
    sample_path.write_text(
        '{"status": "completed", "samples": [{"prompt": "p", "completion": "c", "score": null}]}',
        encoding="utf-8",
    )
    plan = ExperimentPlan(
        plan_id="plan_test",
        domain="llm_trl",
        objective="tiny sft",
        inputs={
            "method": "sft",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "dataset": "timdettmers/openassistant-guanaco",
        },
        stages=[StageSpec(name="inspect"), StageSpec(name="evaluate")],
        expected_artifacts=["adapter"],
    )

    result = LLMTRLAdapter().evaluate(plan, run_dir=str(tmp_path))

    assert result["metrics"]["status"] == "completed"
    assert result["metrics"]["sample_count"] == 1
    assert result["samples"][0]["completion"] == "c"


def test_llm_evaluate_builds_improvement_verdict_from_modal_metrics(tmp_path):
    output = tmp_path / "modal_artifacts" / "sft_output"
    output.mkdir(parents=True)
    (output / "sample_generations.json").write_text(
        '{"status": "completed", "samples": [{"prompt": "p", "completion": "adapter", "score": null}]}',
        encoding="utf-8",
    )
    (output / "base_generations.json").write_text(
        '{"status": "completed", "samples": [{"prompt": "p", "completion": "base"}]}',
        encoding="utf-8",
    )
    (output / "adapter_generations.json").write_text(
        '{"status": "completed", "samples": [{"prompt": "p", "completion": "adapter"}]}',
        encoding="utf-8",
    )
    (output / "eval_metrics.json").write_text(
        '{"status": "completed", "base": {"loss": 2.0, "perplexity": 7.4, "examples": 20}, '
        '"adapter": {"loss": 1.8, "perplexity": 6.0, "examples": 20}}',
        encoding="utf-8",
    )
    plan = ExperimentPlan(
        plan_id="plan_test",
        domain="llm_trl",
        objective="tiny sft",
        inputs={
            "method": "sft",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "dataset": "timdettmers/openassistant-guanaco",
        },
        stages=[StageSpec(name="inspect"), StageSpec(name="evaluate")],
        expected_artifacts=["adapter"],
    )

    result = LLMTRLAdapter().evaluate(plan, run_dir=str(tmp_path))
    evidence = result["improvement_evidence"]

    assert evidence["verdict"] == "improved"
    assert result["metrics"]["improvement_verdict"] == "improved"
    assert result["base_samples"][0]["completion"] == "base"
    assert result["adapter_samples"][0]["completion"] == "adapter"
    assert (tmp_path / "improvement_evidence.json").exists()


def test_llm_evaluate_marks_regression_from_modal_metrics(tmp_path):
    output = tmp_path / "modal_artifacts" / "sft_output"
    output.mkdir(parents=True)
    (output / "sample_generations.json").write_text(
        '{"status": "completed", "samples": [{"prompt": "p", "completion": "adapter", "score": null}]}',
        encoding="utf-8",
    )
    (output / "eval_metrics.json").write_text(
        '{"status": "completed", "base": {"loss": 1.0, "perplexity": 2.7, "examples": 20}, '
        '"adapter": {"loss": 1.2, "perplexity": 3.3, "examples": 20}}',
        encoding="utf-8",
    )
    plan = ExperimentPlan(
        plan_id="plan_test",
        domain="llm_trl",
        objective="tiny sft",
        inputs={
            "method": "sft",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "dataset": "timdettmers/openassistant-guanaco",
        },
        stages=[StageSpec(name="inspect"), StageSpec(name="evaluate")],
        expected_artifacts=["adapter"],
    )

    result = LLMTRLAdapter().evaluate(plan, run_dir=str(tmp_path))

    assert result["improvement_evidence"]["verdict"] == "regressed"


def test_llm_report_includes_sample_generation(tmp_path):
    (tmp_path / "llm_dataset_inspect.json").write_text('{"valid": true}', encoding="utf-8")
    (tmp_path / "llm_smoke_test.json").write_text('{"passed": true}', encoding="utf-8")
    (tmp_path / "llm_eval.json").write_text(
        '{"metrics": {"status": "completed", "sample_count": 1}, '
        '"samples": [{"prompt": "p", "completion": "c"}]}',
        encoding="utf-8",
    )
    plan = ExperimentPlan(
        plan_id="plan_test",
        domain="llm_trl",
        objective="tiny sft",
        inputs={
            "method": "sft",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "dataset": "timdettmers/openassistant-guanaco",
        },
        stages=[StageSpec(name="inspect"), StageSpec(name="report")],
        expected_artifacts=["adapter"],
    )

    result = LLMTRLAdapter().report(plan, run_dir=str(tmp_path))
    report = (tmp_path / "llm_report.md").read_text(encoding="utf-8")

    assert result["report_path"].endswith("llm_report.md")
    assert "Sample Generations" in report
    assert "Completion: c" in report


def test_llm_report_does_not_overclaim_without_evidence(tmp_path):
    (tmp_path / "llm_dataset_inspect.json").write_text('{"valid": true}', encoding="utf-8")
    (tmp_path / "llm_smoke_test.json").write_text('{"passed": true}', encoding="utf-8")
    (tmp_path / "llm_eval.json").write_text(
        '{"metrics": {"status": "completed", "sample_count": 1}, '
        '"samples": [{"prompt": "p", "completion": "c"}]}',
        encoding="utf-8",
    )
    plan = ExperimentPlan(
        plan_id="plan_test",
        domain="llm_trl",
        objective="tiny sft",
        inputs={
            "method": "sft",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "dataset": "timdettmers/openassistant-guanaco",
        },
        stages=[StageSpec(name="inspect"), StageSpec(name="report")],
        expected_artifacts=["adapter"],
    )

    LLMTRLAdapter().report(plan, run_dir=str(tmp_path))
    report = (tmp_path / "llm_report.md").read_text(encoding="utf-8")

    assert "Improvement verdict: `inconclusive`" in report
    assert "model improvement is not established" in report
    assert "Fine-tuning improved the model" not in report


def test_llm_report_includes_base_vs_adapter_metrics_and_training_state(tmp_path):
    output = tmp_path / "modal_artifacts" / "sft_output"
    state_dir = output / "checkpoint-5"
    state_dir.mkdir(parents=True)
    (tmp_path / "llm_dataset_inspect.json").write_text('{"valid": true}', encoding="utf-8")
    (tmp_path / "llm_smoke_test.json").write_text('{"passed": true}', encoding="utf-8")
    (tmp_path / "llm_eval.json").write_text(
        '{"metrics": {"status": "completed", "sample_count": 1}, '
        '"eval_metrics": {"base": {"loss": 2.0, "perplexity": 7.4, "examples": 20}, '
        '"adapter": {"loss": 1.8, "perplexity": 6.0, "examples": 20}}, '
        '"improvement_evidence": {"verdict": "improved", "reason": "Adapter held-out loss improved by 10.00%."}, '
        '"base_samples": [{"prompt": "p", "completion": "base"}], '
        '"adapter_samples": [{"prompt": "p", "completion": "adapter"}], '
        '"samples": [{"prompt": "p", "completion": "adapter"}]}',
        encoding="utf-8",
    )
    (state_dir / "trainer_state.json").write_text(
        '{"log_history": [{"step": 1, "loss": 1.5, "learning_rate": 0.0002, "mean_token_accuracy": 0.62}]}',
        encoding="utf-8",
    )
    plan = ExperimentPlan(
        plan_id="plan_test",
        domain="llm_trl",
        objective="tiny sft",
        inputs={
            "method": "sft",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "dataset": "timdettmers/openassistant-guanaco",
        },
        stages=[StageSpec(name="inspect"), StageSpec(name="report")],
        expected_artifacts=["adapter"],
    )

    LLMTRLAdapter().report(plan, run_dir=str(tmp_path))
    report = (tmp_path / "llm_report.md").read_text(encoding="utf-8")

    assert "Improvement verdict: `improved`" in report
    assert "| Base | 2.0000 | 7.4000 | 20 |" in report
    assert "| Adapter | 1.8000 | 6.0000 | 20 |" in report
    assert "| 1 | 1.5000 | 0.0002 | 0.6200 |" in report
    assert "Base vs Adapter Samples" in report
