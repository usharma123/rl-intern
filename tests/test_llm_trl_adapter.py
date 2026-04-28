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

    assert "--max-samples" in script
    assert "--fp16" in script
    assert "bf16=args.bf16" in script
    assert "dtype=torch_dtype" in script
    assert "dataset.select(range(args.max_samples))" in script
    assert "[rl-intern] loading model" in script
    assert "low_cpu_mem_usage=True" in script
    assert "sample_generations.json" in script


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
