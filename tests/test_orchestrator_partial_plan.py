import asyncio

from agent.tools import orchestrator
from agent.tools.orchestrator import (
    create_experiment_plan,
    get_artifact_manifest_handler,
    run_experiment_stage,
    run_experiment_stage_handler,
    update_experiment_plan,
    validate_experiment_plan,
)


def test_partial_plan_recovers_saved_plan_and_aliases(tmp_path):
    created = create_experiment_plan(
        domain="llm_trl",
        objective="tiny sft",
        inputs={
            "method": "sft",
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "dataset_name": "dummy",
        },
        run_dir=str(tmp_path),
    )

    partial = {
        "plan_id": created["plan_id"],
        "domain": "llm_trl",
        "objective": created["objective"],
        "inputs": created["inputs"],
    }
    validated = validate_experiment_plan(partial, run_dir=str(tmp_path))

    assert validated["valid"] is True
    assert validated["plan"]["inputs"]["model"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    result = run_experiment_stage(partial, "prepare", run_dir=str(tmp_path))

    assert result["result"]["method"] == "sft"
    assert result["result"]["script_path"].endswith("train_trl.py")


def test_llm_use_modal_inputs_normalize_runner(tmp_path):
    created = create_experiment_plan(
        domain="llm_trl",
        objective="tiny sft",
        inputs={
            "method": "sft",
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "dataset_name": "dummy",
            "use_modal": True,
            "modal_gpu": "t4",
        },
        run_dir=str(tmp_path),
    )

    assert created["runner"]["backend"] == "modal"
    assert created["runner"]["hardware"] == "gpu-t4"


def test_create_experiment_plan_adds_inspect_before_train(tmp_path):
    created = create_experiment_plan(
        domain="llm_trl",
        objective="tiny sft",
        inputs={"method": "sft", "model": "m", "dataset": "d"},
        stages=["prepare", "train", "evaluate", "report"],
        run_dir=str(tmp_path),
    )

    assert [stage["name"] for stage in created["stages"]] == [
        "inspect",
        "prepare",
        "train",
        "evaluate",
        "report",
    ]


def test_create_experiment_plan_handler_returns_validation_errors():
    output, success = asyncio.run(
        orchestrator.create_experiment_plan_handler(
            {
                "domain": "llm_trl",
                "objective": "bad grpo",
                "inputs": {"method": "grpo", "model": "m", "dataset": "d"},
                "stages": ["inspect", "train"],
                "reward": {"type": "none"},
                "expected_artifacts": ["adapter"],
            }
        )
    )

    assert success is False
    assert "python_verifier" in output
    assert "hint" in output


def test_partial_plan_preserves_saved_modal_runner(tmp_path):
    created = create_experiment_plan(
        domain="llm_trl",
        objective="tiny sft",
        inputs={"method": "sft", "model": "m", "dataset": "d", "use_modal": True, "modal_gpu": "t4"},
        run_dir=str(tmp_path),
    )
    partial = {
        "plan_id": created["plan_id"],
        "domain": "llm_trl",
        "objective": created["objective"],
        "inputs": created["inputs"],
    }

    validated = validate_experiment_plan(partial, run_dir=str(tmp_path))

    assert validated["valid"] is True
    assert validated["plan"]["runner"]["backend"] == "modal"
    assert validated["plan"]["runner"]["hardware"] == "gpu-t4"


def test_partial_plan_train_uses_saved_modal_runner(monkeypatch, tmp_path):
    created = create_experiment_plan(
        domain="llm_trl",
        objective="tiny sft",
        inputs={"method": "sft", "model": "m", "dataset": "d", "use_modal": True, "modal_gpu": "t4"},
        run_dir=str(tmp_path),
    )
    partial = {
        "plan_id": created["plan_id"],
        "domain": "llm_trl",
        "objective": created["objective"],
        "inputs": created["inputs"],
    }
    captured = {}

    def fake_run_modal_job(**kwargs):
        captured.update(kwargs)
        return {"status": "running", "backend_id": "fc-test"}

    monkeypatch.setattr("rl_intern.runners.modal_backend.run_modal_job", fake_run_modal_job)

    result = run_experiment_stage(partial, "train", run_dir=str(tmp_path))

    assert result["result"]["status"] == "running"
    assert captured["hardware"] == "gpu-t4"


def test_run_experiment_stage_handler_treats_running_job_as_success(monkeypatch, tmp_path):
    created = create_experiment_plan(
        domain="llm_trl",
        objective="tiny sft",
        inputs={"method": "sft", "model": "m", "dataset": "d", "use_modal": True, "modal_gpu": "t4"},
        run_dir=str(tmp_path),
    )

    def fake_run_modal_job(**kwargs):
        return {"status": "running", "error": None, "backend_id": "fc-test"}

    monkeypatch.setattr("rl_intern.runners.modal_backend.run_modal_job", fake_run_modal_job)

    output, success = asyncio.run(
        run_experiment_stage_handler({"plan": created, "stage": "train", "run_dir": str(tmp_path)})
    )

    assert success is True
    assert '"status": "running"' in output


def test_run_experiment_stage_handler_returns_gym_missing_env_id_error():
    output, success = asyncio.run(
        run_experiment_stage_handler(
            {
                "plan": {
                    "plan_id": "plan_bad",
                    "domain": "gym_sb3",
                    "objective": "train a gym env",
                    "inputs": {},
                    "stages": [{"name": "inspect"}],
                    "expected_artifacts": ["model.zip"],
                },
                "stage": "inspect",
            }
        )
    )

    assert success is False
    assert "inputs.env_id" in output
    assert "CartPole-v1" in output


def test_artifact_manifest_handler_returns_summary_not_full_context(tmp_path):
    create_experiment_plan(
        domain="llm_trl",
        objective="tiny sft",
        inputs={"method": "sft", "model": "m", "dataset": "d"},
        run_dir=str(tmp_path),
    )
    for i in range(20):
        (tmp_path / f"metric_{i}.json").write_text("{}", encoding="utf-8")
        from rl_intern.orchestrator.manifest import append_manifest_item

        append_manifest_item(tmp_path, "metrics", tmp_path / f"metric_{i}.json")

    output, success = asyncio.run(get_artifact_manifest_handler({"run_dir": str(tmp_path)}))

    assert success is True
    assert '"counts"' in output
    assert "metric_0.json" not in output
    assert "metric_19.json" in output


def test_update_experiment_plan_structurally_updates_runner(tmp_path):
    created = create_experiment_plan(
        domain="llm_trl",
        objective="tiny sft",
        inputs={"method": "sft", "model": "m", "dataset": "d"},
        run_dir=str(tmp_path),
    )

    updated = update_experiment_plan(
        run_dir=str(tmp_path),
        plan=created,
        updates={"runner": {"backend": "modal", "hardware": "t4"}},
    )

    assert updated["runner"]["backend"] == "modal"
    assert updated["runner"]["hardware"] == "gpu-t4"


def test_run_experiment_stage_handler_uses_thread(monkeypatch, tmp_path):
    created = create_experiment_plan(
        domain="llm_trl",
        objective="tiny sft",
        inputs={"method": "sft", "model": "m", "dataset": "d"},
        run_dir=str(tmp_path),
    )
    called = {"to_thread": False}

    async def fake_to_thread(func, **kwargs):
        called["to_thread"] = True
        return func(**kwargs)

    monkeypatch.setattr(orchestrator.asyncio, "to_thread", fake_to_thread)

    output, success = asyncio.run(
        orchestrator.run_experiment_stage_handler(
            {"plan": created, "stage": "prepare", "run_dir": str(tmp_path)}
        )
    )

    assert success is True
    assert "train_trl.py" in output
    assert called["to_thread"] is True


def test_gym_sb3_stage_handler_uses_main_thread_on_macos(monkeypatch, tmp_path):
    created = create_experiment_plan(
        domain="gym_sb3",
        objective="cartpole",
        inputs={"env_id": "CartPole-v1", "algorithm": "PPO"},
        run_dir=str(tmp_path),
    )
    called = {"stage": False}

    async def fail_to_thread(func, **kwargs):
        raise AssertionError("gym_sb3 stages must not run in a worker thread on macOS")

    def fake_run_experiment_stage(**kwargs):
        called["stage"] = True
        return {"stage": kwargs["stage"], "domain": "gym_sb3", "result": {"ok": True}}

    monkeypatch.setattr(orchestrator.sys, "platform", "darwin")
    monkeypatch.setattr(orchestrator.asyncio, "to_thread", fail_to_thread)
    monkeypatch.setattr(orchestrator, "run_experiment_stage", fake_run_experiment_stage)

    output, success = asyncio.run(
        orchestrator.run_experiment_stage_handler(
            {"plan": created, "stage": "inspect", "run_dir": str(tmp_path)}
        )
    )

    assert success is True
    assert '"domain": "gym_sb3"' in output
    assert called["stage"] is True
