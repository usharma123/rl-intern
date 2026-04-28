import asyncio

from agent.tools import orchestrator
from agent.tools.orchestrator import create_experiment_plan, run_experiment_stage, validate_experiment_plan


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
