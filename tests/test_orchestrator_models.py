from pathlib import Path

import pytest

from rl_intern.orchestrator.manifest import append_manifest_item, load_manifest
from rl_intern.orchestrator.models import ExperimentPlan, RewardSpec, RunnerSpec, StageSpec


def test_gym_plan_validates_with_expected_artifacts():
    plan = ExperimentPlan(
        plan_id="plan_test",
        domain="gym_sb3",
        objective="train cartpole",
        inputs={"env_id": "CartPole-v1", "algorithm": "PPO"},
        reward=RewardSpec(type="environment"),
        runner=RunnerSpec(backend="local"),
        stages=[StageSpec(name="inspect"), StageSpec(name="train")],
        expected_artifacts=["model.zip"],
    )

    assert plan.domain == "gym_sb3"


def test_train_requires_inspect_stage():
    with pytest.raises(ValueError, match="inspect"):
        ExperimentPlan(
            plan_id="plan_bad",
            domain="gym_sb3",
            objective="bad",
            inputs={"env_id": "CartPole-v1"},
            stages=[StageSpec(name="train")],
            expected_artifacts=["model.zip"],
        )


def test_grpo_requires_python_verifier_reward():
    with pytest.raises(ValueError, match="python_verifier"):
        ExperimentPlan(
            plan_id="plan_bad",
            domain="llm_trl",
            objective="grpo",
            inputs={"method": "grpo", "dataset": "x", "model": "y"},
            stages=[StageSpec(name="inspect"), StageSpec(name="train")],
            expected_artifacts=["adapter"],
        )


def test_manifest_append_groups_artifacts(tmp_path: Path):
    artifact = tmp_path / "eval.json"
    artifact.write_text("{}", encoding="utf-8")

    append_manifest_item(tmp_path, "metrics", artifact, kind="evaluation", run_id="run_test")
    manifest = load_manifest(tmp_path)

    assert manifest.run_id == "run_test"
    assert manifest.metrics[0].path == str(artifact)
