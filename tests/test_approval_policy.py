from agent.core.agent_loop import _display_tool_name, _needs_approval


def test_run_experiment_stage_only_train_needs_approval():
    assert _needs_approval("run_experiment_stage", args={"stage": "inspect"}) is False
    assert _needs_approval("run_experiment_stage", args={"stage": "prepare"}) is False
    assert _needs_approval("run_experiment_stage", args={"stage": "smoke_test"}) is False
    assert _needs_approval("run_experiment_stage", args={"stage": "evaluate"}) is False
    assert _needs_approval("run_experiment_stage", args={"stage": "report"}) is False
    assert _needs_approval("run_experiment_stage", args={"stage": "train"}) is True


def test_modal_job_still_needs_approval():
    assert _needs_approval("modal_job_run") is True


def test_run_experiment_stage_display_name_includes_stage():
    assert _display_tool_name("run_experiment_stage", {"stage": "inspect"}) == "stage:inspect"
    assert _display_tool_name("run_experiment_stage", {"stage": "report"}) == "stage:report"
    assert _display_tool_name("modal_job_run", {"stage": "train"}) == "modal_job_run"
