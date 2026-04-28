from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from agent.tools.common import artifact_dir


def _default_report_path(
    training_result: dict[str, Any],
    evaluation_result: dict[str, Any],
    output_path: str | None,
) -> Path:
    if output_path:
        return Path(output_path)
    env_id = training_result.get("env_id") or evaluation_result.get("env_id") or "unknown-env"
    algorithm = (
        training_result.get("algorithm")
        or evaluation_result.get("algorithm")
        or "unknown-algorithm"
    )
    seed = int(training_result.get("seed", evaluation_result.get("seed", 0)))
    return artifact_dir(env_id, algorithm, seed) / "report.md"


def generate_report(
    env_result: dict,
    smoke_test_result: dict,
    random_baseline_result: dict,
    training_result: dict,
    evaluation_result: dict,
    rollout_result: dict | None = None,
    output_path: str | None = None,
) -> dict:
    try:
        template_dir = Path(__file__).resolve().parents[2] / "rl_intern" / "templates"
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(default_for_string=False),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.get_template("report.md.j2")
        report_path = _default_report_path(training_result, evaluation_result, output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        content = template.render(
            env_result=env_result,
            smoke_test_result=smoke_test_result,
            random_baseline_result=random_baseline_result,
            training_result=training_result,
            evaluation_result=evaluation_result,
            rollout_result=rollout_result,
        )
        report_path.write_text(content, encoding="utf-8")
        return {"report_path": str(report_path)}
    except Exception as exc:
        return {"error": f"Could not generate report: {exc}"}
