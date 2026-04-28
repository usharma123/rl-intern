from typing import Any, Optional

from pydantic import BaseModel


class ReportInput(BaseModel):
    env_result: dict[str, Any]
    smoke_test_result: dict[str, Any]
    random_baseline_result: dict[str, Any]
    training_result: dict[str, Any]
    evaluation_result: dict[str, Any]
    rollout_result: Optional[dict[str, Any]] = None
    output_path: Optional[str] = None


class ReportResult(BaseModel):
    report_path: str
