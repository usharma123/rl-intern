from typing import Optional

from pydantic import BaseModel, Field


class EnvInspectionResult(BaseModel):
    env_id: str
    observation_space: str
    action_space: str
    action_space_type: str
    observation_space_type: str
    is_discrete_action: bool
    is_continuous_action: bool
    max_episode_steps: Optional[int] = None
    reward_range: Optional[str] = None
    render_modes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
