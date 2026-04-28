from typing import Optional

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    env_id: str
    algorithm: str
    total_timesteps: int = 100_000
    seed: int = 0
    learning_rate: Optional[float] = None
    gamma: float = 0.99
    log_dir: str = "runs"
    output_dir: str = "artifacts"
