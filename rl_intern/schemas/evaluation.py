from pydantic import BaseModel


class EvaluationResult(BaseModel):
    env_id: str
    algorithm: str
    episodes: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    seed: int
