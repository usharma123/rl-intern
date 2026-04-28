from pydantic import BaseModel


class ModalExperimentRequest(BaseModel):
    run_id: str
    env_id: str
    algorithm: str = "PPO"
    total_timesteps: int = 100_000
    seed: int = 0
    eval_episodes: int = 20
    max_steps: int = 1000


class ModalJobReference(BaseModel):
    runner: str = "modal"
    modal_app: str
    modal_function: str
    modal_call_id: str
    status: str
