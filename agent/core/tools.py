import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from agent.tools.algorithm_select import choose_algorithm
from agent.tools.common import tool_handler
from agent.tools.env_inspect import inspect_env
from agent.tools.env_smoke_test import smoke_test_env
from agent.tools.evaluate_policy import evaluate_policy
from agent.tools.random_baseline import run_random_baseline
from agent.tools.record_rollout import record_rollout
from agent.tools.report import generate_report
from agent.tools.train_sb3 import train_sb3

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Optional[Callable[[dict[str, Any]], Awaitable[tuple[str, bool]]]] = None


class ToolRouter:
    """Routes built-in RL tool calls."""

    def __init__(
        self,
        mcp_servers: dict[str, Any] | None = None,
        local_mode: bool = False,
        **_: Any,
    ):
        self.tools: dict[str, ToolSpec] = {}
        self.mcp_servers = mcp_servers or {}
        self.local_mode = local_mode
        for tool in create_builtin_tools(local_mode=local_mode):
            self.register_tool(tool)

    def register_tool(self, tool: ToolSpec) -> None:
        self.tools[tool.name] = tool

    async def __aenter__(self) -> "ToolRouter":
        logger.info("Agent ready with %d built-in RL tools", len(self.tools))
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def get_tool_specs_for_llm(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools.values()
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session: Any = None,
        tool_call_id: str | None = None,
    ) -> tuple[str, bool]:
        tool = self.tools.get(tool_name)
        if not tool or not tool.handler:
            return f"Unknown tool: {tool_name}", False
        return await tool.handler(arguments)


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "required": required or [],
        "additionalProperties": False,
        "properties": properties,
    }


def create_builtin_tools(local_mode: bool = False) -> list[ToolSpec]:
    tools = [
        ToolSpec(
            name="inspect_env",
            description="Inspect a Gymnasium environment's observation/action spaces, metadata, render modes, and warnings.",
            parameters=_schema(
                {"env_id": {"type": "string", "description": "Gymnasium environment ID."}},
                ["env_id"],
            ),
            handler=tool_handler(inspect_env),
        ),
        ToolSpec(
            name="smoke_test_env",
            description="Run random actions through a Gymnasium environment to validate reset, step, observations, rewards, and termination.",
            parameters=_schema(
                {
                    "env_id": {"type": "string"},
                    "episodes": {"type": "integer", "default": 3},
                    "max_steps": {"type": "integer", "default": 1000},
                    "seed": {"type": "integer", "default": 0},
                },
                ["env_id"],
            ),
            handler=tool_handler(smoke_test_env),
        ),
        ToolSpec(
            name="choose_algorithm",
            description="Choose RL algorithms compatible with a Gymnasium environment, optionally validating a user-requested algorithm.",
            parameters=_schema(
                {
                    "env_id": {"type": "string"},
                    "user_preference": {
                        "type": ["string", "null"],
                        "description": "Optional requested algorithm such as PPO or DQN.",
                    },
                },
                ["env_id"],
            ),
            handler=tool_handler(choose_algorithm),
        ),
        ToolSpec(
            name="run_random_baseline",
            description="Evaluate a random-action baseline across episodes and return reward statistics.",
            parameters=_schema(
                {
                    "env_id": {"type": "string"},
                    "episodes": {"type": "integer", "default": 10},
                    "seed": {"type": "integer", "default": 0},
                },
                ["env_id"],
            ),
            handler=tool_handler(run_random_baseline),
        ),
        ToolSpec(
            name="train_sb3",
            description="Train a Stable-Baselines3 PPO or DQN model locally and save model/config artifacts.",
            parameters=_schema(
                {
                    "env_id": {"type": "string"},
                    "algorithm": {"type": "string", "enum": ["PPO", "DQN"], "default": "PPO"},
                    "total_timesteps": {"type": "integer", "default": 100000},
                    "seed": {"type": "integer", "default": 0},
                    "output_dir": {"type": "string", "default": "artifacts"},
                    "log_dir": {"type": "string", "default": "runs"},
                },
                ["env_id"],
            ),
            handler=tool_handler(train_sb3),
        ),
        ToolSpec(
            name="evaluate_policy",
            description="Load a saved SB3 policy and evaluate it deterministically across episodes.",
            parameters=_schema(
                {
                    "env_id": {"type": "string"},
                    "algorithm": {"type": "string", "enum": ["PPO", "DQN"]},
                    "model_path": {"type": "string"},
                    "episodes": {"type": "integer", "default": 20},
                    "seed": {"type": "integer", "default": 0},
                },
                ["env_id", "algorithm", "model_path"],
            ),
            handler=tool_handler(evaluate_policy),
        ),
        ToolSpec(
            name="record_rollout",
            description="Record one rollout from a saved SB3 policy using rgb_array rendering and save a video.",
            parameters=_schema(
                {
                    "env_id": {"type": "string"},
                    "algorithm": {"type": "string", "enum": ["PPO", "DQN"]},
                    "model_path": {"type": "string"},
                    "seed": {"type": "integer", "default": 0},
                    "max_steps": {"type": "integer", "default": 1000},
                    "output_dir": {"type": "string", "default": "artifacts"},
                },
                ["env_id", "algorithm", "model_path"],
            ),
            handler=tool_handler(record_rollout),
        ),
        ToolSpec(
            name="generate_report",
            description="Generate a Markdown RL experiment report from environment, baseline, training, evaluation, and rollout results.",
            parameters=_schema(
                {
                    "env_result": {"type": "object"},
                    "smoke_test_result": {"type": "object"},
                    "random_baseline_result": {"type": "object"},
                    "training_result": {"type": "object"},
                    "evaluation_result": {"type": "object"},
                    "rollout_result": {"type": ["object", "null"]},
                    "output_path": {"type": ["string", "null"]},
                },
                [
                    "env_result",
                    "smoke_test_result",
                    "random_baseline_result",
                    "training_result",
                    "evaluation_result",
                ],
            ),
            handler=tool_handler(generate_report),
        ),
    ]
    logger.info("Loaded %d built-in RL tools", len(tools))
    return tools
