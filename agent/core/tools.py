import logging
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from agent.tools.algorithm_select import choose_algorithm
from agent.tools.common import tool_handler
from agent.tools.env_inspect import inspect_env
from agent.tools.env_smoke_test import smoke_test_env
from agent.tools.evaluate_policy import evaluate_policy
from agent.tools.modal_runner import (
    fetch_modal_artifacts,
    get_modal_run_status,
    launch_modal_experiment,
)
from agent.tools.llm_trl import (
    generate_trl_script_handler,
    inspect_llm_dataset_handler,
    validate_grpo_verifier_handler,
)
from agent.tools.modal_primitives import (
    modal_job_artifacts_handler,
    modal_job_cancel_handler,
    modal_job_logs_handler,
    modal_job_run_handler,
    modal_job_status_handler,
    modal_sandbox_create_handler,
    modal_sandbox_edit_handler,
    modal_sandbox_exec_handler,
    modal_sandbox_read_handler,
    modal_sandbox_terminate_handler,
    modal_sandbox_write_handler,
)
from agent.tools.orchestrator import (
    create_experiment_plan_handler,
    get_artifact_manifest_handler,
    list_domain_adapters_handler,
    run_experiment_stage_handler,
    update_experiment_plan_handler,
    validate_experiment_plan_handler,
)
from agent.tools.random_baseline import run_random_baseline
from agent.tools.record_rollout import record_rollout
from agent.tools.report import generate_report
from agent.tools.research_tools import (
    docs_fetch_handler,
    docs_search_handler,
    github_find_examples_handler,
    github_read_file_handler,
    hf_repo_files_handler,
    paper_search_handler,
    research_handler,
    web_search_handler,
)
from agent.tools.run_files import (
    edit_run_file_handler,
    read_run_file_handler,
    write_run_file_handler,
)
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
        sig = inspect.signature(tool.handler)
        kwargs = {}
        if "session" in sig.parameters:
            kwargs["session"] = session
        if "tool_call_id" in sig.parameters:
            kwargs["tool_call_id"] = tool_call_id
        return await tool.handler(arguments, **kwargs)


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
            name="create_experiment_plan",
            description="Create and persist a typed RL experiment plan before heavy execution.",
            parameters=_schema(
                {
                    "domain": {"type": "string", "enum": ["gym_sb3", "llm_trl"]},
                    "objective": {"type": "string"},
                    "inputs": {"type": "object"},
                    "stages": {
                        "type": ["array", "null"],
                        "items": {
                            "type": "string",
                            "enum": [
                                "inspect",
                                "prepare",
                                "smoke_test",
                                "train",
                                "evaluate",
                                "report",
                                "publish_optional",
                            ],
                        },
                    },
                    "reward": {"type": ["object", "null"]},
                    "runner": {"type": ["object", "null"]},
                    "expected_artifacts": {"type": ["array", "null"], "items": {"type": "string"}},
                    "research_required": {"type": "boolean", "default": False},
                    "research_completed": {"type": "boolean", "default": False},
                    "run_dir": {"type": ["string", "null"]},
                },
                ["domain", "objective", "inputs"],
            ),
            handler=create_experiment_plan_handler,
        ),
        ToolSpec(
            name="validate_experiment_plan",
            description="Validate an ExperimentPlan and return adapter artifact expectations.",
            parameters=_schema({"plan": {"type": "object"}, "run_dir": {"type": ["string", "null"]}}, ["plan"]),
            handler=validate_experiment_plan_handler,
        ),
        ToolSpec(
            name="update_experiment_plan",
            description="Structurally update and persist an existing ExperimentPlan without string-editing JSON.",
            parameters=_schema(
                {
                    "run_dir": {"type": ["string", "null"]},
                    "plan": {"type": ["object", "null"]},
                    "updates": {"type": "object"},
                },
                ["updates"],
            ),
            handler=update_experiment_plan_handler,
        ),
        ToolSpec(
            name="run_experiment_stage",
            description="Run one validated experiment stage through the domain adapter.",
            parameters=_schema(
                {
                    "plan": {"type": "object"},
                    "stage": {
                        "type": "string",
                        "enum": ["inspect", "prepare", "smoke_test", "train", "evaluate", "report"],
                    },
                    "run_dir": {"type": ["string", "null"]},
                },
                ["plan", "stage"],
            ),
            handler=run_experiment_stage_handler,
        ),
        ToolSpec(
            name="get_artifact_manifest",
            description="Read the structured artifact manifest for a run directory.",
            parameters=_schema({"run_dir": {"type": "string"}}, ["run_dir"]),
            handler=get_artifact_manifest_handler,
        ),
        ToolSpec(
            name="list_domain_adapters",
            description="List available RL domain adapters and artifact schemas.",
            parameters=_schema({}),
            handler=list_domain_adapters_handler,
        ),
        ToolSpec(
            name="research",
            description="Research papers, docs, and web context for an RL/ML task before implementing training.",
            parameters=_schema(
                {
                    "task": {"type": "string"},
                    "library": {"type": ["string", "null"]},
                },
                ["task"],
            ),
            handler=research_handler,
        ),
        ToolSpec(
            name="web_search",
            description="Search the web for current RL/ML implementation context.",
            parameters=_schema({"query": {"type": "string"}}, ["query"]),
            handler=web_search_handler,
        ),
        ToolSpec(
            name="paper_search",
            description="Search Semantic Scholar for papers relevant to an RL/ML training recipe.",
            parameters=_schema(
                {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
                ["query"],
            ),
            handler=paper_search_handler,
        ),
        ToolSpec(
            name="docs_search",
            description="Search documentation for a library/API before generating training code.",
            parameters=_schema(
                {"library": {"type": "string"}, "query": {"type": "string"}},
                ["library", "query"],
            ),
            handler=docs_search_handler,
        ),
        ToolSpec(
            name="docs_fetch",
            description="Fetch documentation page text by URL.",
            parameters=_schema({"url": {"type": "string"}}, ["url"]),
            handler=docs_fetch_handler,
        ),
        ToolSpec(
            name="github_find_examples",
            description="Find Python implementation examples in GitHub repositories.",
            parameters=_schema(
                {
                    "query": {"type": ["string", "null"]},
                    "repo": {"type": ["string", "null"]},
                    "keyword": {"type": ["string", "null"]},
                    "limit": {"type": "integer", "default": 5},
                },
            ),
            handler=github_find_examples_handler,
        ),
        ToolSpec(
            name="github_read_file",
            description="Read a raw file from a GitHub repository.",
            parameters=_schema(
                {
                    "repo": {"type": "string"},
                    "path": {"type": "string"},
                    "ref": {"type": "string", "default": "main"},
                },
                ["repo", "path"],
            ),
            handler=github_read_file_handler,
        ),
        ToolSpec(
            name="hf_repo_files",
            description="List or read files from a Hugging Face Hub repository.",
            parameters=_schema(
                {
                    "repo_id": {"type": "string"},
                    "repo_type": {"type": "string", "enum": ["model", "dataset", "space"], "default": "model"},
                    "path": {"type": ["string", "null"]},
                },
                ["repo_id"],
            ),
            handler=hf_repo_files_handler,
        ),
        ToolSpec(
            name="inspect_llm_dataset",
            description="Inspect SFT, DPO, or GRPO dataset format for TRL post-training.",
            parameters=_schema(
                {
                    "dataset_path": {"type": ["string", "null"]},
                    "rows": {"type": ["array", "null"], "items": {"type": "object"}},
                    "method": {"type": "string", "enum": ["sft", "dpo", "grpo"], "default": "sft"},
                    "sample_rows": {"type": "integer", "default": 3},
                },
            ),
            handler=inspect_llm_dataset_handler,
        ),
        ToolSpec(
            name="validate_grpo_verifier",
            description="Validate a Python GRPO verifier with score(example, completion) -> number or {'score': number}.",
            parameters=_schema(
                {
                    "verifier_path": {"type": ["string", "null"]},
                    "verifier_source": {"type": ["string", "null"]},
                    "example": {"type": ["object", "null"]},
                    "completion": {"type": "string", "default": "test completion"},
                },
            ),
            handler=validate_grpo_verifier_handler,
        ),
        ToolSpec(
            name="generate_trl_script",
            description="Generate a baseline TRL SFT/DPO/GRPO training script.",
            parameters=_schema({"method": {"type": "string", "enum": ["sft", "dpo", "grpo"]}}, ["method"]),
            handler=generate_trl_script_handler,
        ),
        ToolSpec(
            name="read_run_file",
            description="Read a local file inside a run directory, such as a generated training script or log.",
            parameters=_schema({"run_dir": {"type": "string"}, "path": {"type": "string"}}, ["run_dir", "path"]),
            handler=read_run_file_handler,
        ),
        ToolSpec(
            name="write_run_file",
            description="Write a local file inside a run directory, such as an edited training script.",
            parameters=_schema(
                {"run_dir": {"type": "string"}, "path": {"type": "string"}, "content": {"type": "string"}},
                ["run_dir", "path", "content"],
            ),
            handler=write_run_file_handler,
        ),
        ToolSpec(
            name="edit_run_file",
            description="Edit a local file inside a run directory by string replacement.",
            parameters=_schema(
                {
                    "run_dir": {"type": "string"},
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                    "replace_all": {"type": "boolean", "default": False},
                },
                ["run_dir", "path", "old_str", "new_str"],
            ),
            handler=edit_run_file_handler,
        ),
        ToolSpec(
            name="modal_sandbox_create",
            description="Create a persistent Modal sandbox for iterative script development.",
            parameters=_schema(
                {
                    "run_id": {"type": "string"},
                    "hardware": {"type": "string", "default": "cpu-basic"},
                    "image": {"type": ["string", "null"]},
                    "timeout": {"type": "integer", "default": 21600},
                },
                ["run_id"],
            ),
            handler=modal_sandbox_create_handler,
        ),
        ToolSpec(
            name="modal_sandbox_exec",
            description="Execute a shell command inside an existing Modal sandbox.",
            parameters=_schema(
                {
                    "sandbox_id": {"type": "string"},
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "default": 120},
                },
                ["sandbox_id", "command"],
            ),
            handler=modal_sandbox_exec_handler,
        ),
        ToolSpec(
            name="modal_sandbox_read",
            description="Read a file from an existing Modal sandbox.",
            parameters=_schema({"sandbox_id": {"type": "string"}, "path": {"type": "string"}}, ["sandbox_id", "path"]),
            handler=modal_sandbox_read_handler,
        ),
        ToolSpec(
            name="modal_sandbox_write",
            description="Write a file into an existing Modal sandbox.",
            parameters=_schema(
                {"sandbox_id": {"type": "string"}, "path": {"type": "string"}, "content": {"type": "string"}},
                ["sandbox_id", "path", "content"],
            ),
            handler=modal_sandbox_write_handler,
        ),
        ToolSpec(
            name="modal_sandbox_edit",
            description="Edit a file in an existing Modal sandbox by string replacement.",
            parameters=_schema(
                {
                    "sandbox_id": {"type": "string"},
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                    "replace_all": {"type": "boolean", "default": False},
                },
                ["sandbox_id", "path", "old_str", "new_str"],
            ),
            handler=modal_sandbox_edit_handler,
        ),
        ToolSpec(
            name="modal_sandbox_terminate",
            description="Terminate an existing Modal sandbox.",
            parameters=_schema({"sandbox_id": {"type": "string"}}, ["sandbox_id"]),
            handler=modal_sandbox_terminate_handler,
        ),
        ToolSpec(
            name="modal_job_run",
            description="Launch a detached generic Modal job for heavy RL/LLM work.",
            parameters=_schema(
                {
                    "run_id": {"type": "string"},
                    "stage": {"type": "string"},
                    "run_dir": {"type": ["string", "null"]},
                    "script_path": {"type": ["string", "null"]},
                    "script": {"type": ["string", "null"]},
                    "command": {"type": ["array", "string", "null"], "items": {"type": "string"}},
                    "script_args": {"type": ["array", "null"], "items": {"type": "string"}},
                    "dependencies": {"type": ["array", "null"], "items": {"type": "string"}},
                    "hardware": {"type": "string", "default": "cpu-basic"},
                    "timeout": {"type": "string", "default": "30m"},
                    "env": {"type": ["object", "null"]},
                    "secrets": {"type": ["object", "null"]},
                },
                ["run_id", "stage"],
            ),
            handler=modal_job_run_handler,
        ),
        ToolSpec(
            name="modal_job_status",
            description="Poll detached Modal job status.",
            parameters=_schema(
                {
                    "backend_id": {"type": "string"},
                    "run_dir": {"type": ["string", "null"]},
                    "timeout": {"type": "number", "default": 0},
                },
                ["backend_id"],
            ),
            handler=modal_job_status_handler,
        ),
        ToolSpec(
            name="modal_job_logs",
            description="Fetch detached Modal job stdout/stderr logs.",
            parameters=_schema(
                {
                    "backend_id": {"type": "string"},
                    "run_dir": {"type": ["string", "null"]},
                    "timeout": {"type": "number", "default": 0},
                },
                ["backend_id"],
            ),
            handler=modal_job_logs_handler,
        ),
        ToolSpec(
            name="modal_job_cancel",
            description="Cancel a detached Modal job.",
            parameters=_schema(
                {"backend_id": {"type": "string"}, "run_dir": {"type": ["string", "null"]}},
                ["backend_id"],
            ),
            handler=modal_job_cancel_handler,
        ),
        ToolSpec(
            name="modal_job_artifacts",
            description="Fetch detached Modal job artifacts into the local run directory.",
            parameters=_schema(
                {
                    "backend_id": {"type": "string"},
                    "run_dir": {"type": "string"},
                    "timeout": {"type": "number", "default": 0},
                },
                ["backend_id", "run_dir"],
            ),
            handler=modal_job_artifacts_handler,
        ),
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
        ToolSpec(
            name="launch_modal_experiment",
            description="Launch a trusted remote Modal job for a full SB3 RL experiment and return Modal job metadata.",
            parameters=_schema(
                {
                    "env_id": {"type": "string"},
                    "algorithm": {"type": "string", "enum": ["PPO", "DQN"], "default": "PPO"},
                    "total_timesteps": {"type": "integer", "default": 100000},
                    "seed": {"type": "integer", "default": 0},
                    "eval_episodes": {"type": "integer", "default": 20},
                    "max_steps": {"type": "integer", "default": 1000},
                    "run_id": {"type": ["string", "null"]},
                },
                ["env_id"],
            ),
            handler=tool_handler(launch_modal_experiment),
        ),
        ToolSpec(
            name="get_modal_run_status",
            description="Poll a Modal function call for status or completed RL experiment results.",
            parameters=_schema(
                {
                    "modal_call_id": {"type": "string"},
                    "timeout": {"type": "number", "default": 0},
                },
                ["modal_call_id"],
            ),
            handler=tool_handler(get_modal_run_status),
        ),
        ToolSpec(
            name="fetch_modal_artifacts",
            description="Fetch completed Modal job artifacts into the local run directory.",
            parameters=_schema(
                {
                    "modal_call_id": {"type": "string"},
                    "timeout": {"type": "number", "default": 0},
                },
                ["modal_call_id"],
            ),
            handler=tool_handler(fetch_modal_artifacts),
        ),
    ]
    logger.info("Loaded %d built-in RL tools", len(tools))
    return tools
