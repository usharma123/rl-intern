# RL Intern

The agent that trains agents.

`rl-intern` is a local web app for running reinforcement learning experiments from a chat interface. The React frontend talks to a local FastAPI run server, which launches the agent runtime, streams tool events over WebSocket, manages approvals, runs local or Modal-backed jobs, and stores reports and artifacts for each run.

## Current Capabilities

- Chat-driven experiment planning and execution
- Local Gymnasium + Stable-Baselines3 workflows
- LLM fine-tuning workflows built around TRL
- Local and Modal runners
- Run monitor for plans, jobs, artifacts, and evidence
- Environment inspection, smoke tests, algorithm selection, training, evaluation, rollout recording, and Markdown reports
- Persistent run history under `artifacts/runs/<run_id>/`

## Prerequisites

- Python 3.11+
- `uv`
- Bun 1.3+
- Modal account and CLI authentication for Modal jobs
- OpenRouter API key for the default agent model
- Hugging Face token for model or dataset access

## Quick Start

```bash
git clone <repo>
cd rl-intern/frontend
bun install
bun run setup
bun run dev
```

Then open http://localhost:5173.

`bun run dev` starts the local run server on `127.0.0.1:8765` if it is not already running, then starts the Vite frontend.

## One-Time Setup

Run setup from the frontend directory:

```bash
cd frontend
bun run setup
```

The setup script:

- runs `uv sync --extra server --extra modal --extra llm`
- creates or updates the repo-root `.env`
- configures `OPENROUTER_API_KEY`
- configures both `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN`
- sets `RL_INTERN_DEFAULT_MODEL` if it is missing
- runs `modal setup` when Modal is not authenticated
- deploys `rl_intern/modal_jobs/generic.py`
- deploys `rl_intern/modal_jobs/sb3.py`

Existing `.env` values are preserved. To re-enter setup values:

```bash
bun run setup -- --force
```

## Daily Development

Start the full local app:

```bash
cd frontend
bun run dev
```

Useful frontend commands:

```bash
cd frontend
bun run dev:frontend
bun run build
bun run lint
bun run preview
```

Useful Python commands:

```bash
uv sync --extra server --extra modal --extra llm
uv run pytest
uv run rl-intern-server --host 127.0.0.1 --port 8765
```

## Configuration

Runtime configuration is loaded from `.env` and `configs/main_agent_config.json`.

Common environment variables:

```text
OPENROUTER_API_KEY=...
HF_TOKEN=...
HUGGINGFACE_HUB_TOKEN=...
RL_INTERN_DEFAULT_MODEL=openrouter/anthropic/claude-sonnet-4.5
RL_INTERN_SERVER_PORT=8765
VITE_RL_INTERN_API_URL=http://127.0.0.1:8765
```

If `RL_INTERN_DEFAULT_MODEL` is set, it overrides the model in `configs/main_agent_config.json`.

## Architecture

```text
frontend/                    React + Vite + MUI chat UI
frontend/scripts/            Bun setup and dev orchestration
rl_intern/server/            FastAPI run server and WebSocket bridge
rl_intern/rpc.py             newline-delimited JSON runtime bridge
agent/core/                  agent loop, sessions, tool routing
agent/tools/                 built-in RL, Modal, file, and research tools
rl_intern/orchestrator/      typed experiment plans and domain adapters
rl_intern/domains/gym_sb3/   Gymnasium + Stable-Baselines3 adapter
rl_intern/domains/llm_trl/   TRL adapter for SFT, DPO, and GRPO-style jobs
rl_intern/runners/           local and Modal execution backends
rl_intern/modal_jobs/        deployed Modal apps
```

The browser opens `/api/ws/chat` with a session id. The server spawns `python -m rl_intern.rpc`, forwards browser messages to the agent runtime, and streams normalized events back to the UI.

## Experiment Domains

### `gym_sb3`

The Gym/SB3 adapter supports:

- environment inspection
- algorithm compatibility checks
- smoke tests
- random baselines
- SB3 training
- policy evaluation
- rollout videos
- Markdown reports

Common artifacts include `model.zip`, `config.json`, `eval.json`, `rollout.mp4`, `gym_*_summary.json`, and `report.md`.

### `llm_trl`

The LLM/TRL adapter supports:

- dataset inspection
- generated TRL training scripts
- local dry runs for lightweight validation
- Modal-backed training jobs for heavy runs
- SFT, DPO, and GRPO-style evaluation evidence
- base-vs-adapter sample and metric comparisons
- Markdown reports

Common artifacts include `train_trl.py`, `llm_training_config.json`, `llm_eval.json`, `improvement_evidence.json`, `evaluation_summary.json`, `llm_report.md`, and Modal output under `modal_artifacts/`.

## Run Artifacts

Each run writes local state under:

```text
artifacts/runs/<run_id>/
```

Common files:

```text
metadata.json
session.jsonl
artifact_manifest.json
report.md
llm_report.md
gym_*_summary.json
llm_*.json
modal_artifacts/
```

The artifact manifest groups outputs into buckets such as `checkpoints`, `adapters`, `configs`, `metrics`, `logs`, `videos`, `reports`, `samples`, and `errors`.

## Run Server API

The local server is intentionally bound to `127.0.0.1` or `localhost`.

```text
GET    /api/health
GET    /api/setup/status
POST   /api/session
WS     /api/ws/chat
POST   /runs
GET    /runs
GET    /runs/<run_id>
DELETE /runs/<run_id>
GET    /runs/<run_id>/events.jsonl
GET    /runs/<run_id>/artifacts
GET    /runs/<run_id>/report.md
GET    /runs/<run_id>/viewer
```

## Deprecated CLI

The prompt CLI still exists for debugging, but normal use should go through the web app:

```bash
uv run rl-intern "inspect CartPole-v1"
uv run rl-intern --runner modal "train CartPole-v1 with PPO"
```

## Testing

Run the Python test suite from the repo root:

```bash
uv run pytest
```

Run frontend checks from `frontend/`:

```bash
bun run lint
bun run build
```
