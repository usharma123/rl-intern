# RL Intern

The agent that trains agents.

`rl-intern` is a local web app for running reinforcement learning experiments from a chat interface. The React frontend talks to a local FastAPI run server, which launches the agent runtime, streams tool events over WebSocket, manages approvals, runs local or Modal-backed jobs, and stores reports and artifacts for each run.

## Current capabilities

- Chat-driven experiment planning and execution
- Local Gymnasium + Stable-Baselines3 workflows
- LLM fine-tuning workflows built around TRL
- Local and Modal runners
- Run monitor for plans, jobs, artifacts, and evidence
- Environment inspection, smoke tests, algorithm selection, training, evaluation, rollout recording, and Markdown reports
- Persistent run history under `artifacts/runs/<run_id>/`

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Bun](https://bun.sh/) 1.3+ (repo pins `packageManager` in `frontend/package.json`)
- Modal account and CLI authentication for Modal jobs (`bun run setup` runs `modal setup` when needed)
- OpenRouter API key for the default agent model
- Hugging Face token when you pull private models or datasets

## Quick start

```bash
git clone https://github.com/usharma123/rl-intern.git
cd rl-intern/frontend
bun install
bun run setup
bun run dev
```

Then open http://localhost:5173 (or http://127.0.0.1:5173 — the dev script binds Vite to `127.0.0.1`).

`bun run dev` starts `rl-intern-server` on `127.0.0.1:8765` (or `RL_INTERN_SERVER_PORT`) if health checks fail, then starts the Vite frontend.

## One-time setup

Run setup from the `frontend/` directory:

```bash
cd frontend
bun run setup
```

The setup script:

- runs `uv sync --extra server --extra modal --extra llm`
- creates or updates the repo-root `.env`
- configures `OPENROUTER_API_KEY`
- configures both `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN` from one prompt
- sets `RL_INTERN_DEFAULT_MODEL` if it is missing (default: `openrouter/anthropic/claude-sonnet-4.5`)
- runs `modal setup` when Modal is not authenticated
- deploys `rl_intern/modal_jobs/generic.py`
- deploys `rl_intern/modal_jobs/sb3.py`

Existing `.env` values are preserved. To re-enter setup values:

```bash
bun run setup -- --force
```

**Tests:** setup does **not** install the `dev` extra (pytest lives there). Sync once before running Python tests:

```bash
uv sync --extra server --extra modal --extra llm --extra dev
```

## Daily development

Start the full local app:

```bash
cd frontend
bun run dev
```

Useful frontend commands:

```bash
cd frontend
bun run dev:frontend   # Vite only (expects API already reachable at VITE_RL_INTERN_API_URL)
bun run dev:full       # alias of dev.ts (same as bun run dev)
bun run build
bun run lint
bun run preview
```

Useful Python commands:

```bash
uv sync --extra server --extra modal --extra llm [--extra dev]
uv run rl-intern-server --host 127.0.0.1 --port 8765
uv run pytest   # requires --extra dev sync
```

## Configuration

Runtime configuration is loaded from `.env` at the repo root and `configs/main_agent_config.json`.

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
frontend/                    React 19 + Vite + MUI chat UI
frontend/scripts/            Bun setup and dev orchestration
rl_intern/server/            FastAPI run server and WebSocket bridge
rl_intern/rpc.py             newline-delimited JSON runtime bridge
rl_intern/viewer/            bundled HTML viewer for run logs
agent/core/                  agent loop, sessions, tool routing
agent/tools/                 built-in RL, Modal, file, and research tools
rl_intern/orchestrator/      typed experiment plans and domain adapters
rl_intern/domains/gym_sb3/   Gymnasium + Stable-Baselines3 adapter
rl_intern/domains/llm_trl/   TRL adapter for SFT, DPO, and GRPO-style jobs
rl_intern/runners/           local and Modal execution backends
rl_intern/modal_jobs/        deployed Modal apps
```

Typical browser flow:

1. `POST /api/session` allocates a run id used as `session_id`.
2. The client opens `/api/ws/chat?session_id=...` (optional `model`, `runner` query params).

The server spawns `python -m rl_intern.rpc`, forwards browser messages to the agent runtime, and streams normalized events back to the UI. Subprocess stderr is filtered and surfaced as synthetic `bridge_log` events.

## Experiment domains

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

## Run artifacts

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

## Run server API

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

The `uv run rl-intern` CLI still exists for debugging, but normal use should go through the web app:

```bash
uv run rl-intern "inspect CartPole-v1"
uv run rl-intern --runner modal "train CartPole-v1 with PPO"
```

## Testing

From the repo root, after `uv sync` **with** `--extra dev`:

```bash
uv run pytest
```

Frontend checks from `frontend/`:

```bash
bun run lint
bun run build
```
