# RL Intern

The agent that trains agents.

`rl-intern` is a local web app for launching reinforcement learning runs. The
frontend talks to a local FastAPI run server, which owns agent execution,
approvals, Modal jobs, artifacts, reports, and session logs.

## Quick Start

```bash
git clone <repo>
cd rl-intern/frontend
bun install
bun run setup
bun run dev
```

Then open http://localhost:5173.

Daily development is just:

```bash
cd frontend
bun run dev
```

`bun run dev` starts the local run server on `127.0.0.1:8765` when it is not
already running, then starts the Vite frontend.

## Setup

Run setup once per machine:

```bash
cd frontend
bun run setup
```

The setup script:

- syncs Python dependencies with `uv`
- writes missing secrets to the repo-root `.env`
- configures `OPENROUTER_API_KEY`
- configures `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN`
- authenticates Modal when needed
- deploys the Modal backend apps

Existing `.env` values are preserved. To re-enter values, run:

```bash
bun run setup -- --force
```

## What It Does

- Inspects Gymnasium environments
- Runs smoke tests and random baselines
- Chooses compatible RL algorithms
- Trains Stable-Baselines3 agents
- Evaluates policies across episodes
- Records rollout videos
- Generates Markdown experiment reports
- Launches trusted jobs on Modal

## Internal Commands

Most users should use the frontend scripts above. These commands are kept for
debugging and tests:

```bash
uv run rl-intern-server --host 127.0.0.1 --port 8765
uv run modal deploy rl_intern/modal_jobs/generic.py
uv run modal deploy rl_intern/modal_jobs/sb3.py
```

The old prompt CLI is deprecated for normal use:

```bash
uv run rl-intern "inspect CartPole-v1"
```

## Run Artifacts

Each run writes local artifacts under:

```text
artifacts/runs/<run_id>/
```

Common outputs include:

```text
model.zip
config.json
eval.json
rollout.mp4
report.md
session.jsonl
```

The run server also exposes:

```text
GET  /api/health
GET  /api/setup/status
POST /api/session
GET  /runs
GET  /runs/<run_id>
GET  /runs/<run_id>/events.jsonl
GET  /runs/<run_id>/artifacts
GET  /runs/<run_id>/report.md
GET  /runs/<run_id>/viewer
```
