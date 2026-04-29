# RL Intern

The agent that trains agents.

`rl-intern` is an autonomous reinforcement learning engineer in your terminal. It inspects Gymnasium environments, runs random baselines, chooses compatible algorithms, trains Stable-Baselines3 agents, evaluates policies, records rollout videos, and generates reproducible experiment reports.

## Install

```bash
git clone <repo>
cd rl-intern
uv sync
uv tool install -e .
```

## Usage

```bash
rl-intern
rl-intern "train PPO on CartPole-v1 and give me a report"
rl-intern "compare DQN and PPO on LunarLander-v3"
rl-intern "inspect this custom Gymnasium env and tell me if it is trainable"
```

Use OpenRouter through LiteLLM:

```bash
export OPENROUTER_API_KEY="..."
rl-intern --model openrouter/openai/gpt-oss-120b:free "inspect CartPole-v1"
```

## Optional Modal Runner

Local execution is the default. Modal is an optional remote runner for trusted
Gymnasium/Stable-Baselines3 jobs.

```bash
uv sync --extra modal
modal setup
uv run modal deploy rl_intern/modal_jobs/sb3.py
uv run rl-intern --runner modal --model openrouter/openai/gpt-oss-120b:free \
  "train PPO on CartPole-v1 for 10000 timesteps, evaluate it for 20 episodes, record a rollout, and generate a report"
```

Modal is not used for arbitrary custom code in this version. Generated
environments, reward functions, and paper reproduction scripts should still run
locally until sandbox support is added.

## Web Frontend

A web frontend lives in `frontend/` (Vite + React + MUI, mirroring
[`huggingface/ml-intern`](https://github.com/huggingface/ml-intern)). It talks
to the run server over a single WebSocket that bridges to the existing
`rl_intern.rpc` JSON-line protocol — Python remains the only place RL tools,
approvals, training, and artifacts run.

```bash
# 1. start the run server (FastAPI on 127.0.0.1:8765)
uv sync --extra server
rl-intern-server --host 127.0.0.1 --port 8765

# 2. start the dev frontend in another shell
cd frontend
bun install
bun run dev
```

Then open http://localhost:5173.

For a single command that starts the run server when it is not already running,
use `bun run dev:full` from `frontend/`.

## Run Server and Viewer

Every run writes a canonical session log:

```text
artifacts/runs/<run_id>/session.jsonl
```

Run the local-only server:

```bash
uv sync --extra server
rl-intern-server --host 127.0.0.1 --port 8765
```

Endpoints:

```text
GET  /runs
GET  /runs/<run_id>
GET  /runs/<run_id>/events.jsonl
GET  /runs/<run_id>/artifacts
GET  /runs/<run_id>/report.md
GET  /runs/<run_id>/viewer
```

The viewer page links the JSONL session log for Euphony. Euphony is optional:
open https://openai.github.io/euphony/ and load
`artifacts/runs/<run_id>/session.jsonl`, or use the local
`/runs/<run_id>/events.jsonl` URL when your browser allows localhost loading.

## What It Does

- Inspects Gymnasium environments
- Runs smoke tests and random baselines
- Chooses compatible RL algorithms
- Trains Stable-Baselines3 agents
- Evaluates policies across episodes
- Records rollout videos
- Generates Markdown experiment reports
- Optionally launches trusted SB3 jobs on Modal

## v0.1 Supported Algorithms

- PPO
- DQN

## v0.1 Supported Environment API

- Gymnasium

## First Demo

```bash
rl-intern "train PPO on CartPole-v1 for 10000 timesteps, evaluate it for 20 episodes, record a rollout, and generate a report"
```

Expected artifacts:

```text
artifacts/runs/<run_id>/model.zip
artifacts/runs/<run_id>/config.json
artifacts/runs/<run_id>/eval.json
artifacts/runs/<run_id>/rollout.mp4
artifacts/runs/<run_id>/report.md
artifacts/runs/<run_id>/session.jsonl
```

## Guardrails

- Do not claim success without evaluation.
- Do not hide failed training runs.
- Do not assume reward curves are stable from one seed.
- Do not require cloud credentials for core functionality.
- Do not add web UI or distributed training in v0.1.

## Roadmap

- CleanRL script generation
- W&B sweeps and training curve plots
- PettingZoo multi-agent support
- Minari offline RL support
- RLlib distributed training
- TRL post-training workflows
- Hugging Face Hub publishing
