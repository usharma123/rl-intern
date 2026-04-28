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

## What It Does

- Inspects Gymnasium environments
- Runs smoke tests and random baselines
- Chooses compatible RL algorithms
- Trains Stable-Baselines3 agents
- Evaluates policies across episodes
- Records rollout videos
- Generates Markdown experiment reports

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
artifacts/CartPole-v1/PPO/seed_0/model.zip
artifacts/CartPole-v1/PPO/seed_0/config.json
artifacts/CartPole-v1/PPO/seed_0/eval.json
artifacts/CartPole-v1/PPO/seed_0/rollout.mp4
artifacts/CartPole-v1/PPO/seed_0/report.md
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
