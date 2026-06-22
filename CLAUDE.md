# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project at a glance

PPO training framework (Stable-Baselines3) on PyBullet physics envs, with a Flask GUI for experiment management. Python ≥ 3.10. Two env families: bipedal walker (`walker_bullet`, single-agent Gymnasium) and 2-agent organism arena (`organism_arena_parallel`, PettingZoo). Configs live in `src/rl_framework/configs/experiments/*.yaml`.

## Environment setup

- A venv lives at `.venv/`. **Always activate it before running python or pytest** — `source .venv/bin/activate` — or the wrong interpreter / missing deps will silently break things.
- Install from `requirements-lock.txt` (CI does this, not `pip install -e .`). The lockfile must contain every direct dep from `pyproject.toml`; CI fails otherwise.

## CLI

Everything goes through one entry point:

```
python -m rl_framework.cli.main <subcommand> [--config-name <name>] [...]
```

Subcommands: `train`, `eval`, `arena-eval`, `arena-tournament`, `sweep`, `multi-seed`, `render-replay`, `morph-search`, `gui`. `--config-name` is required for all except `gui`. Default `--config-dir` is `src/rl_framework/configs/experiments`. Outputs land in `outputs/<experiment_name>/seed_<seed>/{checkpoints,logs,videos}/`.

- `arena-eval`: head-to-head of `--policy` vs `--opponent` (either may be `random`); reports win/draw/timeout rates.
- `arena-tournament`: round-robin over `--checkpoints` (comma-separated files/dirs; dirs contribute their `*.zip`) plus optional `--include-random`. Reports a Bradley-Terry Elo ranking; `--output` writes JSON, `--markdown-out` writes a report table.

## Tests, lint, repo policy

```
pytest -q --cov=src/rl_framework --cov-fail-under=60   # CI target
ruff check src tests scripts
python scripts/check_repo_policy.py                    # custom — enforces no tracked __pycache__/.venv/.egg-info and lockfile completeness
```

`mypy src --ignore-missing-imports` runs in CI but is non-blocking.

## Walker env quirks

- Recent overhaul: env uses **PD position control** by default. The `sim.control` block in YAML is **nested** (`mode`, `position_gain`, `velocity_gain`); the wizard previously flattened it to `control: null`, which the env tolerates by falling back to defaults — but a hand-written YAML should keep the nested form.
- Physics is **240 Hz** with **frame_skip=4** → 60 Hz control. `timestep`, `frame_skip`, `settle_steps` are all in the `sim` block.
- Termination is **contact-based** (torso touches ground) plus `min_height` and `max_height` fallbacks. **Tilt does not end episodes.** `max_height: 1.5 m` is a deliberate backstop against PD-jackhammer self-launching exploits — don't loosen it without thinking.
- Obs shape is **35**: `pos(3)+quat(4)+lin_vel(3)+ang_vel(3)+joint_pos(10)+joint_vel(10)+mass_scale(1)+friction_scale(1)`. The last two surface domain-randomization to the policy.
- Action shape is **10**: `[rHip, rKnee, rAnkle, lHip, lKnee, lAnkle, rShoulder, rElbow, lShoulder, lElbow]`. `action[i]=0` maps to the joint's rest pose (NOT joint midpoint).
- Robot is Atlas-DRC-class: 28 kg torso, 65.9 kg total, hip/knee 190/220 N·m, ankle 100, shoulder 90, elbow 100 N·m.

## Training throughput

- This machine has 24 physical cores / 32 threads. `num_envs: 24` saturates the cores; over-subscription beyond that doesn't help.
- For SubprocVecEnv launches, prefix with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` so each subprocess doesn't fight others for threads.
- PyBullet is CPU-only. Use `device: cpu` for the small MLP policies — GPU is slower for sub-100-dim obs/action spaces because of data-transfer overhead.
- The `Monitor` wrapper is already wired in `sb3_runner.py`; `rollout/ep_rew_mean` and `rollout/ep_len_mean` appear in TensorBoard at `outputs/<exp>/seed_<seed>/logs/PPO_*/`.

## GUI

- Launch: `python -m rl_framework.cli.main gui --port 5001` (user prefers 5001 over the default 5000).
- Auto-reload is on (`use_reloader=True` in `run_gui`), so Python edits restart the server — but any in-process training is killed on restart. Stop runs first if you want to preserve them.
- The wizard schema lives in `src/rl_framework/gui/app.py::get_schema`. Nested groups (like `sim.control`) need the recursive `populateGroup` JS helper in `gui/static/app.js`; flat dicts under top-level sections render as inputs.

## Arena config & training paths

Full arena config reference (obs/action layout, `sim`/`morphology`/`battle_rules` keys and defaults, self-play/annealing/curriculum): `docs/organism_arena_config.md`.


Two distinct vec-env paths in `sb3_runner.py`, chosen by whether self-play is on:
- **Self-play** (`self_play.enabled: true`): `SelfPlayEnvWrapper` exposes one live agent, wrapped by `SingleAgentArenaEnv` (a `gymnasium.Env`) onto SB3's native `DummyVecEnv`/`SubprocVecEnv`. **`num_envs > 1` is supported and parallelizes across cores** — `env_method` (reward annealing, curriculum) propagates to subprocess workers, and `Monitor` gives `rollout/ep_rew_mean`. Each worker is seeded per rank so spawns/opponents differ. This is the preferred arena path.
- **Shared-policy** (`self_play.enabled: false`): needs SuperSuit's 2-agent vec conversion + `_ArenaVecEnvAdapter` (seed()/uint8-dones patches), which only works single-process. A guard rejects `num_envs > 1` here.

## Subprocess gotcha

`SubprocVecEnv` can't be smoke-tested from a stdin heredoc — multiprocessing tries to re-run the parent script and fails on `<stdin>`. For smoke tests of the parallel path, use `num_envs: 1` (DummyVecEnv) or write to a temp `.py` file and invoke it (`PYTHONPATH=src python /tmp/foo.py`).

## Workflow

- **Direct commits to master** — no PRs, no feature branches.
- Commit in **logical groups within a session**, not a single end-of-session megacommit and not a commit per line.
- **Push to origin/master automatically after tests pass.** No need to ask first.
- **Small change** (single-file edit, tweak): `pytest` is sufficient before committing.
- **Big change** (env behavior, training pipeline, reward/termination): `pytest` + a short functional smoke test (env reset/step, a few hundred timesteps of training, or an eval/replay of an existing checkpoint) before committing.
- Skip hooks (`--no-verify`) is never warranted unless explicitly asked.
