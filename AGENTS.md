# AGENTS.md

This file provides guidance to coding agents working in this repository. It is
the single source of truth; `CLAUDE.md` imports it.

## Project at a glance

PPO training framework (Stable-Baselines3) on PyBullet physics envs, with a Flask GUI for experiment management. Python ≥ 3.11. Two env families: bipedal walker (`walker_bullet`, single-agent Gymnasium) and 2-agent organism arena (`organism_arena_parallel`, PettingZoo). Configs live in `src/rl_framework/configs/experiments/*.yaml`.

## Environment setup

- A venv lives at `.venv/`. **Always activate it before running python or pytest** — `source .venv/bin/activate` — or the wrong interpreter / missing deps will silently break things.
- Install in two steps, in this order (CI does the same):
  ```
  pip install -r requirements-lock.txt
  pip install --no-deps -e .
  ```
  The lockfile is the only source of dependency versions; `--no-deps` stops the editable install from re-resolving them. The package is imported as `rl_framework` from anywhere once installed; there is no root-level shim. The lockfile must contain every direct dep from `pyproject.toml`; CI fails otherwise. See [`docs/decisions/004-editable-install-src-layout.md`](docs/decisions/004-editable-install-src-layout.md).

## CLI

Everything goes through one entry point:

```
python -m rl_framework.cli.main <subcommand> [--config-name <name>] [...]
```

Subcommands: `train`, `eval`, `arena-eval`, `arena-tournament`, `sweep`, `multi-seed`, `render-replay`, `morph-search`, `gui`, `registry`, `quality-study`. `--config-name` is required for all except `gui`, `registry`, and `quality-study`. Default `--config-dir` is `src/rl_framework/configs/experiments`. Outputs land in `outputs/<experiment_name>/seed_<seed>/{checkpoints,logs,videos}/`.

- `registry`: `--registry-action inspect` (default; summary counts), `export` (full dump; capture with `--json`/`--json-out`), or `prune` (needs `--status`, `--older-than-days`, `--missing-only`, or explicit `--all`; `--dry-run` previews). Reads `<--base-dir>/run_registry.sqlite3` (default `outputs`).
- `quality-study`: runs the priority-3 study chosen by `--study` (default `all`) over `--seeds` (default 0,1,2); state and report go to `--study-output-dir` (default `outputs/quality_studies`); supports `--dry-run` and `--resume-incomplete`.
- `arena-eval`: head-to-head of `--policy` vs `--opponent` (either may be `random`); reports win/draw/timeout rates.
- `arena-tournament`: round-robin over `--checkpoints` (comma-separated files/dirs; dirs contribute their `*.zip`) plus optional `--include-random`. Reports a Bradley-Terry Elo ranking; `--output` writes JSON, `--markdown-out` writes a report table.

## Tests, lint, repo policy

```
pytest -q --cov=src/rl_framework --cov-fail-under=60   # CI target
ruff check src tests scripts
python scripts/check_repo_policy.py                    # custom — enforces no tracked __pycache__/.venv/.egg-info and lockfile completeness
```

`mypy src --ignore-missing-imports` runs in CI and is blocking.

## Walker env quirks

- Recent overhaul: env uses **PD position control** by default. The `sim.control` block in YAML is **nested** (`mode`, `position_gain`, `velocity_gain`). A flattened `control: null` is now an error (it used to fall back to defaults silently). Unknown `reward`/`termination` keys and an invalid `velocity_mode` also fail at construction.
- Physics is **240 Hz** with **frame_skip=4** → 60 Hz control. `timestep`, `frame_skip`, `settle_steps` are all in the `sim` block.
- Termination is **contact-based** (torso touches ground) plus `min_height` and `max_height` fallbacks. **Tilt does not end episodes.** `max_height: 1.5 m` is a deliberate backstop against PD-jackhammer self-launching exploits — don't loosen it without thinking.
- **Terrain geometry (rebuilt 2026-11-14):** `uneven` is a smooth heightfield in [0, `height`] with `wavelength` (default 1 m); `obstacles` tile the track every `obstacle_spacing` ± `obstacle_spacing_jitter` (1.2 ± 0.3 m) with optional `obstacle_height_jitter`; both span x = 0.6 to 0.6 + `length` (25 m) and are redrawn every reset from a dedicated RNG unless `randomize: false`. Terrain bodies are **pooled** (PyBullet cannot free shapes) — don't create bodies per reset. Pushes act at the torso COM for `push_recovery.duration_s` (0.1 s). `gait.clearance_point: sole` measures clearance at the sole (default `com` is the legacy metric). `sim.limb_collision: true` gives thighs/shins collision.
- **Terrain is per-episode when `terrain.mix` is set.** Each entry is a full terrain spec (`preset` plus its params, e.g. `height`, `obstacle_height`, `push_recovery`) with an optional relative `weight`. One is drawn per `reset()` from the env's seeded RNG, so a seeded run replays the same terrain sequence, and rollout workers diverge so a batch mixes terrains. Without `mix`, behaviour is unchanged. Sampling rebuilds **only** the static terrain boxes — `_build_world()` still runs once per env, preserving the reset optimisation (+0.23 ms/reset, ~0.04% of an episode). The episode's terrain is reported in `info["terrain_preset"]`; a mixed run's aggregate metrics otherwise hide which terrain caused a fall. An explicit but empty `mix: []` is an error, not "no mix". Curriculum terrain updates take effect at the next reset. Unknown live parameter keys raise an error.
- Coordinate-free v2 obs shape is **35**: `z(1)+quat(4)+lin_vel(3)+ang_vel(3)+joint_pos(10)+joint_vel(10)+mass_scale(1)+friction_scale(1)+foot_contacts(2)`. Contacts expose the contact state used by contact-driven rewards. Set `observation.include_previous_action: true` to append the 10-D normalized action actually applied to PD control (shape **45**); this is checkpoint-incompatible.
- Action shape is **10**: `[rHip, rKnee, rAnkle, lHip, lKnee, lAnkle, rShoulder, rElbow, lShoulder, lElbow]`. `action[i]=0` maps to the joint's rest pose (NOT joint midpoint).
- Robot is Atlas-DRC-class: 28 kg torso, 65.9 kg total, hip/knee 190/220 N·m, ankle 100, shoulder 90, elbow 100 N·m.

## Walker env layout

`physics/locomotion/walker_bullet.py` only orchestrates `reset`/`step`; the rest is `config.py` (typed config; the live-parameter set is *derived* from `live_field` declarations — add a tunable field there, not in a hand-written set), `terrain.py`, `robot.py`, `sensing.py` (per-step `FootSnapshot`), `perturbations.py`, and `observation.py` (`ObservationLayout` named slices + `STEP_INFO_KEYS`). Analysis tooling (`quality_study`, `walker_diagnostics`) lives in `physics/locomotion/analysis/`. Live-parameter plumbing is shared in `core/config/live_params.py`. See ADR-003.

## Training throughput

- This machine has 24 physical cores / 32 threads. `num_envs: 24` saturates the cores; over-subscription beyond that doesn't help.
- For SubprocVecEnv launches, prefix with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` so each subprocess doesn't fight others for threads.
- PyBullet is CPU-only. Use `device: cpu` for the small MLP policies — GPU is slower for sub-100-dim obs/action spaces because of data-transfer overhead.
- The `Monitor` wrapper is already wired in `core/training/sb3_runner.py`; `rollout/ep_rew_mean` and `rollout/ep_len_mean` appear in TensorBoard at `outputs/<exp>/seed_<seed>/logs/PPO_*/`.

## GUI

- Launch: `python -m rl_framework.cli.main gui --port 5001` (the CLI default is 5001).
- Auto-reload is on (`use_reloader=True` in `run_gui`), so Python edits restart the server — but any in-process training is killed on restart. Stop runs first if you want to preserve them.
- The wizard schema lives in `src/rl_framework/gui/app.py::get_schema`. Nested groups (like `sim.control`) need the recursive `populateGroup` JS helper in `gui/static/app.js`; flat dicts under top-level sections render as inputs.

## Arena config & training paths

Full arena config reference (obs/action layout, `sim`/`morphology`/`battle_rules` keys and defaults, self-play/annealing/curriculum): `docs/organism_arena_config.md`.

Two distinct vec-env paths in `games/arena/training.py`, chosen by whether self-play is on:

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

## Repository layout

| Path | Role |
|---|---|
| `src/rl_framework/` | The package. `core/`, `physics/`, `games/`, `cli/`, `gui/`, `configs/`, `plugins.py` |
| `src/rl_framework/configs/experiments/` | Experiment YAMLs — the default `--config-dir` |
| `tests/` | pytest suite; CI gates coverage at 60% |
| `scripts/` | Repo tooling, including `check_repo_policy.py`; `benchmark_device_matrix.py` is a shim over the package module |
| `docs/` | Roadmap, studies, config references, the walker decision ledger |
| `reviews/` | Point-in-time review artifacts |
| `outputs/` | Run outputs — **gitignored, disposable, rebuildable** |

## Important invariants

- **Activate `.venv/` first.** Without it the wrong interpreter silently
  produces wrong results rather than an error.
- **Install the lockfile first, then `pip install --no-deps -e .`.** The
  lockfile is the only source of dependency versions, and it must contain every
  direct dep from `pyproject.toml`, or CI fails. (This replaced the old
  "lockfile, not `pip install -e .`" rule; see
  `docs/decisions/004-editable-install-src-layout.md`.)
- `max_height: 1.5 m` is a deliberate backstop against the PD-jackhammer
  self-launch exploit. Loosening it re-opens a known degenerate solution.
- `sim.control` is **nested**. A flattened `control: null` is an error; it used
  to fall back to defaults, which looked like a working run and was not one.
- An explicit but empty `terrain.mix: []` is an error, not "no mix".
- Changing observation shape (35 → 45 with `include_previous_action`) is
  **checkpoint-incompatible**. Old checkpoints cannot be loaded.
- Unknown live parameter keys are errors. Terrain updates take effect at the next reset.
- Results go in [`docs/walker_learning_history.md`](docs/walker_learning_history.md),
  **including refuted hypotheses**. A study that only records successes is not
  a decision ledger.

## Validation requirements

Beyond the gate below, a change to env behaviour, the training pipeline, or
reward/termination needs a functional smoke test: env reset/step, a few hundred
timesteps of training, or an eval/replay of an existing checkpoint. A learning
claim needs **multi-seed** evidence — single-seed results have repeatedly proven
misleading here, which is why `multi-seed` is a first-class subcommand.

Record the outcome in `docs/walker_learning_history.md` whether or not the
hypothesis held.

## Cross-repository contracts

None identified. This repository is self-contained; its only external contracts
are library versions pinned in `requirements-lock.txt`.

## Security / safety constraints

- No credentials or personal data belong in this repository.
- `outputs/` is gitignored and disposable — never put the only copy of anything
  there.
- Training saturates 24 cores. Prefix `SubprocVecEnv` launches with
  `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` or workers will fight for threads.

## Definition of done

1. `pytest -q --cov=src/rl_framework --cov-fail-under=60` passes.
2. `ruff check src tests scripts` is clean.
3. `python scripts/check_repo_policy.py` passes.
4. Big changes additionally have a functional smoke test (above).
5. Experiment results — including negative ones — recorded in
   `docs/walker_learning_history.md`.
6. `docs/STATUS.md` updated if the current state changed.

Note: the Workflow section above authorises direct commits and automatic pushes
to `master` after tests pass. That is a deliberate repo-local policy and it
overrides the more conservative default in `ai-hub/AGENTS.md` §9 — per §5,
project-local instructions win.

## Where to find deeper context

| Topic | Document |
|---|---|
| Current state | [`docs/STATUS.md`](docs/STATUS.md) |
| Validation evidence | [`docs/VALIDATION.md`](docs/VALIDATION.md) |
| Decision records | [`docs/decisions/`](docs/decisions/) |
| Active roadmap | [`docs/open_items_todo.md`](docs/open_items_todo.md) |
| Walker decision ledger (canonical) | [`docs/walker_learning_history.md`](docs/walker_learning_history.md) |
| Arena config reference | [`docs/organism_arena_config.md`](docs/organism_arena_config.md) |
| Learning quality studies | [`docs/learning_quality_studies.md`](docs/learning_quality_studies.md) |
| Training presets | [`docs/training_presets.md`](docs/training_presets.md) |
| Visualization | [`VISUALIZATION_GUIDE.md`](VISUALIZATION_GUIDE.md) |
