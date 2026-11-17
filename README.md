# RL Experiment Framework

Train and evaluate PPO agents on two [PyBullet](https://pybullet.org/) environments, driven entirely by YAML files:

- **Walker** (`walker_bullet`): a single-agent, Atlas-class bipedal robot learning to walk, built on [Gymnasium](https://gymnasium.farama.org/).
- **Organism arena** (`organism_arena_parallel`): an N-agent competitive arena, built on [PettingZoo](https://pettingzoo.farama.org/), with self-play, a frozen-opponent league, and head-to-head evaluation.

Training uses [Stable-Baselines3](https://stable-baselines3.readthedocs.io/). A command-line tool and a small web GUI cover training, sweeps, multi-seed runs, evaluation, tournaments, and replay rendering.

---

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command reference](#command-reference)
- [Configuration](#configuration)
- [Environments](#environments)
- [Outputs](#outputs)
- [Web GUI](#web-gui)
- [Shipped experiments](#shipped-experiments)
- [Development](#development)
- [Project layout](#project-layout)
- [Documentation](#documentation)

---

## Requirements

- Python 3.10 or newer. CI and the Docker image use **Python 3.11**, which is the recommended version.
- Linux is the tested platform. PyBullet builds from a wheel on common platforms.
- CPU training works out of the box. The walker and arena policies are small MLPs and train faster on CPU than on GPU. Set `training.device` to `cuda` only if you have a tested GPU setup.

---

## Installation

The dependencies are pinned in `requirements-lock.txt`. Install from the lockfile so your environment matches CI.

```bash
git clone https://github.com/sparktron/MLenvironment.git
cd MLenvironment

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e .
```

`pip install -e .` installs the `rl_framework` package itself. It does not change any pinned versions. These are the same steps CI runs.

Check the install:

```bash
python -m rl_framework.cli.main --help
```

Always activate `.venv` before running the CLI or `pytest`. Otherwise you can end up using a different Python with different packages.

### Docker

The Docker image runs the same CLI. Build it from the repository root:

```bash
docker build -t rl-framework .
```

Mount `outputs/` so results persist after the container exits:

```bash
docker run --rm -v "$(pwd)/outputs:/app/outputs" rl-framework \
  train --config-name walker_smoke_cpu
```

The image installs PyTorch and its CUDA libraries from the lockfile, so it is several gigabytes. `.dockerignore` keeps `.venv/`, `outputs/` and `.git/` out of the build context.

---

## Quick start

Run a short training job to confirm everything works (about a minute on CPU):

```bash
python -m rl_framework.cli.main train --config-name walker_smoke_cpu --total-timesteps 1024
```

The command prints the path of the saved model, for example `outputs/walker_smoke_cpu/seed_0/checkpoints/final_model`. Evaluate that checkpoint:

```bash
python -m rl_framework.cli.main eval \
  --config-name walker_smoke_cpu \
  --model-path outputs/walker_smoke_cpu/seed_0/checkpoints/final_model.zip
```

Then train the arena and compare two checkpoints head to head:

```bash
python -m rl_framework.cli.main train --config-name organisms_fight_arena --total-timesteps 1024

python -m rl_framework.cli.main arena-eval \
  --config-name organisms_fight_arena \
  --policy outputs/organisms_fight_arena/seed_11/checkpoints/final_model.zip \
  --opponent random \
  --n-episodes 20
```

For a real experiment, drop `--total-timesteps` to use the config's full budget, and view learning curves with TensorBoard:

```bash
python -m rl_framework.cli.main train --config-name robot_walk_basic
tensorboard --logdir outputs/
```

---

## Command reference

Every command goes through one entry point:

```bash
python -m rl_framework.cli.main <command> [options]
```

| Command | Purpose |
|---|---|
| `train` | Train one run from a config. |
| `eval` | Evaluate a walker checkpoint on its environment. |
| `arena-eval` | Head-to-head evaluation of two arena policies (or one vs. `random`). |
| `arena-tournament` | Round-robin over checkpoints, ranked by Bradley-Terry Elo. |
| `sweep` | Expand `sweep.parameters` into a grid of runs. |
| `multi-seed` | Run one config across several seeds in parallel and aggregate the results. |
| `render-replay` | Render a GIF of a policy (walker or arena matchup). |
| `morph-search` | Mutate arena morphology parameters and keep the best by tournament Elo. |
| `registry` | Inspect, export, or prune the run registry. |
| `quality-study` | Run the staged learning-quality study matrices. |
| `gui` | Start the web experiment manager. |

`--config-name` is required for every command except `gui`, `registry` and `quality-study`. Configs are read from `src/rl_framework/configs/experiments/` unless you pass `--config-dir`.

### Common options

| Option | Meaning |
|---|---|
| `--config-name NAME` | YAML file name without the `.yaml` extension. |
| `--config-dir DIR` | Directory containing experiment YAMLs. Default: `src/rl_framework/configs/experiments`. |
| `--seed N` | Override the run seed (sets both the global and environment seed). |
| `--device DEVICE` | Override `training.device`: `auto`, `cpu`, `cuda`, or `cuda:N`. |
| `--total-timesteps N` | Override `training.total_timesteps`. Useful for smoke tests. |
| `--json` | Print the result as one JSON line and suppress human-readable output. Good for scripts. |
| `--json-out PATH` | Write the JSON result to a file. |

### `train`

Trains one run and saves the final model.

```bash
python -m rl_framework.cli.main train --config-name robot_walk_basic
python -m rl_framework.cli.main train --config-name robot_walk_basic --resume outputs/robot_walk_basic/seed_42/checkpoints/final_model.zip
```

`--resume PATH` continues training from a saved `.zip` model. Checkpoints are tied to their observation shape, so resume only with a config that matches the one that produced the checkpoint.

### `eval`

Runs a walker checkpoint for `evaluation.episodes` episodes and reports mean return, episode length, and termination statistics. Requires `--model-path`.

```bash
python -m rl_framework.cli.main eval --config-name robot_walk_basic \
  --model-path outputs/robot_walk_basic/seed_42/checkpoints/final_model.zip
```

### `arena-eval`

Plays `--policy` against `--opponent` for `--n-episodes` episodes per spawn orientation. Roles are swapped by default so a spawn-position advantage does not inflate results. Reports win, draw and timeout rates.

| Option | Default | Meaning |
|---|---|---|
| `--policy PATH` | required | Checkpoint under test. |
| `--opponent PATH\|random` | `random` | Opponent checkpoint, or `random` for a random-action baseline. |
| `--n-episodes N` | `100` | Episodes per spawn orientation. |
| `--no-swap-roles` | off | Keep the fixed spawn positions (introduces positional bias). |
| `--policies A,B,...` | none | N-agent evaluation: one checkpoint per arena slot, in order. |
| `--output PATH` | none | Write the result as JSON. |

```bash
python -m rl_framework.cli.main arena-eval --config-name organisms_fight_arena \
  --policy outputs/organisms_fight_arena/seed_11/checkpoints/final_model.zip \
  --opponent random --n-episodes 50 --json-out results/eval.json
```

### `arena-tournament`

Plays every checkpoint against every other and ranks them by Bradley-Terry Elo.

| Option | Meaning |
|---|---|
| `--checkpoints A,B,DIR` | Comma-separated checkpoint files and/or directories. Each directory contributes its `*.zip` files. |
| `--include-random` | Add a random-action baseline. |
| `--n-episodes N` | Episodes per pairing. |
| `--output PATH` | Write the standings as JSON. |
| `--markdown-out PATH` | Write the standings as a Markdown table. |

```bash
python -m rl_framework.cli.main arena-tournament --config-name organisms_fight_arena \
  --checkpoints outputs/organisms_fight_arena/runs/ --include-random \
  --markdown-out reports/league.md
```

### `sweep`

Expands `sweep.parameters` into the Cartesian product of values and trains each combination. Run names and output directories are derived from the swept values.

```bash
python -m rl_framework.cli.main sweep --config-name robot_walk_basic --dry-run   # list the planned runs
python -m rl_framework.cli.main sweep --config-name robot_walk_basic             # run them
python -m rl_framework.cli.main sweep --config-name robot_walk_basic --resume-incomplete
```

`--resume-incomplete` continues an interrupted sweep from its saved state. Sweep runs are written under `outputs/<experiment>/runs/<run_id>/`.

### `multi-seed`

Runs the same config across several seeds in parallel and aggregates the results. A learning claim needs multiple seeds, so use this rather than a single run when comparing configs.

| Option | Default | Meaning |
|---|---|---|
| `--seeds 0,1,2,3,4` | `multi_seed.seeds` in the config, else `0`–`4` | Seeds to run. |
| `--max-workers N` | CPU count | Parallel worker processes. |
| `--resume PATH` | none | Resume each seed from a checkpoint path containing `{seed}`. |

```bash
python -m rl_framework.cli.main multi-seed --config-name robot_walk_basic --seeds 0,1,2,3,4 --max-workers 4
```

The aggregate is written to `outputs/<experiment>/multi_seed_summary/aggregate.csv`.

### `render-replay`

Renders a replay of a policy into the run's `videos/` directory. The walker writes an MP4 from one checkpoint. The arena writes a GIF: `agent_0` is `--model-path`, and `agent_1` is `--replay-opponent` (or the same policy if omitted).

```bash
python -m rl_framework.cli.main render-replay --config-name organisms_fight_arena \
  --model-path outputs/organisms_fight_arena/seed_11/checkpoints/final_model.zip \
  --replay-opponent random
```

### `morph-search`

Arena only. Trains `--trials` mutations of the arena morphology, scores them by tournament Elo, and reports the best parameters. It rejects `mean_return` scoring: the arena is zero-sum, so mean return ranks trials on noise rather than skill.

```bash
python -m rl_framework.cli.main morph-search --config-name organisms_fight_arena --trials 5
```

### `registry`

Inspects the SQLite run registry at `outputs/run_registry.sqlite3`.

| Option | Meaning |
|---|---|
| `--registry-action inspect\|export\|prune` | Summary counts (default), full JSON dump, or delete matching records. |
| `--base-dir DIR` | Directory holding the registry. Default: `outputs`. |
| `--prune-target analysis-jobs\|runs\|artifacts` | What `prune` deletes. Default: `analysis-jobs`. |
| `--status a,b` | Only records with these statuses (for example `failed,interrupted`). |
| `--older-than-days N` | Only records older than N days. |
| `--missing-only` | For `artifacts`: only remove index rows whose files are gone. |
| `--all` | Allow an unfiltered prune. |
| `--dry-run` | Show what would be pruned without deleting anything. |

```bash
python -m rl_framework.cli.main registry --registry-action inspect
python -m rl_framework.cli.main registry --registry-action prune --prune-target analysis-jobs \
  --status failed --older-than-days 30 --dry-run
```

`prune` never deletes artifact files from disk. `--prune-target artifacts` only removes index rows.

### `quality-study`

Runs the staged learning-quality study matrices and writes state, models and a report to `--study-output-dir` (default `outputs/quality_studies`).

| Option | Meaning |
|---|---|
| `--study NAME` | One of `walker`, `walker-velocity`, `walker-gait`, `walker-velocity-ramp`, `walker-swing-touchdown`, `walker-slip-recovery`, `walker-action-memory`, `arena`, `algorithms`, `all` (default). |
| `--seeds 0,1,2` | Seeds for the study. Default `0,1,2`. |
| `--study-step-budget N` | Override the per-run step budget. |
| `--study-wall-clock-seconds N` | Per-run wall-clock budget. Default `900`. |
| `--study-eval-episodes N` | Episodes per evaluation. Default `20`. |
| `--resume-incomplete` | Resume from saved state. |
| `--dry-run` | Plan the study without training. |

```bash
python -m rl_framework.cli.main quality-study --study walker-gait --dry-run
```

Study results and their interpretation are recorded in [`docs/learning_quality_studies.md`](docs/learning_quality_studies.md).

---

## Configuration

Each experiment is one YAML file in `src/rl_framework/configs/experiments/`. The loader and validator reject unknown keys and invalid values before anything starts, so typos fail loudly rather than falling back to defaults.

### Top-level sections

| Section | Required | Purpose |
|---|---|---|
| `experiment_name` | yes | Names the output directory. |
| `seed` | yes | Global seed. |
| `environment` | yes | Environment type, physics, rewards, and rules. |
| `training` | yes | PPO hyperparameters and device. |
| `evaluation` | no | Episodes per evaluation, optional best-model tracking. |
| `output` | no | `base_dir` (default `outputs`). |
| `sweep` | no | `parameters`: a mapping of dotted keys to lists of values. |
| `self_play`, `reward_annealing`, `curriculum` | arena only | Arena training features. |
| `morphology_search` | arena only | Settings for `morph-search`. |

Dotted keys address nested values, for example `environment.reward.target_velocity`.

### Walker example

This is `robot_walk_basic.yaml` (validated). Physics and control live under `environment.sim`, and the PD control block is nested.

```yaml
experiment_name: robot_walk_basic
seed: 42

environment:
  type: walker_bullet
  seed: 42
  sim:
    gravity: -9.81
    mass: 28.0
    friction: 0.9
    max_force: 35.0
    timestep: 0.004166667
    frame_skip: 4
    settle_steps: 30
    control:
      mode: pd
      position_gain: 0.1
      velocity_gain: 1.0
  reward:
    alive_bonus: 0.25
    forward_velocity_weight: 2.0
    target_velocity: 1.0
    velocity_sigma: 0.5
    orientation_penalty_weight: 0.3
    torque_penalty_weight: 0.01
    fall_penalty: 10.0
  terrain:
    preset: flat
  termination:
    min_height: 0.18
    max_height: 1.5
    max_steps: 800
  reset_randomization:
    position_xy_noise: 0.02
    yaw_noise: 0.08
  domain_randomization:
    mass_scale_range: [0.95, 1.05]
    friction_range: [0.9, 1.1]

training:
  policy: MlpPolicy
  total_timesteps: 2000000
  num_envs: 24
  n_steps: 2048
  batch_size: 512
  n_epochs: 10
  learning_rate: 0.0003
  learning_rate_end: 0.00005
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.005
  vf_coef: 0.5
  max_grad_norm: 0.5
  checkpoint_every: 50000
  normalize_observations: true
  device: cpu

evaluation:
  episodes: 5

output:
  base_dir: outputs

sweep:
  parameters:
    environment.reward.target_velocity: [0.5, 1.0, 1.5]
    environment.reward.torque_penalty_weight: [0.005, 0.01]
```

### Arena example

This is `organisms_fight_arena.yaml` (validated). Arena rules are split across `sim`, `morphology`, `battle_rules`, `resources`, and `reward`, and the training features are top-level sections.

```yaml
experiment_name: organisms_fight_arena
seed: 11

environment:
  type: organism_arena_parallel
  seed: 11
  sim:
    arena_half_extent: 1.2
    move_speed: 0.05
    collision_radius: 0.08
    speed_size_exponent: 1.0
    spawn_jitter: 0.1
  morphology:
    base_size: 1.0
    episode_growth_scale: 0.0
    health: 1.2
  battle_rules:
    damage: 0.06
    collision_damage: 0.0
    attack_range: 0.2
    cooldown_steps: 3
    max_steps: 400
    win_health_threshold: 0.0
  resources:
    initial_energy: 1.0
    max_energy: 1.0
    movement_cost: 0.01
    attack_cost: 0.04
    food_count: 2
    food_energy: 0.35
    food_radius: 0.10
    food_respawn_steps: 40
    food_placement: uniform
  reward:
    approach_weight: 0.0

training:
  policy: MlpPolicy
  total_timesteps: 30000
  learning_rate: 0.0003
  n_steps: 1024
  batch_size: 256
  checkpoint_every: 5000
  normalize_observations: true
  device: cpu
  num_envs: 8

evaluation:
  episodes: 5

self_play:
  enabled: true
  snapshot_freq: 5000
  max_league_size: 10
  sampling_mode: uniform

reward_annealing:
  enabled: true
  anneal_steps: 15000
  min_eliminations: 10

curriculum:
  enabled: true
  metric: arena/agent_0_win_rate
  warmup_steps: 5000
  max_level: 3
  level_up_thresholds:
    0: 0.60
    1: 0.65
    2: 0.70
  level_params:
    1:
      battle_rules.cooldown_steps: 3
      battle_rules.damage: 0.08
    2:
      battle_rules.cooldown_steps: 2
      battle_rules.damage: 0.10
    3:
      battle_rules.cooldown_steps: 2
      battle_rules.damage: 0.12

output:
  base_dir: outputs

sweep:
  parameters:
    environment.battle_rules.attack_range: [0.15, 0.2, 0.25]
    environment.battle_rules.cooldown_steps: [2, 3, 4]
```

### Frequently changed walker keys

| Key | Meaning |
|---|---|
| `environment.observation.version` | `v1` (35-D) or `v2` (37-D, adds foot-contact bits). |
| `environment.observation.coordinate_free` | Drops global x/y. Requires `v2`. |
| `environment.observation.include_previous_action` | Appends the 10-D action last applied (45-D total with the default base). **Changing this changes the observation shape and makes older checkpoints unloadable.** |
| `environment.terrain.preset` | `flat`, `uneven`, or `obstacles`. |
| `environment.terrain.mix` | List of terrain specs, each with an optional relative `weight`. One is drawn per reset. An explicit empty list is an error. |
| `environment.push_recovery` | Random pushes applied at the torso centre of mass for `duration_s` (0.1 s). |
| `environment.gait.clearance_point` | `com` (legacy) or `sole`. |

### Walker defaults

When a key is omitted, these defaults apply. The shipped configs override some of them (for example, `robot_walk_basic` uses `mass: 28.0`).

| Group | Key | Default |
|---|---|---|
| `sim` | `mass`, `friction`, `max_torque`, `timestep`, `frame_skip`, `settle_steps` | `3.0`, `0.8`, `35.0`, `0.00416667`, `4`, `30` |
| `reward` | `alive_bonus`, `forward_velocity_weight`, `target_velocity`, `velocity_sigma` | `0.25`, `2.0`, `1.0`, `0.5` |
| `reward` | `orientation_penalty_weight`, `torque_penalty_weight`, `fall_penalty` | `0.3`, `0.01`, `10.0` |
| `reward` | `velocity_mode` | `gaussian` |
| `termination` | `min_height`, `max_height`, `max_steps` | `0.18`, `1.5`, `1000` |
| `gait` | `cycle_period_s`, `stance_duty`, `clearance_point`, `touchdown_debounce_steps` | `1.0`, `0.6`, `com`, `3` |

Gait, slip and clearance reward weights default to `0.0`, so they are off unless a config enables them.

### Arena defaults

| Group | Key | Default |
|---|---|---|
| `environment` | `num_agents` | `2` (must be at least 2) |
| `sim` | `arena_half_extent`, `move_speed`, `spawn_jitter` | `1.0`, `0.05`, `0.1` |
| `morphology` | `base_size`, `health`, `episode_growth_scale` | `1.0`, `1.0`, `0.0` |
| `battle_rules` | `damage`, `attack_range`, `cooldown_steps`, `sensing_radius` | `0.05`, `0.2`, `3`, `2.0` |
| `battle_rules` | `max_steps`, `win_health_threshold`, `attack_falloff` | `400`, `0.0`, `linear` |
| `resources` | `initial_energy`, `max_energy`, `food_count`, `food_energy` | `1.0`, `1.0`, `2`, `0.35` |
| `reward` | `approach_weight`, `damage_scale` | `0.0`, `1.0` |

Unknown keys under the arena sections and the walker's `reward` and `termination` blocks are rejected at load time. The full arena reference, including the N-agent rules and what each setting does, is in [`docs/organism_arena_config.md`](docs/organism_arena_config.md).

---

## Environments

### Walker (`walker_bullet`)

A single-agent bipedal robot (Atlas-DRC class, 65.9 kg total, 28 kg torso) in PyBullet.

- **Physics:** 240 Hz simulation with `frame_skip: 4`, so the policy acts at 60 Hz.
- **Control:** PD position control by default. `action[i] = 0` maps to the joint's rest pose, not its midpoint.
- **Action (10-D):** `[rHip, rKnee, rAnkle, lHip, lKnee, lAnkle, rShoulder, rElbow, lShoulder, lElbow]`, each in `[-1, 1]`.
- **Observation:** torso height, orientation quaternion, linear and angular velocity, joint positions and velocities, mass and friction scale, and (in `v2`) foot-contact flags. Shape is 35 (`v1`, or `v2` coordinate-free), 37 (`v2`), or 45 with `include_previous_action`.
- **Termination:** contact-based (the torso touches the ground), with `min_height` and `max_height` as fallbacks. **Tilt does not end an episode.** `max_height: 1.5` m is a deliberate backstop against a PD-control exploit in which the robot launches itself. Do not loosen it without reading [`docs/walker_learning_history.md`](docs/walker_learning_history.md) first.
- **Terrain:** `flat`, `uneven` (a smooth heightfield), or `obstacles` (boxes every `obstacle_spacing ± jitter` metres), redrawn each reset unless `randomize: false`. Use `terrain.mix` to draw a different terrain each episode. The episode's terrain is reported in `info["terrain_preset"]`.
- **Randomisation:** reset noise on position and yaw, mass and friction scaling, and random pushes (`push_recovery`).

### Organism arena (`organism_arena_parallel`)

An N-agent PettingZoo `ParallelEnv` on a square arena (agents `agent_0` … `agent_{N-1}`, default N = 2). Agents spawn on a circle, and the policy is slot-symmetric, so one shared policy can fill any slot.

- **Action (3-D):** `[move_x, move_y, attack_trigger]`. The move vector is norm-clamped to `move_speed`. An attack fires when `attack_trigger > 0.5` and the attacker is off cooldown.
- **Observation (13-D):** own velocity (x, y), health fraction, energy fraction, size, relative position of the nearest opponent (x, y), that opponent's health, cooldown fraction, a visibility flag, and the nearest food position (x, y) with a visibility flag.
- **Combat:** an attack hits the nearest living opponent within `attack_range`. Damage falls off linearly with distance (or steps down with `attack_falloff: binary`), and it scales with the attacker's size. Attacks in one step resolve together, so mutual knockouts are draws.
- **Endings:** a knockout gives the winner `+1` and the loser `-1`. The episode also ends at `max_steps`, which counts as a timeout.
- **Training features:** self-play against a frozen league (`self_play`), annealing the dense per-hit reward to zero (`reward_annealing`), a win-rate-gated difficulty curriculum (`curriculum`), and morphology search (`morph-search`).

---

## Outputs

Each run writes to `outputs/<experiment_name>/seed_<seed>/`. Sweep runs go to `outputs/<experiment_name>/runs/<run_id>/seed_<seed>/`.

```
outputs/<experiment_name>/seed_<seed>/
├── run_metadata.json              # run provenance and configuration
├── checkpoints/
│   ├── final_model.zip            # final policy (load this for eval and replay)
│   ├── final_model_vecnormalize.pkl
│   ├── ppo_model_<steps>_steps.zip  # periodic checkpoints (checkpoint_every)
│   └── best_model.zip             # when evaluation.best_model.enabled is true
├── logs/
│   ├── PPO_*/                     # TensorBoard event files
│   ├── eval_metrics.csv           # one row per evaluation
│   └── monitor_env*.csv.monitor.csv
└── videos/                        # render-replay output (walker: MP4, arena: GIF)
```

Notes:

- **Observation normalisation.** When `normalize_observations: true`, the policy needs its matching `*_vecnormalize.pkl`. The loader looks for `<model-stem>_vecnormalize.pkl` first, then falls back to a shared `vecnormalize.pkl`. Keep the pair together.
- **Multi-seed.** `multi-seed` writes `outputs/<experiment>/multi_seed_summary/aggregate.csv`, with the mean, standard deviation, median, interquartile mean, and 95% CI of mean return across seeds.
- **Registry.** Runs are indexed in `outputs/run_registry.sqlite3`, which `registry` reads. The `outputs/` directory is disposable and gitignored, so copy anything you need to keep.

To watch progress:

```bash
tensorboard --logdir outputs/
```

The key curves are `rollout/ep_rew_mean` and `rollout/ep_len_mean`.

---

## Web GUI

The GUI is a Flask app for building experiments and watching them.

```bash
python -m rl_framework.cli.main gui --port 5001
```

Open `http://127.0.0.1:5001`. The wizard has four steps: **Environment**, **Parameters**, **Training**, and **Review & Launch**. It builds experiment configs from the same schema the CLI validates against.

Options: `--host` (default `127.0.0.1`) and `--port` (default `5001`).

The GUI reloads when its Python code changes. **Stop any running training before editing `gui/app.py`, because a reload kills in-process training.**

---

## Shipped experiments

All 29 configs in `src/rl_framework/configs/experiments/` pass validation. Start with these:

| Config | What it is |
|---|---|
| `walker_smoke_cpu` | Short walker run for checking the install. |
| `robot_walk_basic` | Walker baseline with target-velocity and torque-penalty sweep. |
| `robot_push_recovery` | Walker with random push perturbations. |
| `walker_curriculum_flat`, `_uneven`, `_obstacles` | Single-terrain walker runs. |
| `walker_balance_curriculum`, `walker_stochastic_balance_candidate` | Balance-focused walker variants. |
| `walker_reward_v2` … `v12` | Reward-shaping variants used in the walker learning history. |
| `walker_ppo_baseline`, `walker_sac_baseline`, `walker_td3_baseline` | Algorithm comparison baselines. |
| `organisms_fight_arena` | Arena with self-play, reward annealing, and a curriculum. |
| `organisms_growth_competition` | Arena variant with growth. |

Experiment results and the reasoning behind each variant are recorded in [`docs/walker_learning_history.md`](docs/walker_learning_history.md), including refuted hypotheses.

---

## Development

Set up as in [Installation](#installation), then run the same checks CI runs:

```bash
pytest -q --cov=src/rl_framework --cov-fail-under=60   # tests, coverage gate at 60%
ruff check src tests scripts                            # lint
mypy src --ignore-missing-imports                       # type check (blocking in CI)
python scripts/check_repo_policy.py                     # no tracked caches or venvs; lockfile completeness
```

- Dependencies are pinned in `requirements-lock.txt`. Every direct dependency in `pyproject.toml` must appear there, or CI fails.
- Run `pytest` and the policy check before committing. For changes to environment behaviour, the training pipeline, or rewards and termination, also run a short functional smoke test (an env reset and step, a few hundred training steps, or an eval of an existing checkpoint).
- Results, including negative ones, go in [`docs/walker_learning_history.md`](docs/walker_learning_history.md).
- Architecture decisions are in [`docs/decisions/`](docs/decisions/).

### Performance notes

- `num_envs: 24` matches the 24 physical cores of the reference machine. More workers than cores does not help.
- For `SubprocVecEnv` runs, export `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` so the workers do not compete for threads.

---

## Project layout

```
src/rl_framework/
├── cli/            # command-line entry point (cli/main.py)
├── configs/        # experiment YAMLs (configs/experiments/) and config helpers
├── core/           # shared config, training runners, registry, utilities
├── physics/
│   └── locomotion/ # walker env: config, terrain, robot, rewards, observation, analysis
├── games/
│   └── arena/      # organism arena env, tournaments, morphology search
├── gui/            # Flask app and static assets
└── plugins.py      # task registration hooks
tests/              # pytest suite
scripts/            # repo tooling (policy check, device benchmark, transfer report)
docs/               # status, config references, decisions, learning history
```

The `rl_framework/` directory at the repository root contains only an `__init__.py` and is not the package.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/STATUS.md`](docs/STATUS.md) | Current state of the project and known problems. |
| [`docs/walker_learning_history.md`](docs/walker_learning_history.md) | Walker decision ledger, including refuted hypotheses. |
| [`docs/organism_arena_config.md`](docs/organism_arena_config.md) | Full arena configuration reference. |
| [`docs/learning_quality_studies.md`](docs/learning_quality_studies.md) | The `quality-study` matrices and how to read them. |
| [`docs/training_presets.md`](docs/training_presets.md) | Training presets. |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | Validation evidence. |
| [`docs/decisions/`](docs/decisions/) | Architecture decision records. |
| [`VISUALIZATION_GUIDE.md`](VISUALIZATION_GUIDE.md) | Visualization and replay guide. |

## License

No license file is included yet.
