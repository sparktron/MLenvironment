# 🤖 RL Experiment Framework

**A powerful, modular reinforcement learning framework for training and evaluating single-agent locomotion and multi-agent competitive agents.**

Built on [Gymnasium](https://gymnasium.farama.org/), [PettingZoo](https://pettingzoo.farama.org/), [PyBullet](https://pybullet.org/), and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

✨ **Key features:** YAML-driven experiments • No code changes needed • GPU-accelerated (auto-detected) • Multi-seed aggregation • Self-play league • Domain randomization • Curriculum learning

---

## 🚀 Quick Start

```bash
# 1. Install
git clone https://github.com/sparktron/MLenvironment.git && cd MLenvironment
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Train a robot walker
python -m rl_framework.cli.main train --config-name robot_walk_basic

# 3. Monitor with TensorBoard
tensorboard --logdir outputs/

# 4. Evaluate and render video
python -m rl_framework.cli.main render-replay \
  --config-name robot_walk_basic \
  --model-path outputs/robot_walk_basic/seed_0/checkpoints/final_model.zip
```

**Or run in Docker:**

```bash
docker build -t rl-framework . && \
docker run --rm -v "$(pwd)/outputs:/app/outputs" rl-framework train --config-name robot_walk_basic
```

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  - [Local (pip)](#local-pip)
  - [Docker](#docker)
- [Web GUI](#-web-based-gui)
- [CLI Reference](#-cli-commands)
  - [train](#train)
  - [eval](#eval)
  - [sweep](#sweep)
  - [multi-seed](#multi-seed)
  - [render-replay](#render-replay)
  - [registry](#registry)
  - [quality-study](#quality-study)
- [Configuration](#configuration)
  - [YAML Structure](#yaml-structure)
  - [Included Experiments](#included-experiments)
  - [Environment Types](#environment-types)
- [Features](#features)
  - [Domain Randomisation](#domain-randomisation)
  - [Sensor Noise](#sensor-noise)
  - [Action Latency](#action-latency)
  - [Curriculum Learning](#curriculum-learning)
  - [Parallel CPU Rollouts](#parallel-cpu-rollouts)
  - [Multi-Seed Aggregation](#multi-seed-aggregation)
  - [Self-Play League](#self-play-league)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Extending the Framework](#extending-the-framework)
  - [Add a New Environment](#add-a-new-environment)
  - [Add a New Morphology](#add-a-new-morphology)
  - [Swap the RL Algorithm](#swap-the-rl-algorithm)
- [Architecture](#architecture)
- [Performance Tuning](#performance-tuning)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Maintenance Notes](#maintenance-notes)
- [Changelog](#changelog)
- [License](#license)

---

## Requirements

| Requirement | Minimum |
|---|---|
| **Python** | 3.10+ |
| **pip** | 21+ |
| **OS** | Linux / macOS (Windows via WSL) |

> 💡 PyBullet GUI rendering on Linux requires an X11 display

### Dependencies (auto-installed)

| Package | Version | Purpose |
|---|---|---|
| gymnasium | >= 0.29 | Single-agent environment API |
| pettingzoo | >= 1.24 | Multi-agent environment API |
| supersuit | >= 3.9 | PettingZoo-to-VecEnv wrappers |
| stable-baselines3 | >= 2.3 | PPO training + checkpoints |
| pybullet | >= 3.2 | Physics simulation |
| omegaconf | >= 2.3 | Config loading and composition |
| numpy | >= 1.26 | Numerical operations |
| tensorboard | >= 2.15 | Training visualisation |
| PyYAML | >= 6.0 | YAML parsing |
| flask | >= 3.0 | Web GUI server |

### Optional dev dependencies

| Package | Version | Purpose |
|---|---|---|
| pytest | >= 8.0, != 9.0.2 | Test runner |
| pytest-cov | >= 7.1 | Coverage reporting and CI coverage gate |
| ruff | >= 0.4 | Linting and formatting |

---

## 📦 Installation

### 💻 Local (pip)

```bash
git clone https://github.com/sparktron/MLenvironment.git && cd MLenvironment
python -m venv .venv && source .venv/bin/activate
pip install -e .                    # Core dependencies
pip install -e ".[dev]"             # + dev tools (pytest, pytest-cov, ruff)
# CI uses pinned dependencies from requirements-lock.txt
```

**Verify:**

```bash
python -c "from rl_framework.cli.main import main; print('✓ Ready')"
```

### 🐳 Docker

```bash
docker build -t rl-framework .
docker run --rm -v "$(pwd)/outputs:/app/outputs" \
  rl-framework train --config-name robot_walk_basic
```

*Image: `python:3.11-slim` + editable install. All CLI args forwarded.*

---

## 🌐 Web-Based GUI

**New!** Launch the interactive web GUI to set up and monitor experiments without touching YAML or the terminal.

```bash
python -m rl_framework.cli.main gui              # http://127.0.0.1:5001
python -m rl_framework.cli.main gui --port 8080   # custom port
```

### GUI Features

| Feature | Description |
|---|---|
| **4-Step Experiment Wizard** | Choose environment → Configure parameters → Set training hyperparameters → Review & launch |
| **Visual Parameter Editor** | All fields with descriptions, type hints, and min/max ranges — no YAML editing |
| **Template Loading** | Start from any existing config (robot_walk_basic, organisms_fight_arena, etc.) |
| **Real-Time Dashboard** | Live training metrics: reward, loss, entropy, learning rate, timesteps — updates every 2 seconds |
| **Reward Chart** | Interactive canvas chart tracking episode reward over training time |
| **Live Parameter Tuning** | Modify learning rate, reward weights, and termination thresholds **during training** — changes apply at the next rollout |
| **Outputs Browser** | Browse and inspect completed experiments and their saved checkpoints |
| **Run Analysis** | Compare registry-backed metrics and artifacts, surface best checkpoints, launch replays, and rate self-play leagues |

**Example flow:**
1. Click "Walker (Locomotion)" environment card
2. Fill in experiment name and environment parameters (GUI validates ranges)
3. Set total_timesteps, learning_rate, batch_size
4. Review YAML preview
5. Click "Launch Training" — training runs in a background thread
6. Switch to Dashboard tab to watch real-time metrics and apply live parameter tweaks

---

## 🎮 CLI Commands

**Syntax:** `python -m rl_framework.cli.main <command> --config-name <name> [options]`

`gui` and `registry` do not require `--config-name`.

| Flag | Required | Default | Description |
|---|---|---|---|
| `--config-name` | **Yes** | — | YAML config name (without extension) |
| `--config-dir` | No | `src/rl_framework/configs/experiments` | Config directory |
| `--model-path` | Eval/replay only | — | Path to trained model `.zip` |
| `--seeds` | multi-seed only | — | Comma-separated: `0,1,2,3,4` |
| `--max-workers` | multi-seed only | `1` when `num_envs > 1`; otherwise CPU count | Parallel worker processes |
| `--device` | No | config value | Override training device for this run: `auto`, `cpu`, `cuda`, `cuda:<N>` |
| `--resume` | train only | — | Path to a saved matching-algorithm `.zip` to continue training from |
| `--trials` | morph-search only | 5 | Number of morphology mutations to evaluate |

### 🏋️ `train` — Train an SB3 agent

```bash
python -m rl_framework.cli.main train --config-name robot_walk_basic
```

**Output files:**

```
outputs/
├── <experiment_name>/
│   └── seed_<N>/
│       ├── checkpoints/final_model.zip          # Trained policy
│       ├── checkpoints/vecnormalize.pkl         # Normalization stats
│       └── logs/
│           ├── events.out.tfevents.* (TensorBoard)
│           └── eval_metrics.csv
```

**Monitor in real-time:**

```bash
tensorboard --logdir outputs/
```

### 📊 `eval` — Evaluate a trained policy

```bash
python -m rl_framework.cli.main eval \
  --config-name robot_walk_basic \
  --model-path outputs/robot_walk_basic/seed_42/checkpoints/final_model.zip
```

**Output:**
- Stdout: `mean_return` and `std_return`
- CSV append: `outputs/<experiment>/seed_<N>/logs/eval_metrics.csv`

Works for **locomotion** (Gymnasium) and **organism arena** (PettingZoo).

### 🔍 `sweep` — Hyperparameter grid search

```bash
python -m rl_framework.cli.main sweep --config-name robot_walk_basic
```

Runs a **Cartesian-product** sweep over `sweep.parameters` in your YAML:

```yaml
sweep:
  parameters:
    environment.reward.target_velocity: [0.5, 1.0, 1.5]        # 3 values
    environment.reward.torque_penalty_weight: [0.005, 0.01]   # 2 values
    # Result: 3 × 2 = 6 independent training runs
```

### 🎲 `multi-seed` — Train across multiple seeds (statistical significance)

```bash
python -m rl_framework.cli.main multi-seed \
  --config-name robot_walk_basic --seeds 0,1,2,3,4

# Run seeds in parallel (default: all CPUs):
python -m rl_framework.cli.main multi-seed \
  --config-name robot_walk_basic --seeds 0,1,2,3,4 --max-workers 4

# Force CPU for this invocation (without editing YAML):
python -m rl_framework.cli.main multi-seed \
  --config-name robot_walk_basic --seeds 0,1,2,3,4 --max-workers 4 --device cpu

# Force sequential (useful when training already saturates CPUs via num_envs):
python -m rl_framework.cli.main multi-seed \
  --config-name robot_walk_basic --seeds 0,1,2 --max-workers 1
```

Trains and evaluates **the same config** across multiple random seeds for statistical rigor. Seeds run in **parallel by default** using separate processes.

### 🧪 Automated 4-run CPU/GPU benchmark matrix (CLI only)

Run the fixed matrix:
- `CPU-4workers`
- `CPU-8workers`
- `GPU-1worker`
- `GPU-2workers`

```bash
python scripts/benchmark_device_matrix.py \
  --config-name robot_walk_basic \
  --seeds 0,1,2,3
```

If your shell cannot find the script path, run the module entrypoint instead:

```bash
python -m rl_framework.benchmark_device_matrix \
  --config-name robot_walk_basic \
  --seeds 0,1,2,3
```

By default, the benchmark overrides each run to `--total-timesteps 20000` so the matrix completes quickly.

The script runs regimes from smaller to larger worker counts (`CPU-4workers` → `CPU-8workers` → `GPU-1worker` → `GPU-2workers`). It prints the matrix version, resolved script path, and execution order before launching any workers. It streams each regime's terminal output live, prints periodic heartbeats while waiting, appends progress events to `outputs/benchmark_device_matrix_progress.jsonl`, persists completed regimes in `outputs/benchmark_device_matrix_state.json`, measures wall-clock runtime, and prints a JSON summary plus a winner. Resume a matching interrupted matrix with `--resume`; changed inputs require a new state file.

If the first run line still shows `CPU-10workers` or lacks `[matrix] version:` / `[matrix] script:` lines, you are running an older checkout or a different copy of `scripts/benchmark_device_matrix.py`; update that checkout before rerunning.

**Decision rule (default):**
- Find the best `mean_return_mean` across the 4 regimes.
- Keep regimes within **3%** of that best reward.
- Choose the **fastest** (lowest wall-clock time) among those.

You can tighten/relax reward guardrails:

```bash
python scripts/benchmark_device_matrix.py \
  --config-name robot_walk_basic \
  --seeds 0,1,2,3 \
  --reward-tolerance-ratio 0.02
```

Run a longer benchmark:

```bash
python scripts/benchmark_device_matrix.py \
  --config-name robot_walk_basic \
  --seeds 0,1,2,3 \
  --total-timesteps 100000
```

If you suspect a stall, tune watchdogs:

```bash
python scripts/benchmark_device_matrix.py \
  --config-name robot_walk_basic \
  --seeds 0,1,2,3 \
  --inactivity-timeout-s 600 \
  --heartbeat-s 15
```

Maximum debug tracing (default is already enabled):

```bash
python scripts/benchmark_device_matrix.py \
  --config-name robot_walk_basic \
  --seeds 0,1,2,3 \
  --debug
```

**Output:**

```
outputs/
├── <experiment>/
│   ├── seed_0/, seed_1/, ...
│   └── multi_seed_summary/
│       └── aggregate.csv
```

**CSV format:**

```csv
seed,mean_return
0,142.5
1,138.2
...
aggregate_mean,145.98
aggregate_std,5.76
```

### 🎬 `render-replay` — Generate videos of trained policies

```bash
python -m rl_framework.cli.main render-replay \
  --config-name robot_walk_basic \
  --model-path outputs/robot_walk_basic/seed_42/checkpoints/final_model.zip
```

**Output:**
- Gymnasium envs (locomotion): MP4 in `outputs/<experiment>/seed_<N>/videos/`
- PettingZoo parallel envs (organism arena): animated GIF (`replay.gif`) in the same folder

For policies trained with `normalize_observations: true`, replay loads the saved
VecNormalize sidecar next to the model before calling `predict()`.

### ⏯️ Resuming training

Pass `--resume <checkpoint.zip>` to `train` to continue from a saved model.
Both the PPO policy/optimizer state and the sibling `vecnormalize.pkl`
running stats are restored, and TensorBoard timesteps continue where they
left off.

```bash
python -m rl_framework.cli.main train \
  --config-name robot_walk_basic \
  --resume outputs/robot_walk_basic/seed_42/checkpoints/final_model.zip
```

### 🧬 `morph-search` — Random morphology search

```bash
python -m rl_framework.cli.main morph-search \
  --config-name organism_arena_base --trials 8
```

Mutates the organism's morphology parameters (base size, health) across N
trials, trains each, and ranks trials by round-robin tournament Elo
(`morphology_search.scoring: tournament_elo`, the only supported/default
mode). Currently scoped to `organism_arena_parallel`. The legacy
shared-policy `mean_return` mode is rejected outright — the arena is
zero-sum, so a shared policy's mean return sums to ~0 by construction and
would rank trials on noise, not skill.

### 🗂️ `registry` — Inspect, export, and prune run metadata

```bash
# Summary counts, including missing artifact paths
python -m rl_framework.cli.main registry --registry-action inspect

# Full JSON backup of every registry table
python -m rl_framework.cli.main registry --registry-action export \
  --json-out outputs/registry-export.json

# Preview stale analysis-job cleanup, then rerun without --dry-run to apply
python -m rl_framework.cli.main registry --registry-action prune \
  --prune-target analysis-jobs --status interrupted,failed \
  --older-than-days 30 --dry-run

# Remove artifact index rows whose files no longer exist
python -m rl_framework.cli.main registry --registry-action prune \
  --prune-target artifacts --missing-only
```

Use `--base-dir` when the registry is outside `outputs`. Run pruning also
removes the selected runs' event, tuning, and artifact-index rows in one
transaction. Maintenance never deletes checkpoint or other artifact files.
An unfiltered prune is rejected unless `--all` is supplied explicitly.

### 🧭 `quality-study` — Reproducible learning-quality comparisons

Run the Priority 3 walker, arena, and algorithm study matrices with durable,
resumable state and machine-readable reports:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m rl_framework.cli.main quality-study \
  --study all --seeds 0,1,2
```

`--study` also accepts `walker`, `arena`, or `algorithms`. The default promotion
budgets are 750k walker steps, 30k arena steps, and 500k algorithm steps plus a
900-second wall-clock comparison. Override them with `--study-step-budget`,
`--study-wall-clock-seconds`, and `--study-eval-episodes`; use
`--resume-incomplete` to continue from `state.json`. Results are written to
`outputs/quality_studies/{report.json,report.md}`. See
[`docs/learning_quality_studies.md`](docs/learning_quality_studies.md) for the
matrix, metrics, readiness gates, and preliminary findings.

---

## ⚙️ Configuration

Every experiment is defined by a **single YAML file** — no code changes needed. Below is the complete schema:

### Full YAML Schema

```yaml
# ─── Identity ───────────────────────────────────────────────────
experiment_name: my_experiment       # Used as the output directory name
seed: 42                             # Global random seed

# ─── Environment ────────────────────────────────────────────────
environment:
  type: walker_bullet                # "walker_bullet" or "organism_arena_parallel"
  seed: 42                           # Environment-level seed (should match global)

  # --- Locomotion-specific (type: walker_bullet) ---
  sim:
    gravity: -9.81                   # Gravitational acceleration (m/s^2)
    mass: 3.0                        # Robot base mass (kg)
    friction: 0.9                    # Ground lateral friction coefficient
    max_force: 35.0                  # Maximum applied force (N)

  reward:
    alive_bonus: 0.25                # Small uprightness incentive; locomotion dominates
    forward_velocity_weight: 2.0     # Weight on velocity-tracking reward
    target_velocity: 1.0             # Desired forward velocity (m/s)
    orientation_penalty_weight: 1.0  # Weight on roll+pitch penalty
    torque_penalty_weight: 0.01      # Weight on action magnitude penalty

  termination:
    min_height: 0.12                 # Episode ends if z < this (metres)
    max_tilt_radians: 0.8            # Episode ends if |roll| or |pitch| > this
    max_steps: 800                   # Episode truncates after this many steps

  reset_randomization:
    position_xy_noise: 0.02          # Uniform noise on start x, y position
    yaw_noise: 0.08                  # Uniform noise on start yaw angle (radians)

  domain_randomization:
    mass_scale_range: [0.95, 1.05]   # Multiply base mass by U(low, high) each reset
    friction_range: [0.9, 1.1]       # Multiply base friction by U(low, high) each reset
    sensor_noise_std: 0.0            # Gaussian std added to observations (0 = off)
    action_latency_steps: 0          # FIFO delay on actions (0 = off)

  observation:
    version: v2                      # v1 legacy; v2 adds two foot-contact bits
    coordinate_free: true             # v2 only: remove global x/y position

  terrain:
    preset: flat                     # flat | uneven | obstacles | push_recovery
    height: 0.025                    # Uneven terrain height variation (m)
    obstacle_height: 0.10            # Obstacle height (m)
    push_recovery:
      interval_steps: 120
      start_step: 60
      force: 180.0                   # Lateral impulse magnitude

  # --- Multi-agent-specific (type: organism_arena_parallel) ---
  sim:
    arena_half_extent: 1.2           # Arena boundary in each direction
    move_speed: 0.05                 # Base movement speed before size scaling
    collision_radius: 0.08           # Base organism collision radius
    speed_size_exponent: 1.0         # Larger bodies move more slowly

  morphology:
    base_size: 1.0                   # Agent body scale at episode start
    episode_growth_scale: 0.0        # Size increase per step (0 = no growth)
    health: 1.2                      # Base health (scaled by size)

  battle_rules:
    damage: 0.06                     # Base damage per hit
    collision_damage: 0.0           # Optional contact damage, scaled by body size
    attack_range: 0.2                # Distance threshold for attacks
    cooldown_steps: 3                # Steps between consecutive attacks
    max_steps: 400                   # Episode truncation limit
    win_health_threshold: 0.0        # Health at or below triggers termination

  resources:
    initial_energy: 1.0              # Spawn energy (must be <= max_energy)
    max_energy: 1.0
    movement_cost: 0.01              # Cost per movement command magnitude
    attack_cost: 0.04                # Attack requires and consumes energy
    food_count: 2
    food_energy: 0.35
    food_radius: 0.10
    food_respawn_steps: 40
    food_placement: uniform          # uniform | center (contested patch)

# ─── Training ───────────────────────────────────────────────────
training:
  algorithm: PPO                     # PPO (default), SAC, or TD3 (walker-only)
  policy: MlpPolicy                  # SB3 policy class
  total_timesteps: 20000             # Total training steps
  learning_rate: 0.0003              # PPO learning rate
  n_steps: 1024                      # Steps per rollout buffer
  batch_size: 256                    # Minibatch size
  checkpoint_every: 5000             # Save a checkpoint every N environment steps
  normalize_observations: true       # Wrap env in VecNormalize
  check_nans: false                  # Set true to fail fast on NaN/Inf values
  num_envs: 8                        # >1 uses SubprocVecEnv for parallelism
  device: cpu                        # Default for bundled MLP experiments
  # device: auto                     # Opt into CUDA discovery when appropriate
  torch_num_threads: 1               # Optional PyTorch update-thread cap
  worker_start_method: spawn         # Optional: fork | forkserver | spawn

# ─── Evaluation ─────────────────────────────────────────────────
evaluation:
  episodes: 5                        # Number of eval episodes (locomotion)
  max_steps: 400                     # Max steps per eval episode (organism)

# ─── Output ─────────────────────────────────────────────────────
output:
  base_dir: outputs                  # Root directory for all experiment outputs

# ─── Sweep ──────────────────────────────────────────────────────
sweep:
  parameters:                        # Dotted-key → list-of-values
    environment.reward.target_velocity: [0.5, 1.0, 1.5]
    environment.reward.torque_penalty_weight: [0.005, 0.01]

# ─── Multi-Seed ────────────────────────────────────────────────
multi_seed:
  seeds: [0, 1, 2, 3, 4]            # Seeds for aggregate runs

# ─── Curriculum ─────────────────────────────────────────────────
curriculum:
  enabled: false                     # Set true to activate
  level_up_threshold: 150.0          # Mean reward to advance a level
  max_level: 3                       # Maximum curriculum level
  level: 0                           # Starting level (usually 0)
  level_params:                      # Overrides applied at each level
    1:
      reward.target_velocity: 1.0
      termination.max_tilt_radians: 0.9
    2:
      reward.target_velocity: 1.5
      termination.max_tilt_radians: 0.7
    3:
      reward.target_velocity: 2.0
      termination.max_tilt_radians: 0.5

# ─── Self-Play ──────────────────────────────────────────────────
self_play:
  enabled: false                     # Set true to activate (organism only)
  snapshot_freq: 5000                # Save a snapshot every N timesteps
  max_league_size: 10                # Maximum stored opponent snapshots
```

### 📋 Included Experiment Configs

| Config | Environment | Description |
|---|---|---|
| **robot_walk_basic** | `walker_bullet` | Bipedal locomotion with domain randomization. Sweeps velocity & torque penalty. ✅ Good for getting started |
| **robot_push_recovery** | `walker_bullet` | Lateral push-recovery curriculum plus aggressive mass/friction randomization. |
| **walker_curriculum_flat** | `walker_bullet` | Flat-ground speed curriculum. |
| **walker_curriculum_uneven** | `walker_bullet` | Uneven-terrain speed curriculum. |
| **walker_curriculum_obstacles** | `walker_bullet` | Low-obstacle speed curriculum. |
| **organisms_fight_arena** | `organism_arena_parallel` | Self-play combat arena with parallel rollout workers. Sweeps attack range & cooldown. |
| **organisms_growth_competition** | `organism_arena_parallel` | Two-agent arena with in-episode growth. Sweeps base size & damage. |

### 🎮 Environment Types

#### `walker_bullet` — Single-agent locomotion (Gymnasium)

Rigid-body bipedal walker in PyBullet. Goal: maintain upright posture while tracking target velocity.

| Property | Value |
|---|---|
| **Observation** | v1 `Box(35,)`; v2 `Box(37,)` adds right/left foot contacts. Coordinate-free v2 is `Box(35,)` without global x/y. |
| **Action** | `Box(10,)` — normalized joint targets for hips, knees, ankles, shoulders, and elbows |
| **Reward** | alive bonus + velocity tracking − orientation penalty − torque penalty |

#### `organism_arena_parallel` — Multi-agent competitive (PettingZoo)

2D N-agent arena with movement, attacks, optional in-episode growth, and an
on-disk self-play league for frozen past-self opponents.

| Property | Value |
|---|---|
| **Observation** | `Box(13,)` — velocity, health, energy, size, nearest-opponent state, cooldown, and nearest-food state |
| **Action** | `Box(3,)` — move-x, move-y, attack trigger (> 0.5) |
| **Reward** | damage dealt − damage received ± win/loss bonus |

---

## ⭐ Features

### 🔀 Domain Randomization

Mass and friction are randomised at each episode reset, controlled by:

```yaml
domain_randomization:
  mass_scale_range: [0.8, 1.2]     # base_mass *= U(0.8, 1.2)
  friction_range: [0.7, 1.3]       # base_friction *= U(0.7, 1.3)
```

### 🔊 Sensor Noise

Gaussian noise injection for **sim-to-real transfer**. Prevents overfitting to perfect simulator observations.

```yaml
domain_randomization:
  sensor_noise_std: 0.01           # standard deviation (0 = disabled)
```

### ⏱️ Action Latency

FIFO delay between agent decisions and physics application — simulates real-world communication/actuator delays.

```yaml
domain_randomization:
  action_latency_steps: 2          # 0 = off, N > 0 = apply action from N steps ago
```

### 📚 Curriculum Learning

Automatically increases difficulty as the agent improves. SB3 callback monitors `rollout/ep_rew_mean` and advances levels when threshold is exceeded.

```yaml
curriculum:
  enabled: true
  level_up_threshold: 150.0   # default threshold for all levels
  # Optional: per-level thresholds (override the default above)
  level_up_thresholds:
    0: 100.0    # threshold to leave level 0
    1: 150.0
    2: 200.0
  max_level: 3
  level_params:
    1:
      reward.target_velocity: 1.0
    2:
      reward.target_velocity: 1.5
    3:
      reward.target_velocity: 2.0
```

Level parameters use dotted-key notation to override any value in the environment config at runtime. Per-level thresholds let you set different advancement bars at each difficulty stage.

### 🖥️ Training Device

The bundled MLP experiments default to `"cpu"`. PyBullet rollouts are CPU-only,
and these small policies are typically slower on GPU because transfer overhead
outweighs the update cost. Override via the YAML key when using a larger policy:

```yaml
training:
  device: cpu       # default for bundled MLP experiments
  # device: auto    # GPU if available, otherwise CPU
  # device: cuda    # require GPU (fails if CUDA is unavailable)
  # device: cuda:1  # pin to a specific GPU on multi-GPU machines
  # device: cpu     # force CPU
```

Accepted values: `auto`, `cpu`, `cuda`, `cuda:<N>` (e.g. `cuda:0`). Any other string is rejected at config-validation time with a clear error message.

> **Note:** PyBullet physics simulation is CPU-only. GPU acceleration applies to the PPO neural network only and remains useful for larger networks or image-based (`CnnPolicy`) policies.

### ⚡ Parallel CPU Rollouts

Run multiple envs across processes for faster PyBullet rollout collection.
Benchmark before pushing walker runs toward the physical core count, because
larger rollout batches can change PPO update stability.

```yaml
training:
  num_envs: 8                       # SubprocVecEnv (1 = DummyVecEnv, no overhead)
  check_nans: false                 # Set true while diagnosing unstable runs
  torch_num_threads: 1              # Optional cap for PyTorch update threads
  worker_start_method: spawn        # Optional: fork | forkserver | spawn
```

The two runtime controls are opt-in. `torch_num_threads` constrains the PPO
update process; `worker_start_method` is passed to `SubprocVecEnv` and only
matters when `num_envs > 1`. Use `spawn` when process isolation is more
important than startup time, and benchmark before making either setting a
long-run default.

See [local training presets](docs/training_presets.md) for ready-to-run smoke,
overnight, high-throughput, arena self-play, and multi-seed commands.

### 📈 Multi-Seed Aggregation

Train across multiple seeds for **statistically rigorous** results. When each
training run uses parallel rollout workers (`training.num_envs > 1`),
multi-seed defaults to one sequential worker to avoid nested process
oversubscription. Set `--max-workers` explicitly only after sizing the combined
process count. See [`multi-seed` command](#-multi-seed--train-across-multiple-seeds-statistical-significance) above.

### 🏆 Self-Play League

Breaks co-adaptation by maintaining a league of frozen past opponents. The live policy trains against random league members, not copies of itself.

```yaml
self_play:
  enabled: true
  snapshot_freq: 5000               # Freeze a snapshot every 5k steps
  max_league_size: 10               # Keep ≤ 10 past versions
```

Snapshots saved to `checkpoints/league/`. Oldest pruned automatically.
With `self_play.enabled: true`, `training.num_envs > 1` uses the native SB3
vector-env path and can parallelize arena rollout collection. Shared-policy
arena training still uses the SuperSuit path and must keep `num_envs: 1`.
The shared-policy adapter supplies `render_mode` metadata locally, so this
otherwise-compatible boundary no longer emits SB3's missing-render-mode warning.
Arena damage-reward annealing updates environments at rollout boundaries rather
than every step, avoiding unnecessary cross-process calls during parallel runs.
For N-agent matches, the live policy's training episode ends immediately on its
own knockout instead of collecting inert spectator transitions.

---

## 📂 Project Structure

```text
MLenvironment/
├── README.md
├── Dockerfile
├── pyproject.toml
├── requirements.txt
│
├── src/rl_framework/
│   ├── __init__.py
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py                     # CLI entry point and command dispatcher
│   │
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── app.py                      # Flask web app + REST API
│   │   ├── training_manager.py         # Background training orchestrator
│   │   ├── templates/
│   │   │   └── index.html              # Single-page wizard + dashboard
│   │   └── static/
│   │       ├── style.css               # Dark-themed responsive UI
│   │       └── app.js                  # Frontend: wizard, polling, tuning
│   │
│   ├── configs/experiments/
│   │   ├── robot_walk_basic.yaml
│   │   ├── robot_push_recovery.yaml
│   │   ├── organisms_fight_arena.yaml
│   │   └── organisms_growth_competition.yaml
│   │
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── base.py                     # DomainRandomizationConfig, CurriculumConfig, EnvContext
│   │   ├── registry.py                 # make_env() factory
│   │   ├── locomotion/
│   │   │   ├── __init__.py
│   │   │   ├── walker_bullet.py        # Gymnasium walker env (PyBullet)
│   │   │   ├── dynamics.py             # Force / torque application
│   │   │   ├── rewards.py              # Composite reward function
│   │   │   └── terminations.py         # Episode termination logic
│   │   └── organisms/
│   │       ├── __init__.py
│   │       └── arena_parallel.py       # PettingZoo two-agent arena
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sb3_runner.py               # PPO training loop
│   │   ├── eval_runner.py              # Evaluation + metrics CSV
│   │   ├── sweep.py                    # Cartesian hyperparameter sweep
│   │   ├── multi_seed_runner.py        # Multi-seed train+eval+aggregate
│   │   ├── curriculum_callback.py      # SB3 callback for curriculum learning
│   │   ├── self_play_callback.py       # SB3 callback for self-play league
│   │   └── live_tuning_callback.py     # SB3 callback for live parameter updates
│   │
│   ├── evolution/
│   │   └── simple_search.py            # Random morphology mutation hook
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                   # OmegaConf YAML loader
│       └── logging_utils.py            # Directory creation + CSV append
│
└── tests/
    ├── test_env_api.py                 # Gymnasium + PettingZoo API compliance
    └── test_reproducibility.py         # Seed determinism tests
```

---

## ✅ Testing

```bash
pytest                                    # Run all tests
pytest tests/test_env_api.py -v          # Specific test
ruff check src/ tests/                    # Lint & format check
python scripts/check_repo_policy.py       # Lockfile + tracked artifact policy checks
```

### Reproducibility And Best Models

Use strict provenance to require a complete run manifest, and deterministic mode
to enforce framework-level random seeds, deterministic PyTorch algorithms, one
PyTorch update thread, and spawned rollout workers:

```yaml
reproducibility:
  strict: true
  deterministic: true
```

Each training run writes `run_metadata.json` with the resolved config, hash,
git commit, lockfile hash, and runtime details. Deterministic mode improves
repeatability but cannot promise bit-identical PyBullet results across platforms
or drivers.

Walker experiments can also retain the best observed policy, with a separate
VecNormalize evaluation environment and a matching sidecar:

```yaml
evaluation:
  episodes: 5
  best_model:
    enabled: true
    eval_every: 50000
    episodes: 5
```

This writes `checkpoints/best_model.zip` and
`checkpoints/best_model_vecnormalize.pkl`. Best-model selection is currently
supported for `walker_bullet`; use `arena-eval`/`arena-tournament` for arena
checkpoint comparison.

| Test | Verifies |
|---|---|
| `test_env_api.py` | Gymnasium & PettingZoo API compliance (shapes, types, interfaces) |
| `test_reproducibility.py` | Determinism: identical seeds → identical observations |

---

## 🔧 Extending the Framework

### ➕ Add a New Environment

1. Create module in `src/rl_framework/envs/locomotion/` (Gymnasium) or `src/rl_framework/envs/organisms/` (PettingZoo)
2. Separate **dynamics** → **rewards** → **terminations** into distinct files
3. Register in `src/rl_framework/envs/registry.py`:
   ```python
   def make_env(env_type: str, cfg: dict[str, Any]) -> gym.Env | ParallelEnv:
       if env_type == "my_env": return MyEnv(cfg)
   ```
4. Add YAML config in `src/rl_framework/configs/experiments/`
5. Write tests in `tests/` (API compliance + determinism)

### 🦾 Add a New Morphology

1. Add keys to `morphology` block in YAML
2. Parse in `arena_parallel.py` (`_spawn_agent()`, `_current_size()`)
3. Add to `sweep.parameters` for grid search
4. Optionally use `evolution/simple_search.py` for random mutation

### 🤖 Swap the RL Algorithm

Set `training.algorithm: SAC` or `TD3` in a walker config. The runner,
evaluation flow, checkpoints, and VecNormalize sidecars are shared with PPO.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLI (main.py)                                │
│  train│eval│sweep│multi-seed│render-replay │ gui (new!)            │
├──────┴────┴─────┴──────────┴──────────────┴────────────────────────┤
│                                                                     │
│  ┌──────────────────────┐      ┌──────────────────────────────┐     │
│  │  CLI Runners         │      │  Flask GUI (gui/app.py)      │     │
│  │  ├─ sb3_runner       │      │  REST API + web interface    │     │
│  │  ├─ eval_runner      │      │  └─ training_manager.py      │     │
│  │  ├─ sweep            │      │     (background thread)      │     │
│  │  └─ multi_seed       │      └──────────────────────────────┘     │
│  └──────────┬───────────┘                    │                     │
│             │                               │                     │
│  ┌──────────┴───────────────────────────────┴──────────────────┐    │
│  │              Callbacks (SB3)                                │    │
│  │  ├── CheckpointCallback                                    │    │
│  │  ├── CurriculumCallback                                    │    │
│  │  ├── SelfPlayCallback                                      │    │
│  │  └── LiveTuningCallback (reads JSON for real-time tuning)  │    │
│  └──────────────────────┬────────────────────────────────────┘     │
│                         │                                          │
│  ┌──────────────────────┴────────────────────────────────────┐     │
│  │               Environment Layer                           │     │
│  │  registry.py → make_env()                                 │     │
│  │  ┌──────────────────────┐  ┌───────────────────────────┐  │     │
│  │  │   walker_bullet      │  │ organism_arena_parallel   │  │     │
│  │  │   (Gymnasium)        │  │ (PettingZoo Parallel)     │  │     │
│  │  │   dynamics.py        │  │ battle rules + growth     │  │     │
│  │  │   rewards.py         │  └───────────────────────────┘  │     │
│  │  │   terminations.py    │                                 │     │
│  │  └──────────────────────┘                                 │     │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Utils: config.py (OmegaConf) │ logging_utils.py (CSV)      │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Design Decisions

| Concern | Choice | Rationale |
|---|---|---|
| Single-agent API | Gymnasium | Stable, widely supported, simple step/reset contract |
| Multi-agent API | PettingZoo Parallel | Matches SB3 shared-policy path via SuperSuit |
| Physics backend | PyBullet | Lightweight, no licence friction; swap to MuJoCo via registry |
| RL algorithm | SB3 PPO, SAC, TD3 | PPO default; SAC/TD3 walker-only baselines |
| Config system | OmegaConf / Hydra | Readable YAML + structured overrides |
| Experiment ops | TensorBoard + CSV + SB3 checkpoints | Low-dependency, works offline |

---

## ⚡ Performance Tuning

| Technique | Action | Expected Gain |
|---|---|---|
| **Parallel rollouts** | Set `training.num_envs: 4+` | ~3-4× on 4 cores |
| **GPU training** | `device: cpu` for bundled MLP configs | Set `auto`/`cuda` explicitly for larger nets or CnnPolicy |
| **Longer rollouts** | Increase `training.n_steps` | Better sample efficiency |
| **Reduce logging** | Set `verbose: 0` | Minor |

**For massive scale:** Consider [Brax](https://github.com/google/brax) (JAX vectorization) or [RLlib](https://www.ray.io/rllib) (distributed).

---

## ⚠️ Known Limitations

| Limitation | Workaround |
|---|---|
| **CPU physics** | PyBullet simulation is CPU-only; GPU speeds up the neural network only |
| **Shared-policy arena parallelism** | Shared-policy SuperSuit arena training is single-process; enable `self_play` to use the native SB3 parallel path |
| **Single active GUI run** | Use CLI runs for concurrent experiments until GUI run orchestration grows beyond the current single-run policy |
| **Sequential sweeps** | Use `xargs` / `GNU parallel` / job scheduler for parallel hyperparameter sweeps |

---

## Roadmap

The active development plan lives in [`docs/open_items_todo.md`](docs/open_items_todo.md). The next work is grouped into:

- Priority 0 correctness fixes are currently cleared for the arena self-play validation path; new confirmed bugs should be added here first.
- Learning-quality decisions are made through the resumable `quality-study`
  matrices; candidate defaults remain gated on promotion-scale evidence.
- Throughput and operations: multi-run GUI orchestration beyond the current
  single-run policy.
- Feature additions should be proposed only after the study reports identify a
  measured bottleneck or strategic gap.

Arena mechanics now include size-scaled collision radii, optional body-contact
damage, uniform or center-contested food placement, energy costs for movement
and attacks, and inverse size/speed scaling. Terminal episode metrics report
contacts, damage, food pickups, attacks, and energy depletion for tournament
studies. The richer mechanics expand the arena observation from 8 to 13 values,
so existing arena checkpoints are not compatible with them.

Walker observation v2 is checkpoint-incompatible with v1: v2 adds right/left
foot-contact signals, and coordinate-free v2 removes global x/y while retaining
height. Start a new run with `walker_v2_smoke_cpu`, `walker_sac_baseline`, or
`walker_td3_baseline`; do not resume across observation versions.

`arena-eval --policies a.zip,b.zip,c.zip` assigns one checkpoint per N-agent
slot. N-agent tournaments rotate competitors through slots and derive Elo from
placement scores. For N-agent replay, pass comma-separated `--replay-opponent`
paths for every slot after `agent_0`. `morphology_search` ranks trials by
tournament Elo (`morphology_search.scoring: tournament_elo`, the default and
only supported mode).

---

## 🛠️ Maintenance Notes

- Historical audit artifact: `docs/exhaustive_repo_review_2026-04-22.md`
- Incremental fixes report: `docs/fixes_2026-04-22.md`
- Open items and future plan: `docs/open_items_todo.md`
- UI review roadmap: `docs/ui_roadmap.md`

### Run Registry

Each train run is registered in `outputs/run_registry.sqlite3` (or the configured
`output.base_dir`). The SQLite registry assigns the run identity, snapshots the
resolved config, records status/metrics events and tuning commands, indexes
artifacts, and links resumed runs to their parent. GUI tuning uses this durable
queue, so commands survive a GUI process restart until the training callback
claims them. The `registry` CLI reports table/status counts, exports every table
as JSON, and safely prunes filtered runs, analysis jobs, or stale artifact index
entries; see the CLI section for examples.

### GUI Analysis

The **Analysis** tab reads the run registry to compare the latest episode reward
and length across runs and chart persisted metric history. Filters narrow the
comparison by free-text run search, experiment, status, algorithm, and
environment; the history metric is selectable and charts up to the ten newest
matching runs. The view also shows recorded artifacts and distinguishes
`best_model` from final/periodic checkpoints. Replay launches prefer
`best_model.zip` and fall back to `final_model.zip`. Arena runs can launch a
background round-robin rating job for league snapshots; at least two snapshots
and `run_metadata.json` are required. Analysis jobs remain visible in the tab
until completion and do not interrupt training.
- Agent workflow notes: `AGENTS.md`

---

## 📝 Changelog

### v0.4.0

**🐛 Bug Fixes:**
- `eval_runner.py` — Arena eval now treats `episode_outcome: timeout` as truncation and uses the same SuperSuit/SB3 adapter as training, so timeout rates and done masks are recorded correctly
- `gui/app.py` — Fixed path traversal vulnerability in `GET /api/outputs`: the `base_dir` query parameter is now ignored; the server always uses its own `_DEFAULT_OUTPUTS_DIR`
- `gui/app.py` — `POST /api/train/stop/<run_id>` now returns **409 Conflict** (not 404) when the run exists but is not in a stoppable state; 404 is still returned for unknown run IDs
- `live_tuning_callback.py` — Removed redundant `self.model.learning_rate = lr` assignment; SB3 uses `lr_schedule` (not the `learning_rate` attribute) to drive optimizer updates, so the stale write had no effect and was misleading
- `gui/static/app.js` — `api()` fetch helper now wraps calls in `try/catch`; network failures and JSON parse errors previously caused silent unhandled promise rejections

### v0.3.0

**✨ New Features:**
- 🖥️ **GPU acceleration** — `training.device` config key added. Defaults to `"auto"`, which selects an NVIDIA GPU when available and falls back to CPU otherwise. Accepted values: `auto`, `cpu`, `cuda`, `cuda:<N>`. All five bundled experiment configs updated.
- `config.py` — `_validate_device()` rejects unrecognised device strings at startup with a clear error message.

### v0.2.0

**🐛 Bug Fixes:**
- `eval_runner.py` — Multi-agent eval no longer crashes (`len(observation_space)` → `vec_env.num_envs`)
- `config.py` — `_ensure_int` now correctly rejects booleans (Python `bool` is a subclass of `int`)
- `app.py` — Path traversal fixed in `GET /api/configs/<name>` and `PUT /api/configs/<name>`

**✨ Improvements:**
- `arena_parallel.py` — Removed unused `energy` field (dead state that inflated obs from 7 → 8)
- `app.py` — GUI auto-creates `configs/experiments/` directory on startup
- `walker_bullet.py` — PyBullet client now disconnects if `__init__` fails partway through
- `rewards.py` — Forward velocity error clamped to prevent unbounded negative rewards
- `curriculum_callback.py` — Support per-level `level_up_thresholds` dict in addition to the global default
- `training_manager.py` — `stop_run()` now actually halts training via a `_StopOnEvent` callback
- `training/__init__.py` — Lazy imports so unit tests don't crash when torch is unavailable
- `multi_seed_runner.py` — Seeds run in parallel by default (`ProcessPoolExecutor`); `--max-workers 1` for sequential
- `tests/test_env_api.py` — Added edge-case tests for health depletion, max-step truncation, and agent-list clearing

### v0.1.0

**🐛 Bug Fixes:**
- `eval_runner.py` — Multi-agent eval now loads trained model (was: random actions)
- `arena_parallel.py` — In-episode growth now updates size each step (was: always zero)
- `arena_parallel.py` — All five PettingZoo dicts now share consistent keys
- `sweep.py` — Helpful error messages for invalid sweep parameter paths

**✨ New Features:**
- 🌐 **Web GUI** (`python -m rl_framework.cli.main gui`) — interactive wizard for experiment setup, real-time dashboard with reward charts, and live parameter tuning during training
- 🔊 Sensor noise injection (`domain_randomization.sensor_noise_std`)
- ⏱️ Action latency simulation (`domain_randomization.action_latency_steps`)
- 📚 Curriculum learning with level-gated overrides
- ⚡ `SubprocVecEnv` parallel rollouts
- 🎲 Multi-seed aggregation (`multi-seed` command)
- 🏆 Self-play league callback (organisms)

---

## 📜 License

See [LICENSE](LICENSE) in repository.
