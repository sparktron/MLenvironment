# 🤖 RL Experiment Framework

**A powerful, modular reinforcement learning framework for training and evaluating single-agent locomotion and multi-agent competitive agents.**

Built on [Gymnasium](https://gymnasium.farama.org/), [PettingZoo](https://pettingzoo.farama.org/), [PyBullet](https://pybullet.org/), and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

✨ **Key features:** YAML-driven experiments • No code changes needed • GPU-ready • Multi-seed aggregation • Self-play league • Domain randomization • Curriculum learning

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
| hydra-core | >= 1.3 | Config composition (OmegaConf backend) |
| numpy | >= 1.26 | Numerical operations |
| pandas | >= 2.1 | Data analysis |
| tensorboard | >= 2.15 | Training visualisation |
| PyYAML | >= 6.0 | YAML parsing |
| flask | >= 3.0 | Web GUI server |

### Optional dev dependencies

| Package | Version | Purpose |
|---|---|---|
| pytest | >= 8.0 | Test runner |
| ruff | >= 0.4 | Linting and formatting |

---

## 📦 Installation

### 💻 Local (pip)

```bash
git clone https://github.com/sparktron/MLenvironment.git && cd MLenvironment
python -m venv .venv && source .venv/bin/activate
pip install -e .                    # Core dependencies
pip install -e ".[dev]"             # + dev tools (pytest, ruff)
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
python -m rl_framework.cli.main gui              # http://127.0.0.1:5000
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

| Flag | Required | Default | Description |
|---|---|---|---|
| `--config-name` | **Yes** | — | YAML config name (without extension) |
| `--config-dir` | No | `src/rl_framework/configs/experiments` | Config directory |
| `--model-path` | Eval/replay only | — | Path to trained model `.zip` |
| `--seeds` | multi-seed only | — | Comma-separated: `0,1,2,3,4` |
| `--max-workers` | multi-seed only | cpu count | Parallel worker processes (pass `1` for sequential) |

### 🏋️ `train` — Train a PPO agent

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

# Force sequential (useful when training already saturates CPUs via num_envs):
python -m rl_framework.cli.main multi-seed \
  --config-name robot_walk_basic --seeds 0,1,2 --max-workers 1
```

Trains and evaluates **the same config** across multiple random seeds for statistical rigor. Seeds run in **parallel by default** using separate processes.

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

**Output:** MP4 videos in `outputs/<experiment>/seed_<N>/videos/`

> ⚠️ Currently supports **Gymnasium environments only** (locomotion). Organism arena rendering coming soon.

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
    body_half_extents: [0.2, 0.1, 0.08]   # Box collision shape half-extents [x, y, z]

  reward:
    alive_bonus: 1.0                 # Reward per step for staying upright
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

  # --- Multi-agent-specific (type: organism_arena_parallel) ---
  sim:
    arena_half_extent: 1.2           # Arena boundary in each direction

  morphology:
    base_size: 1.0                   # Agent body scale at episode start
    episode_growth_scale: 0.0        # Size increase per step (0 = no growth)
    health: 1.2                      # Base health (scaled by size)

  battle_rules:
    damage: 0.06                     # Base damage per hit
    attack_range: 0.2                # Distance threshold for attacks
    cooldown_steps: 3                # Steps between consecutive attacks
    max_steps: 400                   # Episode truncation limit
    win_health_threshold: 0.0        # Health at or below triggers termination

# ─── Training ───────────────────────────────────────────────────
training:
  policy: MlpPolicy                  # SB3 policy class
  total_timesteps: 20000             # Total training steps
  learning_rate: 0.0003              # PPO learning rate
  n_steps: 1024                      # Steps per rollout buffer
  batch_size: 256                    # Minibatch size
  checkpoint_every: 5000             # Save a checkpoint every N steps
  normalize_observations: true       # Wrap env in VecNormalize
  num_envs: 1                        # >1 uses SubprocVecEnv for parallelism

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
| **robot_push_recovery** | `walker_bullet` | Aggressive randomization (mass 0.8-1.2×, friction 0.7-1.3×) for robustness. |
| **organisms_fight_arena** | `organism_arena_parallel` | Two-agent zero-sum combat. Sweeps attack range & cooldown. |
| **organisms_growth_competition** | `organism_arena_parallel` | Two-agent arena with in-episode growth. Sweeps base size & damage. |

### 🎮 Environment Types

#### `walker_bullet` — Single-agent locomotion (Gymnasium)

Rigid-body bipedal walker in PyBullet. Goal: maintain upright posture while tracking target velocity.

| Property | Value |
|---|---|
| **Observation** | `Box(13,)` — position (3) + quaternion (4) + linear vel (3) + angular vel (3) |
| **Action** | `Box(3,)` — normalized force-x, force-y, torque-z ∈ [-1, 1] |
| **Reward** | alive bonus + velocity tracking − orientation penalty − torque penalty |

#### `organism_arena_parallel` — Multi-agent competitive (PettingZoo)

2D two-agent arena with movement, attacks, and optional in-episode growth.

| Property | Value |
|---|---|
| **Observation** | `Box(8,)` — own state (4) + opponent relative (3) + cooldown (1) |
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

### ⚡ Parallel CPU Rollouts

Run multiple envs across processes for **~3-4× speedup** on 4 cores.

```yaml
training:
  num_envs: 4                       # SubprocVecEnv (1 = DummyVecEnv, no overhead)
```

### 📈 Multi-Seed Aggregation

Train across multiple seeds for **statistically rigorous** results. Seeds run **in parallel by default** (one subprocess per seed, up to `cpu_count`). Pass `--max-workers 1` for sequential. See [`multi-seed` command](#-multi-seed--train-across-multiple-seeds-statistical-significance) above.

### 🏆 Self-Play League

Breaks co-adaptation by maintaining a league of frozen past opponents. The live policy trains against random league members, not copies of itself.

```yaml
self_play:
  enabled: true
  snapshot_freq: 5000               # Freeze a snapshot every 5k steps
  max_league_size: 10               # Keep ≤ 10 past versions
```

Snapshots saved to `checkpoints/league/`. Oldest pruned automatically.

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
```

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

Change `sb3_runner.py` (currently uses `PPO`):

1. Swap import: `from stable_baselines3 import SAC`
2. Update YAML `training` hyperparameters
3. No changes to envs, eval, or sweep 🎉

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
| RL algorithm | SB3 PPO | Reliable baseline; algorithm is swappable in `sb3_runner.py` |
| Config system | OmegaConf / Hydra | Readable YAML + structured overrides |
| Experiment ops | TensorBoard + CSV + SB3 checkpoints | Low-dependency, works offline |

---

## ⚡ Performance Tuning

| Technique | Action | Expected Gain |
|---|---|---|
| **Parallel rollouts** | Set `training.num_envs: 4+` | ~3-4× on 4 cores |
| **GPU training** | Install CUDA PyTorch | Varies (network-dependent) |
| **Longer rollouts** | Increase `training.n_steps` | Better sample efficiency |
| **Reduce logging** | Set `verbose: 0` | Minor |

**For massive scale:** Consider [Brax](https://github.com/google/brax) (JAX vectorization) or [RLlib](https://www.ray.io/rllib) (distributed).

---

## ⚠️ Known Limitations

| Limitation | Workaround |
|---|---|
| **CPU-first** | Manually install CUDA PyTorch for GPU acceleration |
| **Gymnasium replay only** | `render-replay` doesn't support PettingZoo. Multi-agent rendering coming soon. |
| **Shared policy only** | Multi-agent uses parameter-sharing PPO. Use [RLlib](https://www.ray.io/rllib) for multi-policy setups. |
| **Version sensitivity** | Pin versions in `pyproject.toml` for production deployments |
| **Sequential sweeps** | Use `xargs` / `GNU parallel` / job scheduler for parallel hyperparameter sweeps |

---

## 📝 Changelog

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
