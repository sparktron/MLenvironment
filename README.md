# RL Experiment Framework

A modular reinforcement learning framework for single-agent locomotion and multi-agent competition, built on Gymnasium, PettingZoo, PyBullet, and Stable-Baselines3.  Every parameter is controlled by YAML configuration — no code changes needed to run new experiments.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  - [Local (pip)](#local-pip)
  - [Docker](#docker)
- [CLI Reference](#cli-reference)
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

- Python 3.10 or later
- pip 21+ (for PEP 660 editable installs)
- OS: Linux or macOS recommended (PyBullet GUI rendering requires an X display on Linux)

### Core dependencies (installed automatically)

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

### Optional dev dependencies

| Package | Version | Purpose |
|---|---|---|
| pytest | >= 8.0 | Test runner |
| ruff | >= 0.4 | Linting and formatting |

---

## Installation

### Local (pip)

```bash
# Clone the repository
git clone https://github.com/sparktron/MLenvironment.git
cd MLenvironment

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows

# Install in editable mode (includes all core dependencies)
pip install -e .

# Install dev extras (pytest, ruff)
pip install -e ".[dev]"
```

Verify the installation:

```bash
python -c "from rl_framework.cli.main import main; print('OK')"
```

### Docker

```bash
# Build the image
docker build -t rl-framework .

# Run any CLI command (train, eval, sweep, multi-seed, render-replay)
docker run --rm rl-framework train --config-name robot_walk_basic

# Mount a local directory to persist outputs
docker run --rm -v "$(pwd)/outputs:/app/outputs" \
  rl-framework train --config-name robot_walk_basic
```

The Docker image uses `python:3.11-slim` and runs `pip install -e .` inside the container.  The entrypoint is `python -m rl_framework.cli.main`, so all arguments after the image name are forwarded to the CLI.

---

## CLI Reference

All commands follow the pattern:

```
python -m rl_framework.cli.main <command> --config-name <name> [options]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `command` | Yes | — | One of `train`, `eval`, `sweep`, `multi-seed`, `render-replay` |
| `--config-name` | Yes | — | YAML file name without extension (looked up in `--config-dir`) |
| `--config-dir` | No | `src/rl_framework/configs/experiments` | Directory containing YAML configs |
| `--model-path` | For `eval` / `render-replay` | `""` | Path to a saved `.zip` model checkpoint |
| `--seeds` | For `multi-seed` | `""` | Comma-separated seeds (e.g. `0,1,2,3,4`) |

### train

Train a PPO agent from scratch using the given experiment config.

```bash
python -m rl_framework.cli.main train --config-name robot_walk_basic
```

**Output:**
- Checkpoints in `outputs/<experiment_name>/seed_<N>/checkpoints/`
- TensorBoard logs in `outputs/<experiment_name>/seed_<N>/logs/`
- Final model saved as `final_model.zip`
- VecNormalize stats saved as `vecnormalize.pkl` (if observation normalisation is enabled)

**Monitor training with TensorBoard:**

```bash
tensorboard --logdir outputs/
```

### eval

Evaluate a trained checkpoint and write metrics to CSV.

```bash
python -m rl_framework.cli.main eval \
  --config-name robot_walk_basic \
  --model-path outputs/robot_walk_basic/seed_42/checkpoints/final_model.zip
```

**Output:**
- Prints `mean_return` and `std_return` to stdout
- Appends a row to `outputs/<experiment>/seed_<N>/logs/eval_metrics.csv`

Works for both single-agent (locomotion) and multi-agent (organism arena) environments.  Multi-agent eval wraps the environment through SuperSuit and runs the shared PPO policy, matching the training pipeline exactly.

### sweep

Run a Cartesian-product hyperparameter sweep over the `sweep.parameters` block in the YAML config.

```bash
python -m rl_framework.cli.main sweep --config-name robot_walk_basic
```

For a config with:

```yaml
sweep:
  parameters:
    environment.reward.target_velocity: [0.5, 1.0, 1.5]
    environment.reward.torque_penalty_weight: [0.005, 0.01]
```

This launches 3 x 2 = 6 independent training runs, each with a unique experiment name suffix.

### multi-seed

Train and evaluate the same config across multiple seeds, then aggregate results.

```bash
# Specify seeds on the command line
python -m rl_framework.cli.main multi-seed \
  --config-name robot_walk_basic --seeds 0,1,2,3,4

# Or omit --seeds to use the YAML default (or fallback [0,1,2,3,4])
python -m rl_framework.cli.main multi-seed --config-name robot_walk_basic
```

Seeds can also be set in the YAML:

```yaml
multi_seed:
  seeds: [0, 1, 2, 3, 4]
```

**Output:**
- Per-seed training + eval outputs in separate directories
- Aggregate CSV at `outputs/<experiment>/multi_seed_summary/aggregate.csv`
- Prints `mean=X.XXXX  std=X.XXXX` to stdout

### render-replay

Render a trained policy to video (MP4).

```bash
python -m rl_framework.cli.main render-replay \
  --config-name robot_walk_basic \
  --model-path outputs/robot_walk_basic/seed_42/checkpoints/final_model.zip
```

**Output:** Video saved to `outputs/<experiment>/seed_<N>/videos/`

> **Note:** Replay rendering currently supports Gymnasium environments only (locomotion).  Organism arena rendering is not yet supported.

---

## Configuration

### YAML Structure

Every experiment is defined by a single YAML file.  Here is the complete schema with all available keys:

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
    energy: 1.0                      # Base energy

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

### Included Experiments

| Config file | Environment | Description |
|---|---|---|
| `robot_walk_basic.yaml` | `walker_bullet` | Basic bipedal locomotion with mild domain randomisation. Sweeps target velocity and torque penalty. |
| `robot_push_recovery.yaml` | `walker_bullet` | Aggressive domain randomisation (mass 0.8-1.2x, friction 0.7-1.3x) for push-recovery robustness. Sweeps domain randomisation and reset noise. |
| `organisms_fight_arena.yaml` | `organism_arena_parallel` | Two-agent fighting arena, no growth. Sweeps attack range and cooldown. |
| `organisms_growth_competition.yaml` | `organism_arena_parallel` | Two-agent arena with size growth during the episode (`episode_growth_scale: 0.002`). Sweeps base size and damage. |

### Environment Types

#### `walker_bullet` (Gymnasium)

A simple rigid-body walker simulated in PyBullet.  A box-shaped body applies forces (x, y) and torque (z) each step.  The goal is to maintain upright posture while tracking a target forward velocity.

- **Observation space:** `Box(13,)` — position (3), quaternion (4), linear velocity (3), angular velocity (3)
- **Action space:** `Box(3,)` — normalised force-x, force-y, torque-z in [-1, 1]
- **Reward:** alive bonus + velocity tracking - orientation penalty - torque penalty

#### `organism_arena_parallel` (PettingZoo Parallel)

A 2D two-agent arena where agents move, attack, and optionally grow in size over the episode.

- **Observation space:** `Box(8,)` per agent — own position/health/energy + relative opponent position/health + cooldown
- **Action space:** `Box(3,)` per agent — move-x, move-y, attack trigger (fires when > 0.5)
- **Reward:** damage dealt to opponent, minus damage received, plus/minus win/loss bonus

---

## Features

### Domain Randomisation

Mass and friction are randomised at each episode reset, controlled by:

```yaml
domain_randomization:
  mass_scale_range: [0.8, 1.2]     # base_mass *= U(0.8, 1.2)
  friction_range: [0.7, 1.3]       # base_friction *= U(0.7, 1.3)
```

### Sensor Noise

Adds zero-mean Gaussian noise to every element of the observation vector at each step.  Prevents the policy from overfitting to perfect simulator state — a key technique for sim-to-real transfer.

```yaml
domain_randomization:
  sensor_noise_std: 0.01           # standard deviation (0 = disabled)
```

### Action Latency

Introduces a configurable FIFO delay between the agent choosing an action and the action being applied to the physics simulation.  Simulates real-world communication and actuator delays.

```yaml
domain_randomization:
  action_latency_steps: 2          # apply the action from 2 steps ago (0 = disabled)
```

The buffer is pre-filled with zero actions at each `reset()`, so the first N steps apply no-ops while the buffer warms up.

### Curriculum Learning

Progressively increases environment difficulty as the agent improves.  An SB3 callback monitors `rollout/ep_rew_mean` after each rollout and bumps the curriculum level when performance exceeds the threshold.

```yaml
curriculum:
  enabled: true
  level_up_threshold: 150.0
  max_level: 3
  level_params:
    1:
      reward.target_velocity: 1.0
    2:
      reward.target_velocity: 1.5
    3:
      reward.target_velocity: 2.0
```

Level parameters use dotted-key notation to override any value in the environment config at runtime.

### Parallel CPU Rollouts

Run multiple environment instances across separate processes for near-linear data collection speedup.

```yaml
training:
  num_envs: 4                       # uses SubprocVecEnv when > 1
```

When `num_envs` is 1 (the default), `DummyVecEnv` is used (single-process, no overhead).

### Multi-Seed Aggregation

Train and evaluate the same configuration across multiple seeds to produce statistically meaningful results.

```bash
python -m rl_framework.cli.main multi-seed \
  --config-name robot_walk_basic --seeds 0,1,2,3,4
```

**Output format** (`aggregate.csv`):

```
seed,mean_return
0,142.5
1,138.2
2,155.1
3,149.8
4,144.3

aggregate_mean,145.98
aggregate_std,5.76
```

### Self-Play League

For multi-agent organism training, periodically freezes snapshots of the current policy into a league of past opponents.  This breaks the training cycle where two copies of the same live policy co-adapt without meaningful progress.

```yaml
self_play:
  enabled: true
  snapshot_freq: 5000               # freeze a snapshot every 5k steps
  max_league_size: 10               # keep at most 10 past selves
```

The `SelfPlayCallback` exposes `sample_opponent()` which returns a randomly selected frozen PPO model from the league.  Snapshots are saved to `checkpoints/league/` and the oldest are pruned when the league exceeds `max_league_size`.

---

## Project Structure

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
│   │   └── self_play_callback.py       # SB3 callback for self-play league
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

## Testing

```bash
# Run all tests
pytest

# Run a specific test
pytest tests/test_env_api.py::test_walker_env_api -v

# Run with ruff linting
ruff check src/ tests/
```

**Test coverage:**

| Test file | What it verifies |
|---|---|
| `test_env_api.py` | `reset()` and `step()` return correct shapes and types for both walker (Gymnasium) and organism (PettingZoo) environments |
| `test_reproducibility.py` | Two walker envs with the same seed produce identical observations |

---

## Extending the Framework

### Add a New Environment

1. Create a module under `src/rl_framework/envs/locomotion/` (Gymnasium `gym.Env`) or `src/rl_framework/envs/organisms/` (PettingZoo `ParallelEnv`).
2. Separate dynamics, reward, and termination into their own files/classes.
3. Register the environment type in `src/rl_framework/envs/registry.py`:

   ```python
   def make_env(env_type: str, cfg: dict[str, Any]) -> gym.Env | ParallelEnv:
       if env_type == "my_new_env":
           return MyNewEnv(cfg)
       ...
   ```

4. Create a YAML config in `src/rl_framework/configs/experiments/`.
5. Add API compliance and seed-determinism tests in `tests/`.

### Add a New Morphology

1. Add keys to the `morphology` block in your organism YAML config.
2. Read them in `arena_parallel.py` inside `_spawn_agent()` and `_current_size()`.
3. Add the new keys as sweep entries under `sweep.parameters`.
4. Optionally use `evolution/simple_search.py` to mutate the new parameters via random search.

### Swap the RL Algorithm

`sb3_runner.py` currently uses `PPO`.  To switch to another Stable-Baselines3 algorithm (SAC, A2C, TD3, etc.):

1. Change the import and constructor in `sb3_runner.py`.
2. Update any algorithm-specific hyperparameters in the YAML `training` block.
3. No changes needed to environments, evaluation, or sweep infrastructure.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                           CLI (main.py)                           │
│   train │ eval │ sweep │ multi-seed │ render-replay               │
├─────────┴──────┴───────┴────────────┴─────────────────────────────┤
│                                                                   │
│   ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐    │
│   │ sb3_runner   │  │ eval_runner  │  │ multi_seed_runner    │    │
│   │   PPO.learn()│  │  PPO.predict │  │  train() × N seeds   │    │
│   └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘    │
│          │                 │                      │                │
│   ┌──────┴─────────────────┴──────────────────────┘               │
│   │   Callbacks                                                   │
│   │   ├── CheckpointCallback                                      │
│   │   ├── CurriculumCallback                                      │
│   │   └── SelfPlayCallback                                        │
│   └──────┬────────────────────────────────────────                │
│          │                                                        │
│   ┌──────┴───────────────────────────────────────────────────┐    │
│   │                    Environment Layer                      │    │
│   │   registry.py → make_env()                                │    │
│   │   ┌──────────────────┐    ┌───────────────────────────┐   │    │
│   │   │  walker_bullet   │    │ organism_arena_parallel    │   │    │
│   │   │  (Gymnasium)     │    │ (PettingZoo Parallel)      │   │    │
│   │   │  dynamics.py     │    │ battle rules + growth      │   │    │
│   │   │  rewards.py      │    └───────────────────────────┘   │    │
│   │   │  terminations.py │                                    │    │
│   │   └──────────────────┘                                    │    │
│   └──────────────────────────────────────────────────────────┘    │
│                                                                   │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │  Utils: config.py (OmegaConf) │ logging_utils.py (CSV)   │    │
│   └──────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────┘
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

## Performance Tuning

| Technique | How | Expected speedup |
|---|---|---|
| Parallel rollouts | Set `training.num_envs: 4` (or more) | ~3-4x on 4 cores |
| GPU training | Install CUDA PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cu121`) | Varies by network size |
| Reduce logging | Set `verbose: 0` in sb3_runner | Minor |
| Longer rollouts | Increase `training.n_steps` | Better sample efficiency |

For maximum throughput beyond what this framework provides, consider migrating to Brax (JAX-based vectorised physics) or RLlib (distributed workers).

---

## Known Limitations

- **CPU-first**: GPU acceleration requires manually installing the CUDA build of PyTorch.
- **Gymnasium replay only**: `render-replay` does not support PettingZoo organism environments.
- **Shared policy only**: Multi-agent training uses parameter-sharing PPO.  For asymmetric roles, a full multi-policy setup (e.g. RLlib) is needed.
- **Version sensitivity**: Cross-library version changes in Gymnasium / PettingZoo / SuperSuit / SB3 can affect wrapper behaviour.  Pin versions in `pyproject.toml` for production use.
- **Sequential sweeps**: `sweep` runs training configurations sequentially.  For parallel sweeps, use an external orchestrator (e.g. `xargs`, `GNU parallel`, or a job scheduler).

---

## Changelog

### v0.1.0

**Bug fixes:**
- `eval_runner.py`: Multi-agent evaluation now loads and uses the trained PPO model (was using random actions)
- `arena_parallel.py`: Episode growth mechanic (`episode_growth_scale`) now updates agent size each step (was always zero at spawn)
- `arena_parallel.py`: All five return dicts (`observations`, `rewards`, `terminations`, `truncations`, `infos`) now share identical keys per PettingZoo Parallel API
- `sweep.py`: `_set_nested` raises descriptive errors when a sweep parameter path doesn't exist in the config

**New features:**
- Sensor noise injection via `domain_randomization.sensor_noise_std`
- Action latency simulation via `domain_randomization.action_latency_steps`
- Curriculum learning callback with level-gated parameter overrides
- `SubprocVecEnv` parallel rollouts via `training.num_envs`
- Multi-seed aggregate runner (`multi-seed` CLI command)
- Self-play league callback for organism arena training

---

## License

See repository for licence details.
