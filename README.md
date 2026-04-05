# RL Experiment Framework (Locomotion + Organism Arena)

A modular reinforcement learning framework supporting single-agent locomotion (PyBullet) and multi-agent competition (PettingZoo), trained with Stable-Baselines3 PPO and driven entirely by YAML configuration.

---

## Design decisions

| Concern | Choice | Rationale |
|---|---|---|
| Single-agent API | Gymnasium | Stable, widely supported, simple step/reset contract |
| Multi-agent API | PettingZoo Parallel | Matches SB3 shared-policy path via SuperSuit vectorisation |
| Physics backend | PyBullet | Lightweight, no licence friction; swap to MuJoCo via registry |
| RL algorithm | SB3 PPO | Reliable baseline; algorithm is swappable in `sb3_runner.py` |
| Config system | OmegaConf / Hydra ecosystem | Readable YAML + structured overrides |
| Experiment ops | TensorBoard + CSV + SB3 checkpoints | Low-dependency, works offline |

---

## Project structure

```text
src/rl_framework/
  cli/main.py                          # train / eval / sweep / multi-seed / render-replay
  configs/experiments/
    robot_walk_basic.yaml
    robot_push_recovery.yaml
    organisms_fight_arena.yaml
    organisms_growth_competition.yaml
  envs/
    base.py                            # DomainRandomizationConfig, CurriculumConfig, EnvContext
    registry.py                        # make_env() factory
    locomotion/
      dynamics.py                      # Force/torque application
      rewards.py                       # Composite reward function
      terminations.py                  # Episode termination conditions
      walker_bullet.py                 # Gymnasium walker (PyBullet) + sensor noise + action latency
    organisms/
      arena_parallel.py                # PettingZoo Parallel two-agent arena + dynamic growth
  evolution/simple_search.py           # RandomMorphologySearch mutation hook
  training/
    sb3_runner.py                      # PPO training wrapper (DummyVecEnv / SubprocVecEnv)
    eval_runner.py                     # Evaluation + CSV metrics (single + multi-agent)
    sweep.py                           # Cartesian-product hyperparameter sweep
    multi_seed_runner.py               # Train+eval across N seeds, aggregate mean +/- std
    curriculum_callback.py             # SB3 callback: progressive difficulty via level_params
    self_play_callback.py              # SB3 callback: frozen-policy opponent league
  utils/
    config.py                          # OmegaConf YAML loader
    logging_utils.py                   # Directory creation + CSV append
tests/
  test_env_api.py                      # Gymnasium + PettingZoo API compliance
  test_reproducibility.py              # Seed determinism
Dockerfile
pyproject.toml
requirements.txt
```

---

## Quick start

```bash
# Install
pip install -e .

# Single training run
python -m rl_framework.cli.main train --config-name robot_walk_basic

# Evaluate a checkpoint
python -m rl_framework.cli.main eval \
  --config-name robot_walk_basic \
  --model-path outputs/robot_walk_basic/seed_42/checkpoints/final_model.zip

# Hyperparameter sweep
python -m rl_framework.cli.main sweep --config-name robot_walk_basic

# Multi-seed run (train + eval across 5 seeds, aggregate results)
python -m rl_framework.cli.main multi-seed \
  --config-name robot_walk_basic --seeds 0,1,2,3,4

# Render trained policy to video
python -m rl_framework.cli.main render-replay \
  --config-name robot_walk_basic \
  --model-path outputs/robot_walk_basic/seed_42/checkpoints/final_model.zip

# Docker
docker build -t rl-framework .
docker run --rm rl-framework train --config-name robot_walk_basic
```

---

## Assumptions

- Python 3.10+, CPU-first.
- PyBullet locomotion is intentionally simple; not benchmark-grade.
- Multi-agent uses homogeneous agents and a shared-policy PPO.

---

## Features

### Sensor noise (`domain_randomization.sensor_noise_std`)

Adds Gaussian noise to observations in `_get_obs()`, preventing policies from overfitting
to perfect state information.  Critical for sim-to-real transfer.  Controlled by the
`sensor_noise_std` key under `domain_randomization` in any locomotion YAML config.
Defaults to `0.0` (disabled).

### Action latency (`domain_randomization.action_latency_steps`)

FIFO buffer in `step()` delays action application by N physics steps, testing policy
resilience to real-world communication delay.  Controlled by `action_latency_steps`
under `domain_randomization`.  Defaults to `0` (disabled).

### Curriculum learning (`curriculum` config block)

SB3 `CurriculumCallback` monitors `rollout/ep_rew_mean` and bumps `curriculum.level`
when performance exceeds `level_up_threshold`.  Each level applies dotted-key parameter
overrides to the live environment config (e.g. increasing `target_velocity`, tightening
`max_tilt_radians`).

```yaml
curriculum:
  enabled: true
  level_up_threshold: 150.0
  max_level: 3
  level_params:
    1:
      reward.target_velocity: 1.0
      termination.max_tilt_radians: 0.9
    2:
      reward.target_velocity: 1.5
      termination.max_tilt_radians: 0.7
    3:
      reward.target_velocity: 2.0
      termination.max_tilt_radians: 0.5
```

### Parallel CPU rollouts (`training.num_envs`)

When `training.num_envs > 1`, the training runner uses `SubprocVecEnv` instead of
`DummyVecEnv`, running each environment instance in a separate process for near-linear
speedup in data collection.  Defaults to `1` (single-process).

```yaml
training:
  num_envs: 4    # spawn 4 parallel environment workers
```

### Multi-seed runner (`multi-seed` CLI command)

Trains and evaluates the same config across N seeds, computes mean +/- std of returns,
and writes a summary CSV to `outputs/<experiment>/multi_seed_summary/aggregate.csv`.

```bash
python -m rl_framework.cli.main multi-seed \
  --config-name robot_walk_basic --seeds 0,1,2,3,4
```

Seeds can also be specified in the YAML:

```yaml
multi_seed:
  seeds: [0, 1, 2, 3, 4]
```

### Self-play league (`self_play` config block)

SB3 `SelfPlayCallback` periodically freezes snapshots of the current policy into a
league directory.  `sample_opponent()` returns a randomly selected frozen past policy,
enabling training against a distribution of past selves rather than the current live
policy.  Only activates for `organism_arena_parallel` environments.

```yaml
self_play:
  enabled: true
  snapshot_freq: 5000       # save every 5k timesteps
  max_league_size: 10       # prune oldest beyond 10
```

---

## 30-minute extension guide

### Add a new environment
1. Create a module under `envs/locomotion/` or `envs/organisms/` implementing `gym.Env` or `pettingzoo.ParallelEnv`.
2. Keep dynamics, reward, and termination in separate files/classes.
3. Register the new type in `envs/registry.py`.
4. Add a YAML config in `configs/experiments/`.
5. Add API compliance and seeded-determinism tests.

### Add a new morphology
1. Add keys to the `morphology` block in your organism YAML config.
2. Read them in `arena_parallel.py` (`_spawn_agent` / `_current_size`).
3. Add the new keys as sweep entries.
4. Optionally wire `evolution/simple_search.py` for random mutation search.

---

## Domain randomisation hooks

| Hook | Where | Config key |
|---|---|---|
| Mass scaling | `walker_bullet.py → _apply_domain_randomization()` | `domain_randomization.mass_scale_range` |
| Friction scaling | `walker_bullet.py → _apply_domain_randomization()` | `domain_randomization.friction_range` |
| Sensor noise | `walker_bullet.py → _get_obs()` | `domain_randomization.sensor_noise_std` |
| Action latency | `walker_bullet.py → step()` | `domain_randomization.action_latency_steps` |

---

## Performance notes

- Set `training.num_envs > 1` for multi-process CPU rollouts via `SubprocVecEnv`.
- Enable GPU: install the CUDA build of PyTorch used by SB3.
- For high-throughput experimentation, consider Brax (JAX) + RLlib distributed workers.

---

## Pitfalls

- Version differences across Gymnasium / PettingZoo / SuperSuit / SB3 can change wrapper behaviour; pin versions in `pyproject.toml`.
- PettingZoo conversion utilities require homogeneous action/observation spaces.
- Replay rendering currently supports Gymnasium environments only (`render-replay` command).

---

## Bug fixes (applied)

### 1. `eval_runner.py` — multi-agent evaluation ignored the trained model
The `organism_arena_parallel` evaluation branch used random actions regardless of the
`--model-path` argument.  Fixed: wraps env with SuperSuit and loads the PPO checkpoint.

### 2. `arena_parallel.py` — episode growth mechanic was always zero
`_spawn_agent()` computed `growth = episode_growth_scale * self.step_count`, but step_count
was 0 at spawn.  Fixed: added `_current_size()` method that computes live size each step.

### 3. `arena_parallel.py` — return dicts violated PettingZoo Parallel API
`observations` and `infos` used `possible_agents` as keys while `rewards`/`terminations`/
`truncations` used step-start agents.  Fixed: all five dicts now use `active_agents`.

### 4. `sweep.py` — opaque `KeyError` on bad config paths
`_set_nested` threw bare `KeyError` with no context.  Fixed: descriptive error message
naming the full parameter path and the missing segment.

---

## Stack upgrade paths

### Locomotion
1. **Now**: Gymnasium + PyBullet + SB3 PPO — lightweight, easy local iteration.
2. **Next**: Gymnasium + MuJoCo + SB3/RLlib — stronger benchmark parity, richer contact dynamics.

### Organism arena
1. **Now**: PettingZoo Parallel + SuperSuit + SB3 shared-policy PPO — low code overhead.
2. **Next**: PettingZoo + RLlib multi-agent policies + full self-play league — multi-policy, distributed training.
