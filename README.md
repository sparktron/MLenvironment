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
  cli/main.py                          # train / eval / sweep / render-replay commands
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
      walker_bullet.py                 # Gymnasium walker (PyBullet)
    organisms/
      arena_parallel.py                # PettingZoo Parallel two-agent arena
  evolution/simple_search.py           # RandomMorphologySearch mutation hook
  training/
    sb3_runner.py                      # PPO training wrapper
    eval_runner.py                     # Evaluation + CSV metrics
    sweep.py                           # Cartesian-product hyperparameter sweep
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

# Hyperparameter sweep (Cartesian product over configs/experiments/<name>.yaml sweep block)
python -m rl_framework.cli.main sweep --config-name robot_walk_basic

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

## Curriculum + domain randomisation hooks

- **Curriculum**: adjust env config by training stage before each reset. `CurriculumConfig` dataclass in `base.py` is the integration point.
- **Domain randomisation**: mass/friction variation lives in `_apply_domain_randomization()` in `walker_bullet.py`. `DomainRandomizationConfig` in `base.py` also defines `sensor_noise_std` and `action_latency_steps` (see roadmap below).

---

## Performance notes

- Switch `DummyVecEnv` → `SubprocVecEnv` for multi-process CPU rollouts (see roadmap).
- Enable GPU: install the CUDA build of PyTorch used by SB3.
- For high-throughput experimentation, consider Brax (JAX) + RLlib distributed workers.

---

## Pitfalls

- Version differences across Gymnasium / PettingZoo / SuperSuit / SB3 can change wrapper behaviour; pin versions in `pyproject.toml`.
- PettingZoo conversion utilities require homogeneous action/observation spaces.
- Replay rendering currently supports Gymnasium environments only (`render-replay` command).

---

## Bug fixes (applied)

The following bugs were identified and fixed:

### 1. `eval_runner.py` — multi-agent evaluation ignored the trained model
**Symptom**: The `organism_arena_parallel` evaluation branch used random actions regardless of
the `--model-path` argument, making all multi-agent eval results meaningless.
**Fix**: Wrap the environment with SuperSuit (matching the training path), load the PPO
checkpoint, and run deterministic rollouts with `model.predict()`.
**File**: `src/rl_framework/training/eval_runner.py`

### 2. `arena_parallel.py` — episode growth mechanic was always zero
**Symptom**: `_spawn_agent()` computed `growth = episode_growth_scale * self.step_count`, but
`step_count` is reset to 0 before `_spawn_agent` is called. Agents never grew during an episode,
making `organisms_growth_competition.yaml` ineffective.
**Fix**: Remove the growth computation from `_spawn_agent`. Add a `_current_size(agent)` method
that computes live size from `base_size + episode_growth_scale * self.step_count`. Call it each
step inside the movement loop to update `state[agent]["size"]`.
**File**: `src/rl_framework/envs/organisms/arena_parallel.py`

### 3. `arena_parallel.py` — `observations` and `infos` keys violated PettingZoo Parallel API
**Symptom**: After episode end (`self.agents = []`), `observations` used `possible_agents` as
keys while `rewards`/`terminations`/`truncations` used the agents active at step start. The API
requires all five return dicts to share identical keys.
**Fix**: Capture `active_agents = list(self.agents)` at step entry and use it as the key source
for all five dicts. Also guards the winner reward assignment so it only fires if the winner agent
is still in the active set (handles simultaneous-death draw correctly).
**File**: `src/rl_framework/envs/organisms/arena_parallel.py`

### 4. `sweep.py` — `_set_nested` raised an opaque `KeyError` on bad config paths
**Symptom**: A misspelled sweep parameter key (e.g. `"training.learing_rate"`) produced an
unhelpful `KeyError: 'learing_rate'` with no context about which sweep parameter caused it.
**Fix**: Check for the missing intermediate key explicitly and raise a descriptive `KeyError`
naming the full sweep parameter path and the missing segment.
**File**: `src/rl_framework/training/sweep.py`

---

## Proposed enhancements

The following features are ready to implement and build on the existing architecture:

### 1. Apply sensor noise from `DomainRandomizationConfig`
`DomainRandomizationConfig.sensor_noise_std` is defined in `base.py` and exposed in YAML configs
but is never applied. Adding Gaussian noise to the observation in `_get_obs()` (controlled by
`domain_randomization.sensor_noise_std`) would make locomotion policies robust to real-world
sensor imprecision without any API changes.
**Where**: `walker_bullet.py → _get_obs()`, read from `cfg["domain_randomization"]`.

### 2. Action latency simulation
`DomainRandomizationConfig.action_latency_steps` is similarly wired but unused. Implementing a
small FIFO action buffer in `walker_bullet.py` that delays action application by N physics steps
tests whether learned policies degrade gracefully under communication delay — a key sim-to-real
robustness check.
**Where**: `walker_bullet.py → step()`, add `self._action_buffer: deque`.

### 3. Wire `CurriculumConfig` into the training loop
`CurriculumConfig` (level, metrics) exists in `base.py` but is never read. Adding a
`CurriculumCallback` (SB3 `BaseCallback`) that reads rollout metrics (mean reward, episode
length) and bumps `curriculum.level` — which in turn adjusts env parameters like `target_velocity`
or `max_tilt_radians` — enables progressive difficulty without changing any environment code.
**Where**: new `training/curriculum_callback.py`; hook into `sb3_runner.py → model.learn()`.

### 4. `SubprocVecEnv` support for parallel CPU rollouts
`sb3_runner.py` always uses `DummyVecEnv` (single process). Adding a `num_envs` config key and
swapping to `SubprocVecEnv` when `num_envs > 1` gives a near-linear speedup in data collection
on multi-core machines with no algorithm changes.
**Where**: `sb3_runner.py → train()`, check `cfg["training"].get("num_envs", 1)`.

### 5. Multi-seed aggregate runner
Currently each run uses a single seed. Adding a `run_multi_seed(cfg, seeds)` function that
launches `train()` for each seed in a list and then aggregates eval metrics (mean ± std across
seeds) enables statistically sound comparisons between sweep configurations.
**Where**: new `training/multi_seed_runner.py`; expose via a `multi-seed` CLI command.

### 6. Self-play league for organism arena
The current multi-agent setup trains both agents with the same, continuously-updated shared
policy. Adding an SB3 `BaseCallback` that periodically freezes a snapshot of the current policy
and sets it as the fixed opponent for one agent implements a basic self-play league, pushing the
learning agent against a distribution of past selves rather than the current policy.
**Where**: new `training/self_play_callback.py`; requires asymmetric env wrapper for organism
arena where one agent's policy can be frozen independently.

---

## Stack upgrade paths

### Locomotion
1. **Now**: Gymnasium + PyBullet + SB3 PPO — lightweight, easy local iteration.
2. **Next**: Gymnasium + MuJoCo + SB3/RLlib — stronger benchmark parity, richer contact dynamics.

### Organism arena
1. **Now**: PettingZoo Parallel + SuperSuit + SB3 shared-policy PPO — low code overhead.
2. **Next**: PettingZoo + RLlib multi-agent policies + self-play league — multi-policy, distributed training.
