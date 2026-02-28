# RL Experiment Framework (Locomotion + Organism Arena)

## Concise design decision summary
- **Environment API**: Gymnasium for single-agent locomotion and PettingZoo Parallel API for multi-agent fighting.
- **Physics backend**: PyBullet implemented now for locomotion (`walker_bullet`), with swappable backend architecture via environment registry.
- **RL library**: Stable-Baselines3 PPO for both single-agent and parameter-sharing multi-agent (PettingZoo + SuperSuit vectorization).
- **Config system**: YAML configs loaded via OmegaConf/Hydra ecosystem for readability and easy overrides.
- **Experiment ops**: TensorBoard logs + CSV metrics + SB3 checkpointing + video replay rendering.

## Stack choices and tradeoffs
### Locomotion
1. **Recommended now: Gymnasium + PyBullet + SB3 PPO**
   - Pros: lightweight, stable, easy local iteration.
   - Cons: slower than Brax; less realistic than MuJoCo in some dynamics.
2. **Upgrade path: Gymnasium + MuJoCo + SB3/RLlib**
   - Pros: stronger benchmark parity and richer contact dynamics.
   - Cons: dependency/licensing and setup complexity can be higher by environment.

### Organism growth/fighting
1. **Recommended now: PettingZoo Parallel + SuperSuit + SB3 (shared policy)**
   - Pros: simple path to multi-agent RL with low code overhead.
   - Cons: parameter sharing may be limiting for asymmetric roles.
2. **Upgrade path: PettingZoo + RLlib multi-agent policies + self-play league**
   - Pros: robust multi-policy, scalable distributed training.
   - Cons: heavier operational complexity (Ray cluster + policy management).

## Project structure
```text
src/rl_framework/
  cli/main.py
  configs/experiments/
    robot_walk_basic.yaml
    robot_push_recovery.yaml
    organisms_growth_competition.yaml
    organisms_fight_arena.yaml
  envs/
    base.py
    registry.py
    locomotion/
      dynamics.py
      rewards.py
      terminations.py
      walker_bullet.py
    organisms/
      arena_parallel.py
  evolution/simple_search.py
  training/
    sb3_runner.py
    eval_runner.py
    sweep.py
  utils/
    config.py
    logging_utils.py
tests/
  test_env_api.py
  test_reproducibility.py
Dockerfile
pyproject.toml
requirements.txt
```

## Assumptions
- Python 3.10+.
- CPU-first starter implementation.
- PyBullet locomotion task is intentionally simple and extensible (not benchmark-grade yet).
- Multi-agent starter uses homogeneous agents and shared-policy PPO.

## Commands
```bash
python -m rl_framework.cli.main train --config-name robot_walk_basic
python -m rl_framework.cli.main eval --config-name robot_walk_basic --model-path outputs/robot_walk_basic/seed_42/checkpoints/final_model.zip
python -m rl_framework.cli.main sweep --config-name robot_walk_basic
python -m rl_framework.cli.main render-replay --config-name robot_walk_basic --model-path outputs/robot_walk_basic/seed_42/checkpoints/final_model.zip
```

## 30-minute extension guide
### Add a new environment
1. Create a module under `envs/locomotion` or `envs/organisms` implementing Gymnasium `Env` or PettingZoo `ParallelEnv`.
2. Keep dynamics, reward, and termination in separate files/classes.
3. Register it in `envs/registry.py`.
4. Add a YAML config in `configs/experiments/`.
5. Add tests for `reset/step` API compliance and seeded determinism.

### Add a new morphology
1. Extend morphology keys in your organism config (`base_size`, `health`, `attack_range`, etc.).
2. Read parameters in `arena_parallel.py` and apply to spawn/attack dynamics.
3. Add sweep entries for new morphology keys.
4. Optionally use `evolution/simple_search.py` mutation hook for random search.

## Curriculum + domain randomization hooks
- Curriculum hook point: adjust env config by training stage before each reset.
- Domain randomization hook point: in locomotion reset (`_apply_domain_randomization`) for mass/friction/noise.
- Keep these in config, not hard-coded in simulation logic.

## Performance notes
- Use vectorized environments (`DummyVecEnv` now; switch to `SubprocVecEnv` for CPU scale).
- Enable GPU by installing CUDA build of PyTorch used by SB3.
- Increase rollout parallelism and reduce Python overhead in step loops.
- For high-throughput experimentation, consider Brax (JAX) and RLlib distributed workers.

## Pitfalls
- Version differences across Gymnasium/PettingZoo/SuperSuit/SB3 can change wrapper behavior.
- PettingZoo conversion utilities require homogeneous action/obs spaces.
- Replay rendering currently supports Gymnasium environments only.
