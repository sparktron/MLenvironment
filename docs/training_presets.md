# Local Training Presets

These presets target this machine's 24 physical CPU cores. All bundled MLP
configs use `device: cpu`; use CUDA only after measuring a larger policy that
benefits from it.

| Goal | Command | Local guidance |
|---|---|---|
| Quick smoke | `python -m rl_framework.cli.main train --config-name walker_smoke_cpu` | 256 timesteps, 2 spawned workers, deterministic mode, NaN checks, and one PyTorch update thread. The July 12 smoke completed at about 1,300 FPS. |
| Reliable overnight walker | `python -m rl_framework.cli.main train --config-name robot_walk_basic` | 24 workers, 2,048 rollout steps, and batch size 512. Enable `evaluation.best_model` when retaining the best policy matters more than the small periodic evaluation cost. |
| High-throughput walker | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m rl_framework.cli.main train --config-name robot_walk_basic` | The validated 24 x 2,048 x 512 preset was stable at roughly 5,605 FPS in the local benchmark. |
| Arena self-play | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m rl_framework.cli.main train --config-name organisms_fight_arena` | Uses CPU, eight native SB3 workers, a frozen-policy league, reward annealing, and curriculum. Keep shared-policy arena runs at one worker. |
| Multi-seed evaluation | `python -m rl_framework.cli.main multi-seed --config-name robot_walk_basic --seeds 0,1,2,3,4 --max-workers 1` | Run sequentially because each walker seed already uses 24 rollout workers. Use a smaller per-run `num_envs` before raising `max-workers`. |

## Walker Curriculum Presets

| Terrain | Config | Notes |
|---|---|---|
| Balance first | `walker_balance_curriculum` | Explicit 28 kg torso physics; starts at 25% action range and unlocks slow then faster motion only after conjunctive survival/displacement gates. |
| Stochastic balance candidate | `walker_stochastic_balance_candidate` | Winning 100k balance stage with `log_std_init: -1.5`, no entropy bonus, and a 15% action range. |
| Natural-velocity follow-up | `walker_low_velocity_candidate` | Rebalanced 200k continuation with a 0.125 survival bonus, 2.0 terminal fall cost, 50% action range, `gamma: 0.995`, and reopened PPO exploration (`log_std_init: -1.0`, `ent_coef: 1e-3`); the quality-study ramp advances its 0.15 m/s starting target toward 1.0 m/s over 3M continuation steps. |
| Flat | `walker_curriculum_flat` | Starts at 0.4 m/s and raises the speed target. |
| Uneven | `walker_curriculum_uneven` | Deterministic low-height blocks begin beyond the spawn area. |
| Obstacles | `walker_curriculum_obstacles` | Three static 10 cm obstacles along the walking path. |
| Push recovery | `robot_push_recovery` | Applies lateral pushes after a warm-up period while increasing target speed. |

The bundled walker reward uses `alive_bonus: 0.25` and
`forward_velocity_weight: 2.0`, so target-speed locomotion is the dominant
positive signal. Do not compare returns from older `alive_bonus: 5.0` runs
directly with the rebalanced presets.

## Learning Quality Studies

Run all three multi-seed Priority 3 matrices with their promotion-scale
defaults:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m rl_framework.cli.main quality-study \
  --study all --seeds 0,1,2
```

The current walker matrix compares `balance_low_std`,
`balance_low_std_low_entropy`, `balance_low_std_conservative`, and
`balance_low_std_slow_scale` with a common 24×128 rollout schedule plus
zero-action, deterministic, stochastic, transfer, launch-height, and
push-recovery diagnostics. Its short screen requires robust stochastic balance
before a candidate can advance to low-velocity training.
`quality-study --study walker-velocity` then resumes the saved balance
checkpoints and compares tighter velocity-reward distributions with a per-seed
locomotion gate. `quality-study --study walker-gait` continues the strongest
velocity checkpoints with a zero-weight control and calibrated alternating-step
progress / stance-slip ablations; it requires both behavior and gait structure
to pass independently for every seed.
`quality-study --study walker-slip-recovery` evaluates the final 10M natural-
velocity checkpoints, pools half-second pre-fall contact telemetry, and runs
only the one-factor branches justified by the measured fall mechanism. Its
conditional three-seed confirmation requires an explicitly visually approved
sole winner. When that study produces no winner,
`quality-study --study walker-action-memory` starts matched new lineages with
and without the opt-in previous applied action. Both arms keep the bounded 0.5
slip cost and the natural-velocity schedule fixed; a 10M seed-21 screen must
pass every frozen locomotion and real-gait gate before visual approval and
seeds 22–23 confirmation. The arena matrix first runs
resource tournaments, then evaluates
feasible combat (`attack_range: 0.4`, `attack_cost: 0.02`) with and without
approach shaping; native-regime measurements are kept separate from
common-baseline tournaments. The algorithm matrix compares PPO, SAC, and TD3
under both equal-step and equal wall-clock budgets, but remains deferred until
walker behavior clears its promotion gate. See
[`learning_quality_studies.md`](learning_quality_studies.md) for gates and
report interpretation.

## Durable Run Records

Training writes immutable run records to `<output.base_dir>/run_registry.sqlite3`.
They contain the config snapshot, status and metrics events, checkpoint/sidecar
artifacts, and resume lineage. GUI tuning requests are durable queue entries
and are applied at the next rollout boundary.

## Resuming Work

Resume an interrupted sweep only after its variants have written
`sweep_summary/state.json`:

```bash
python -m rl_framework.cli.main sweep \
  --config-name robot_push_recovery \
  --resume-incomplete
```

Completed variants have a per-run `completion.json` marker and are skipped only
when the state fingerprint matches the current parameter grid.

Resume a benchmark matrix with its matching state file:

```bash
python scripts/benchmark_device_matrix.py \
  --config-name robot_walk_basic \
  --resume
```

The benchmark writes `outputs/benchmark_device_matrix_state.json` after each
completed regime. Changing config name, config directory, seeds, timesteps, or
the matrix regime list requires a new state file.
