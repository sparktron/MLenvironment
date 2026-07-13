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
