# Learning Quality Studies

The `quality-study` CLI runs the Priority 3 evidence matrices and writes an
atomic `state.json`, `report.json`, and `report.md`. Interrupted studies can be
continued with `--resume-incomplete`; the saved identity prevents accidentally
resuming with different seeds or budgets.

## Study Matrix

| Study | Candidates | Evidence |
|---|---|---|
| Walker | legacy reward control, rebalanced flat, flat/uneven/obstacle curricula | zero-action baseline, deterministic and stochastic rollouts, cross-terrain transfer, fall rate, displacement, peak height, and push recovery |
| Arena resources | baseline, scarce/high-cost, abundant/low-cost, large/slow | per-seed common-arena round robins plus native-regime tournaments recording attacks, contacts, damage, food pickups, and depleted-energy steps |
| Arena depth | baseline, center-contested food, body collision damage | starts only after resource measurement; uses common-arena Elo and separate native-regime behavior metrics |
| Algorithms | PPO, SAC, TD3 | equal-step and equal-wall-clock training, followed by zero-action, deterministic, and stochastic evaluation |

Territory mechanics are intentionally not enabled without evidence that food
placement and contact damage leave a strategic gap. Both new arena mechanics
default off (`food_placement: uniform`, `collision_damage: 0.0`).

## Commands And Gates

```bash
# Inspect the planned run count without writing state.
python -m rl_framework.cli.main quality-study --study all \
  --seeds 0,1,2 --dry-run

# Run promotion-scale defaults.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m rl_framework.cli.main quality-study \
  --study all --seeds 0,1,2

# Resume the exact same invocation after interruption.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m rl_framework.cli.main quality-study \
  --study all --seeds 0,1,2 --resume-incomplete
```

The walker readiness gate requires at least three seeds and 300k steps per
candidate. Arena requires at least three seeds and 30k steps. The algorithm
comparison requires at least three seeds, 300k equal-step training, and 300
seconds per equal-wall-clock run. The higher built-in defaults provide margin:
750k walker steps, 30k arena steps, 500k algorithm steps, 900 seconds, and 20
evaluation episodes. A false readiness field means the report is diagnostic
only and must not be used to change a shipped preset.

## Preliminary Run — 2026-07-19

Three-seed preliminary studies validated the complete workflow at deliberately
sub-threshold budgets. They are not release evidence and changed no defaults.

- Walker, 10k steps and three evaluation episodes: `rebalanced_flat` ranked
  first, but 53 of 60 terrain evaluations were untrained-equivalent and the
  readiness gate was false.
- Arena, 5k steps and three episodes per role: every tournament timed out.
  Native measurements successfully distinguished food pickup, collision, and
  energy-depletion regimes, but neither a resource nor strategic-depth variant
  earned promotion.
- Algorithms, 10k equal steps and five equal seconds: PPO averaged 5.45 seconds
  for 10k steps versus 42.17 for SAC and 42.69 for TD3, and led deterministic
  return at equal steps. At equal wall time, SAC had the strongest preliminary
  deterministic return. Fall rates remained high and `comparison_ready` was
  false.

Generated checkpoints and detailed reports live under ignored `outputs/`
directories; the repeatable command, study definitions, and interpretation are
the durable project record.
