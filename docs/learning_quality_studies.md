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

## Promotion-Scale Run — 2026-07-25

The requested walker and arena evidence budgets completed successfully. The
readiness fields are true because the seed/budget/completeness requirements
were met; they do not mean a candidate learned acceptable behavior.

### Walker

Command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m rl_framework.cli.main quality-study \
  --study walker --seeds 0,1,2 --study-step-budget 300000 \
  --study-eval-episodes 20 \
  --study-output-dir outputs/quality_studies_walker_promotion
```

All 15 training runs and 60 terrain/seed diagnostics completed. Every
diagnostic verdict was `untrained_equivalent`.

| Rank | Candidate | Behavior score (mean ± std) |
|---:|---|---:|
| 1 | `rebalanced_flat` | -0.067 ± 0.140 |
| 2 | `curriculum_flat` | -0.461 ± 0.163 |
| 3 | `legacy_reward_control` | -0.467 ± 0.134 |
| 4 | `curriculum_obstacles` | -0.665 ± 0.125 |
| 5 | `curriculum_uneven` | -0.717 ± 0.079 |

On flat terrain, `rebalanced_flat` averaged 142.8 deterministic steps,
1.69 m displacement, and a 78.3% fall rate across seeds. Stochastic evaluation
averaged 117.9 steps, 1.18 m, and the same fall rate. Zero action survived all
800 steps without falling. Peak torso height stayed near 0.71 m, ruling out the
launch exploit. The best rendered stochastic rollout (seed 2) visibly stepped
forward 3.01 m for 223 steps and then fell; the behavior was neither standing,
sliding, nor launching, but it was not robust locomotion.

The study also exposed a comparison confound that must be removed before the
next walker promotion attempt. The single-environment baselines use
`n_steps: 1024` and receive roughly 293 PPO rollouts at 300k steps. The
24-environment curriculum candidates use a 30k-sample rollout in this study and
receive only ten. The next walker experiment should hold total rollout size and
optimizer-update count comparable across candidates (for example, lower
`n_steps` when using 24 environments), then increase the total budget only if
learning curves are still improving. Algorithm comparisons remain deferred.

### Arena

Command:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m rl_framework.cli.main quality-study \
  --study arena --seeds 0,1,2 --study-step-budget 30000 \
  --study-eval-episodes 20 \
  --study-output-dir outputs/quality_studies_arena_promotion
```

All 18 training runs completed. Common-arena and resource-native tournaments
timed out universally. Depth-native tournaments timed out 99.7% of the time;
body-contact damage caused only trace eliminations. Resource variants recorded
zero attack hits and zero attack damage, and every common-arena Elo remained
1500.

Targeted probes identified two blockers:

- A trained baseline policy versus random never entered the 0.2 attack range
  in 20 deterministic or 20 stochastic episodes. Deterministic actions never
  crossed the 0.5 attack-trigger threshold; stochastic actions crossed it on
  33% of steps but still produced zero hits because the agents did not close.
- Combat is not food-free feasible under the baseline economics. Collision
  separation holds size-1 agents about 0.16 apart, where the 0.2-range linear
  falloff reduces 0.06 nominal damage to about 0.012 per hit. An optimized
  scripted attacker that approached, stopped, and attacked spent its available
  energy after 22 hits, dealt only 0.257 damage against 1.2 health, and timed
  out. In contrast, a binary-falloff probe resolved after 20 hits at step 66,
  and a linear `attack_range: 0.4` / `attack_cost: 0.02` probe resolved after
  35 hits at step 109.

Do not extend arena budgets yet. First run a small mechanics/shaping matrix
that:

1. makes a knockout feasible without relying on many random food pickups;
2. supplies a measured approach/closing-distance learning signal;
3. keeps dense damage feedback until eliminations occur, instead of annealing
   it to zero at 15k steps solely by elapsed timesteps; and
4. requires nonzero attack hits/damage plus a materially lower timeout rate
   before promotion-scale retraining.
