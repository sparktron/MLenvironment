# Learning Quality Studies

The `quality-study` CLI runs the Priority 3 evidence matrices and writes an
atomic `state.json`, `report.json`, and `report.md`. Interrupted studies can be
continued with `--resume-incomplete`; the saved identity prevents accidentally
resuming with different seeds or budgets.

For the chronological teaching narrative—what changed, what worked, what
failed, and why each result advanced or stopped—see the append-only
[`walker_learning_history.md`](walker_learning_history.md). This document
remains the detailed study and command reference.

## Study Matrix

| Study | Candidates | Evidence |
|---|---|---|
| Walker | balance first, conservative balance first, faster velocity ramp; all use explicit 28 kg torso physics, 24 envs × 128 steps, and batch size 256 | zero-action baseline, deterministic and stochastic rollouts, cross-terrain transfer, fall rate, displacement, peak height, and push recovery |
| Arena resources | baseline, scarce/high-cost, abundant/low-cost, large/slow | per-seed common-arena round robins plus native-regime tournaments recording attacks, contacts, damage, food pickups, and depleted-energy steps |
| Arena combat | baseline, feasible combat, feasible combat plus approach shaping | starts only after resource measurement; uses common-arena Elo and separate native-regime behavior metrics |
| Algorithms | PPO, SAC, TD3 | equal-step and equal-wall-clock training, followed by zero-action, deterministic, and stochastic evaluation |

Feasible combat changes only the study candidate (`attack_range: 0.4`,
`attack_cost: 0.02`); the approach candidate additionally sets
`environment.reward.approach_weight: 0.02`. Shipped combat mechanics and
approach shaping remain unchanged until a candidate clears the behavioral gate.

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

The walker evidence gate requires at least three seeds and 300k steps per
candidate. Its behavioral gate requires every terrain/seed evaluation to
produce at least one action mode averaging 760 steps, 3 m displacement, at
most 10% falls, and peak torso height below 1 m. Arena evidence requires at
least three seeds and 30k steps; its native behavior gate requires timeout
below 90%, at least 0.01 attack hits per episode, and at least 0.001 damage per
episode. Reports expose `evidence_ready` separately from `promotion_ready`;
only the latter authorizes changing a shipped preset.

The algorithm comparison requires at least three seeds, 300k equal-step
training, and 300 seconds per equal-wall-clock run. It remains deferred until
the walker behavioral gate passes.

## Balance-First Implementation And Screen — 2026-07-26

Rendering the strongest focused-study checkpoints exposed two distinct
behaviors:

- `curriculum_flat/seed_0` made small unstable steps and then pitched over.
- The longest `curriculum_uneven/seed_0` rollout spent much of its nominal
  604-step survival window prone on a raised terrain tile. Torso contact had
  only been checked against the ground plane, so contact with generated
  terrain did not terminate the episode.

Torso contact now covers the plane and every generated terrain body. The
walker also publishes per-step reward components, tilt, contacts, applied
action magnitude, action scale, and termination reason. Terminal summaries
are logged under `walker/*`, including episode length, displacement, fall rate,
tilt/action means, and each reward component.

The new `walker_balance_curriculum` preset specifies the complete 28 kg
Atlas-class physics block and begins with `sim.action_scale: 0.25`,
`target_velocity: 0`, and stronger balance/fall terms. Curriculum advancement
uses conjunctive behavior gates:

- level 0 → 1: at least 700 mean steps and at most 10% falls;
- level 1 → 2: at least 600 mean steps, 1 m displacement, and at most 30%
  falls.

A one-seed 100k screen compared the base preset, a conservative 15% initial
action range, and a faster velocity-ramp variant. No policy advanced out of
the balance stage, so the base and velocity-ramp checkpoints are intentionally
identical at this budget. Direct 20-episode flat evaluation produced:

| Candidate | Mode | Episode steps | Displacement | Fall rate |
|---|---|---:|---:|---:|
| `balance_first` | deterministic | 800.0 | 0.10 m | 0% |
| `balance_first` | stochastic | 275.1 | -0.02 m | 100% |
| `balance_first_conservative` | deterministic | 800.0 | -0.03 m | 0% |
| `balance_first_conservative` | stochastic | 405.4 | -0.18 m | 85% |

Videos confirm stationary upright balance rather than walking, sliding, or
launching. The short-run scaling target (400 steps, 1.5 m displacement, below
30% falls) was not met, so another three-seed promotion run and PPO/SAC/TD3
comparisons remain deferred.

### Stochastic balance exploration screen

The next one-seed 100k–150k screen isolates the policy exploration settings
that deterministic evaluation bypasses:

| Candidate | Initial log std | Entropy coefficient | Initial action scale | Later action scales |
|---|---:|---:|---:|---|
| `balance_low_std` | -1.0 | 0.005 | 0.25 | 0.50, 1.00 |
| `balance_low_std_low_entropy` | -1.0 | 0.0 | 0.25 | 0.50, 1.00 |
| `balance_low_std_conservative` | -1.5 | 0.0 | 0.15 | 0.50, 1.00 |
| `balance_low_std_slow_scale` | -1.5 | 0.0 | 0.10 | 0.25, 0.50 |

The screen uses stochastic flat-terrain evaluation only and requires at least
600 mean episode steps, at most 30% falls, absolute displacement no greater
than 0.75 m, and peak torso height below 1 m. Passing this balance gate selects
a candidate for a separate low-velocity walking run; it does not satisfy the
final locomotion promotion gate. All four candidates keep curriculum level 0
frozen until 200k steps, so a 100k–150k screen cannot accidentally mix balance
learning with the movement stage.

The seed-13 100k screen selected `balance_low_std_conservative`:

| Candidate | Stochastic steps | Fall rate | Displacement |
|---|---:|---:|---:|
| `balance_low_std` | 493.7 | 80% | +0.23 m |
| `balance_low_std_low_entropy` | 594.6 | 60% | +0.07 m |
| `balance_low_std_conservative` | 800.0 | 0% | -0.25 m |
| `balance_low_std_slow_scale` | 800.0 | 0% | -0.19 m |

A 150k low-velocity continuation of the selected checkpoint cleared the short
walking gate over 20 episodes: 675.3 stochastic steps, 25% falls, 1.12 m
forward displacement, and 0.682 m peak torso height. Video inspection showed a
small alternating-foot shuffle rather than sliding or launching.

The protocol was then replicated from scratch for seeds 21–23 using 100k
balance steps followed by 200k fixed low-velocity steps:

| Seed | Stochastic steps | Fall rate | Displacement |
|---:|---:|---:|---:|
| 21 | 615.3 | 30% | +0.76 m |
| 22 | 778.8 | 5% | +1.60 m |
| 23 | 706.4 | 20% | +0.44 m |
| **Mean** | **700.2** | **18.3%** | **+0.93 m** |

Balance and fall resistance now generalize across seeds, but forward
locomotion does not. Only seed 22 crossed 1 m, so this is partial evidence and
does not unlock the final 3 m locomotion promotion gate or algorithm
comparison.

### Velocity-reward continuation matrix

`quality-study --study walker-velocity` resumes seed-matched checkpoints from
`--study-source-dir` and compares three fixed low-velocity continuations:

| Candidate | Velocity sigma | Forward-velocity weight |
|---|---:|---:|
| `velocity_sigma_015` | 0.15 | 1.0 |
| `velocity_sigma_010` | 0.10 | 1.0 |
| `velocity_sigma_015_weight_15` | 0.15 | 1.5 |

The study is evidence-ready at three seeds and at least 100k continuation
steps. Promotion requires every seed—not only the aggregate mean—to reach 600
stochastic steps, at most 30% falls, at least 1 m forward displacement, and
peak torso height below 1 m. Runs and diagnostics are resumable through the
same `state.json` mechanism as the other quality studies.

The seeds 21–23 study completed with 150k continuation steps per run:

| Candidate | Mean steps | Mean falls | Mean displacement | Per-seed pass |
|---|---:|---:|---:|---|
| `velocity_sigma_010` | 702.7 | 20.0% | +0.90 m | no, yes, no |
| `velocity_sigma_015` | 714.2 | 25.0% | +0.75 m | no, yes, no |
| `velocity_sigma_015_weight_15` | 705.1 | 25.0% | +0.73 m | no, no, no |

For the recommended `velocity_sigma_010` candidate, seed 21 survived all 800
steps but moved only 0.90 m; seed 22 reached 1.35 m with 30% falls; seed 23
reached 0.45 m with 30% falls. Rendering showed seed 22 repeating a small
forward shuffle while seed 23 mainly leaned and crept. No launch or sliding
exploit appeared. The evidence gate is ready, but the per-seed promotion gate
is false.

### Gait-coordination continuation matrix

`quality-study --study walker-gait` continues sigma-0.10 checkpoints with a
zero-weight control, alternating-touchdown progress, stance-slip penalty, and
their combination. Existing observation and action shapes are unchanged, and
both structural weights default to zero.

Phase 1 baseline telemetry over 20 stochastic episodes per seed showed
16.9–20.5 valid alternating touchdowns per 100 steps, 0.0027–0.0050 m progress
per alternating touchdown, 0.156–0.176 m/s stance slip, and 16.6–18.0% flight.
The initial weights (`40.0` progress, `0.5` slip) contribute no more than the
planned 10–20% of the measured velocity-reward magnitude.

The gait gate requires at least 15 alternating touchdowns per 100 steps,
0.003 m progress per alternating touchdown, at most 0.18 m/s stance slip, at
most 22% flight, and no mean same-foot sequence longer than 5, in addition to
the existing survival, fall, displacement, and peak-height gate. Every seed
must pass; one-seed screens cannot promote a candidate.

The 3k-step, 24-worker smoke and seed-22 50k rejection screen completed for
all four variants. Combined ranked first at 751 stochastic steps, 10% falls,
and 1.86 m displacement; slip-only reached 741 steps, 10%, and 1.80 m; the
control reached 721 steps, 20%, and 1.79 m. Alternating-progress-only was
weakest but still cleared the rejection gate at 660 steps, 30%, and 1.72 m.
Five combined renders all survived 800 steps and showed an upright alternating
shuffle without a visible exploit. The differences are too small for
one-seed selection, so all variants advance to the three-seed 150k matrix.

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

## Walker Slip/Recovery Study — 2026-07-27

The conditional workflow is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m rl_framework.cli.main quality-study \
  --study walker-slip-recovery --seeds 21,22,23 \
  --study-step-budget 1000000 --study-eval-episodes 20 \
  --study-source-dir outputs/walker_low_velocity_candidate \
  --study-output-dir outputs/quality_studies_walker_slip_recovery_20260727
```

Phase 0 always measures five stochastic rollouts per source seed and calibrates
the reward clip from pooled contact telemetry. Phase 1 always runs the fixed
seed-21 continuation and slip weights 0.25/0.5. It runs PD damping only when
touchdown overshoot is the dominant measured fall cause, and increased reset
noise only after a candidate holds the 0.18 m/s slip gate. A sole winner must
also pass the fall, displacement, height, and exploit screens; after visual
review, pass `--study-approved-variant <name> --resume-incomplete` to authorize
the seeds 21–23 × 3M confirmation. Transfer runs only after every seed passes.

The completed run observed five recovery failures and one touchdown overshoot,
so damping was skipped. Slip weights 0.25 and 0.5 reduced stochastic slip from
0.500 m/s to 0.348 and 0.290 m/s, but missed the 0.18 m/s gate and retained
19–23 alternating touchdowns per 100 steps with sub-7 cm candidate strides.
No candidate advanced.

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

## Focused Walker Run — 2026-07-26

The equal-rollout follow-up completed all nine training runs and all
20-episode deterministic/stochastic diagnostics:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m rl_framework.cli.main quality-study \
  --study walker --seeds 0,1,2 --study-step-budget 300000 \
  --study-eval-episodes 20 \
  --study-output-dir outputs/quality_studies_walker_focused_20260726
```

Every candidate used 24 environments, 128 steps per environment, and batch
size 256. Evidence readiness passed, but all behavioral pass rates were zero
and `promotion_ready` remained false.

| Rank | Candidate | Behavior score (mean ± std) |
|---:|---|---:|
| 1 | `curriculum_flat` | -0.190 ± 0.079 |
| 2 | `curriculum_uneven` | -0.308 ± 0.189 |
| 3 | `rebalanced_flat` | -0.446 ± 0.253 |

Flat-terrain evaluation averages across the three seeds were:

| Candidate | Mode | Episode steps | Forward displacement | Fall rate |
|---|---|---:|---:|---:|
| `curriculum_flat` | deterministic | 177.6 | 0.59 m | 73.3% |
| `curriculum_flat` | stochastic | 107.8 | 0.28 m | 75.0% |
| `curriculum_uneven` | deterministic | 119.0 | 0.87 m | 86.7% |
| `curriculum_uneven` | stochastic | 94.1 | 0.46 m | 75.0% |
| `rebalanced_flat` | deterministic | 71.9 | 0.74 m | 78.3% |
| `rebalanced_flat` | stochastic | 84.1 | 0.61 m | 71.7% |

Peak torso height remained near 0.70–0.71 m, so self-launching was not the
failure mode. The best individual flat deterministic displacement was 1.59 m
from `curriculum_uneven/seed_0`, but it averaged only 154 steps with a 90% fall
rate. Equalizing PPO rollout/update settings therefore removed the earlier
methodological confound but did not produce robust locomotion. PPO/SAC/TD3
comparisons remain deferred.

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

## Follow-up Implementation — 2026-07-26

The next-study plan is implemented:

- Walker candidates now share `num_envs: 24`, `n_steps: 128`, and
  `batch_size: 256`, eliminating the prior ten-versus-hundreds PPO rollout
  comparison. The matrix is focused on the three plausible candidates and
  reports both evidence and behavioral gates.
- Arena combat candidates now test the mechanics combination proven feasible
  by the scripted probe, with a second candidate adding bounded
  distance-progress shaping.
- `reward_annealing.min_eliminations` keeps dense damage feedback at full
  strength until real non-timeout outcomes appear. The shipped arena preset
  uses a gate of ten outcomes.
- Arena measurements now report timeout rate per variant, and promotion
  requires timeout, attack-hit, and damage thresholds rather than Elo alone.

Run short one-seed studies first in new output directories. If their learning
signals move in the intended direction, repeat the exact 3-seed promotion
commands above; do not resume the July 25 state because the study definitions
have changed.
