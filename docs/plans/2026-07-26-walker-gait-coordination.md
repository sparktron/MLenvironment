# Walker Gait Coordination Plan

## Objective

Turn the current seed-sensitive forward shuffle into repeatable alternating-foot
locomotion without losing the balance gains already demonstrated by the
stochastic-balance stage.

The velocity-reward study established that scalar reward pressure is no longer
the main bottleneck:

- `velocity_sigma_010` ranked first but only seed 22 passed;
- seed 21 survived 800 steps but stopped at 0.90 m;
- seed 23 reached only 0.45 m;
- increasing the forward-velocity weight reduced mean displacement.

This plan therefore adds gait observability first, then tests narrowly scoped
contact-based structure. PPO/SAC/TD3 comparisons remain out of scope until the
final locomotion gate clears.

## Constraints

- Preserve observation v2 shape and action shape so existing checkpoints remain
  loadable.
- Keep every new reward weight at zero by default.
- Reuse seeds 21–23 balance checkpoints for fair continuation comparisons.
- Do not select variants by reward alone.
- Reject a candidate if any seed fails the behavior gate.

## Phase 1: Measure Gait Structure

Status: **implemented and baselined on 2026-07-26**.

Add episode-level gait telemetry before changing rewards.

### Environment metrics

Track these values in `WalkerBulletEnv`:

- right-only, left-only, double-support, and flight fractions;
- debounced right and left touchdown counts;
- valid alternating touchdown count;
- pelvis forward progress between valid alternating touchdowns;
- mean stance-foot planar slip speed;
- longest same-foot touchdown sequence;
- mean applied-action delta.

Foot world position and velocity should come from PyBullet link state for links
2 and 5. A touchdown is a no-contact to contact transition. Count it as a valid
alternating touchdown only when it follows the opposite foot and passes a
small debounce interval. Telemetry must not affect reward or termination.

### Propagation

- Add the values to `info["walker_episode"]`.
- Log rolling means under `walker/gait_*` in `WalkerMetricsCallback`.
- Extend walker diagnostics with the same episode aggregates.
- Evaluate the existing sigma-0.10 seed-22 and seed-23 checkpoints to establish
  baseline distributions.

### Phase 1 gate

- Existing checkpoints produce finite metrics with no observation/API change.
- Episode support fractions sum to approximately 1.0.
- Touchdown counts and slip measurements agree with rendered rollouts.
- Freeze quantitative gait thresholds before reward ablations begin.

All Phase 1 checks passed without changing the v2 observation shape. Twenty
stochastic episodes from the sigma-0.10 checkpoints produced:

| Seed | Alt. touchdowns / 100 steps | Progress / alt. touchdown | Stance slip | Flight | Longest same-foot sequence |
|---:|---:|---:|---:|---:|---:|
| 21 | 18.68 | 0.0027 m | 0.176 m/s | 17.9% | 3.35 |
| 22 | 20.51 | 0.0050 m | 0.156 m/s | 16.6% | 3.35 |
| 23 | 16.89 | 0.0031 m | 0.163 m/s | 18.0% | 3.20 |

The support fractions sum to one by construction and in the environment
regression test. Seed 22's progress per alternating touchdown distinguishes
its repeatable shuffle from seeds 21 and 23.

## Phase 2: Add Zero-Default Structural Reward Terms

Status: **implemented and smoke-tested on 2026-07-26**.

Introduce two independently configurable terms:

1. `gait_step_progress_weight`
   - Event reward on a valid alternating touchdown.
   - Scale by clipped pelvis progress since the previous valid touchdown.
   - No reward for same-foot tapping or zero-progress oscillation.

2. `stance_slip_penalty_weight`
   - Penalize planar foot speed only while that foot is in contact.
   - Average across contacting feet so double support is not penalized twice.

Add both terms to `WalkerReward.components()` and reward-component telemetry.
Defaults must be `0.0`, preserving all shipped behavior.

Set candidate weights from measured Phase 1 magnitudes so each new term begins
at no more than 10–20% of the mean velocity-reward magnitude. Do not guess
weights before the telemetry baseline is available.

The frozen initial settings are:

- alternating progress: weight `40.0`, per-event progress clipped at `0.03 m`;
- stance slip: penalty weight `0.5`;
- combined: both settings;
- control: both weights remain zero.

Across the three baselines, mean velocity reward was 0.50–0.72 per step.
Progress shaping contributes about 3–6% of that magnitude and stance slip
about 11–16%, keeping each term within the planned shaping budget.

The gait gate is now frozen at at least 15 alternating touchdowns per 100
steps, at least 0.003 m progress per alternating touchdown, at most 0.18 m/s
stance slip, at most 22% flight, and a mean longest same-foot sequence no
greater than 5. Behavior gates remain unchanged.

### Tests

- Reward components still sum exactly to total reward.
- Zero weights reproduce the current reward.
- Same-foot tapping earns no gait-progress reward.
- Alternating touchdown with positive pelvis progress earns a bounded reward.
- Contacting-foot slip is penalized; swing-foot motion is not.
- Reset clears all gait-tracker state.

## Phase 3: Focused Ablation

Status: **completed without promotion on 2026-07-26**.

Use `velocity_sigma: 0.10`, the existing balance checkpoints, and three
continuation candidates:

- alternating-step progress only;
- stance-slip penalty only;
- combined step progress and stance-slip.

A zero-weight continuation control is included to separate structural shaping
from simply training longer. The 24-worker 3k smoke and seed-22 50k rejection
screen completed for all four variants:

| Variant | Steps | Falls | Displacement | Alt. / 100 | Progress / alt. | Slip |
|---|---:|---:|---:|---:|---:|---:|
| combined | 751 | 10% | 1.86 m | 20.00 | 0.0057 m | 0.156 m/s |
| stance slip | 741 | 10% | 1.80 m | 19.68 | 0.0059 m | 0.149 m/s |
| control | 721 | 20% | 1.79 m | 20.92 | 0.0060 m | 0.169 m/s |
| alternating progress | 660 | 30% | 1.72 m | 20.05 | 0.0064 m | 0.162 m/s |

Every variant passed the one-seed rejection gate, although combined's lead
over control is too small for selection. Five rendered combined rollouts all
survived 800 steps; the strongest moved 2.11 m with an upright, small
alternating-foot shuffle and no launch, prone, sliding, or same-foot tapping
exploit. All four variants therefore advance to seeds 21–23 × 150k.

Execution order:

1. 3k-step 24-worker smoke for every candidate.
2. One seed × 50k as a rejection-only screen for reward exploits or destroyed
   balance. A one-seed pass cannot promote a candidate.
3. Seeds 21–23 × 150k for candidates that survive the rejection screen.
4. Twenty deterministic and stochastic episodes per seed.
5. Render the strongest and weakest seed for the leading candidate.

### Behavior gate

Every seed must satisfy:

- stochastic episode length at least 600;
- stochastic fall rate at most 30%;
- forward displacement at least 1.0 m;
- peak torso height below 1.0 m;
- no standing, sliding, launching, or same-foot tapping exploit.

### Gait gate

After Phase 1 freezes thresholds, require:

- a minimum valid alternating-touchdown rate;
- progress per valid alternating touchdown above the baseline floor;
- stance-slip speed below the baseline ceiling;
- no material increase in prolonged flight or same-foot touchdown sequences.

Promotion requires both the behavior and gait gates for all three seeds.

### Evidence-scale result

The seeds 21–23 × 150k matrix completed for all four variants. Slip-only was
the closest to promotion:

| Seed | Steps | Falls | Displacement | Slip | Gate |
|---:|---:|---:|---:|---:|---|
| 21 | 733 | 10% | 1.52 m | 0.190 m/s | fail: slip |
| 22 | 793 | 5% | 2.00 m | 0.140 m/s | pass |
| 23 | 738 | 10% | 1.26 m | 0.178 m/s | pass |

The variant therefore failed the all-seed gate. Stochastic five-rollout
rendering confirmed that seeds 21 and 22 use compact, upright alternating
steps without a visible launch, prone, sliding, or same-foot tapping exploit.
The selected rollouts both survived 800 steps and moved 2.04 m and 2.01 m.
Seed 22 nevertheless fell after 90 and 121 steps in two other samples,
consistent with its stochastic variance and deterministic collapse.

A final seed-21 50k rejection screen tested slip weights 0.75 and 1.0 while
holding every gait threshold fixed:

| Weight | Steps | Falls | Displacement | Slip | Rejection |
|---:|---:|---:|---:|---:|---|
| 0.75 | 682 | 20% | 0.967 m | 0.1794 m/s | displacement below 1.0 m |
| 1.0 | 729 | 20% | 1.093 m | 0.18065 m/s | slip above 0.18 m/s |

Neither weight survived. In particular, the 1.0 result does not justify
loosening or rounding the frozen slip ceiling. No weight advanced to the
three-seed rerun.

## Phase 4: Promotion Validation

Status: **not entered; Phase 3 produced no surviving candidate**.

## Phase 3c: Sustained-Swing Touchdown Rejection Screen

Status: **rejected at seed 21, 50k**.

Calibration used 3,155 individual stochastic completed-swing events from the
fixed-target control. Its p75 values were 50 ms airborne duration and 0.492 mm
clearance, so the candidate used 50 ms and 0.5 mm thresholds, retained target
velocity 0.15 m/s and stance-slip weight 0.5, and rewarded bounded alternating
touchdown progress only when both thresholds were met. All behavior and gait
gates plus a +0.25 m paired stochastic-displacement improvement remained frozen.

The candidate reached 736 stochastic steps, 25% falls, 0.260 m displacement,
0.159 m/s stance slip, 17.01 alternating touchdowns per 100 steps, and 0.00093
m progress per alternating touchdown. It improved over the 0.104 m control by
only +0.157 m, and missed the pre-existing 1 m displacement and 0.003 m
progress floors. It is rejected; do not run seeds 21--23, 300k, or transfer.

## Phase 3d: Touchdown Placement Telemetry

Status: **rejected before training**.

The fixed-target seed-21 control was evaluated with the calibrated 50 ms / 0.5
mm sustained-swing classifier. Across 578 sustained touchdown pairs, mean foot
lead was -1.16 mm (p75 70.99 mm), while mean pelvis progress to the next
touchdown was 7.21 mm (p75 18.42 mm). Their Pearson correlation was -0.054.
Landing lead therefore does not predict the proposed outcome, so a
foot-placement reward would be speculative. Do not run a 50k candidate.

## Phase 3e: Stance-Phase Telemetry

Status: **rejected before training**.

The fixed-target seed-21 control produced 3,985 completed touchdown-to-liftoff
stances across 20 stochastic episodes. Episode-average stance pelvis advance
correlated with displacement at 0.705, but stance duration correlated at -0.492
and mean stance-foot slip at 0.308. The result does not support the required
low-slip, longer-advance push-off mechanism (and advance alone is too close to
the displacement outcome to justify shaping). Do not run a 50k stance reward
candidate.

## Phase 3b: Velocity-Ramp Rejection Screen

Status: **rejected at seed 21, 50k**.

Keep `stance_slip_penalty_weight: 0.5` and the frozen 0.18 m/s slip ceiling.
Before changing reward structure, compare a fixed 0.15 m/s control against a
linear, time-based target ramp from 0.15 to 0.25 m/s. The control and candidate
resume the same seed-matched balance checkpoint and retain all other reward,
action, physics, and optimizer parameters. The candidate must clear every
existing behavior and gait gate and improve mean stochastic displacement by at
least 0.25 m over its paired control. This one-seed 50k screen may reject, but
cannot promote; only a survivor advances to seeds 21--23 at 150k.

The tracker now records same-foot stride length after a completed swing, swing
duration, swing clearance above liftoff height, and qualified-touchdown cadence.
These metrics distinguish longer steps from contact chatter before any
sustained-swing touchdown reward is considered.

The corrected screen (which measures continuation steps rather than the
resumed checkpoint's lifetime steps) rejected the ramp. It improved mean
stochastic displacement from 0.104 m to 0.200 m, only +0.097 m against the
required +0.25 m. It also failed the existing 1 m displacement and 30% fall
limits (666 steps, 35% falls). Slip remained compliant at 0.156 m/s, but stride
length (10.88 mm) and swing duration (41.5 ms) were slightly below the
control's 10.94 mm and 43.4 ms. Do not advance this ramp or add a
swing-gated reward on its basis.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m rl_framework.cli.main quality-study \
  --study walker-velocity-ramp --seeds 21 --study-step-budget 50000 \
  --study-source-dir outputs/walker_stochastic_balance_candidate \
  --study-output-dir outputs/quality_studies_walker_velocity_ramp_20260726
```

Only after a candidate clears Phase 3:

1. Continue three seeds to 300k total staged steps.
2. Evaluate flat, uneven, obstacle, and push-recovery environments.
3. Apply the final robust locomotion gate: at least 760 steps, 3 m
   displacement, at most 10% falls, and peak torso height below 1 m.
4. Render deterministic and stochastic behavior for best and worst seeds.
5. Promote a preset only if all required seed/terrain evaluations pass.

## Expected Files

- `src/rl_framework/envs/locomotion/walker_bullet.py`
- `src/rl_framework/envs/locomotion/rewards.py`
- `src/rl_framework/training/sb3_runner.py`
- `src/rl_framework/training/walker_diagnostics.py`
- `src/rl_framework/training/quality_study.py`
- `src/rl_framework/utils/config.py`
- `src/rl_framework/gui/app.py`
- focused environment, reward, diagnostic, quality-study, config, and GUI tests

## Validation

Run narrow tests first, then:

```bash
source .venv/bin/activate
pytest -q --cov=src/rl_framework --cov-fail-under=60
ruff check src tests scripts
python scripts/check_repo_policy.py
```

For any training-pipeline or reward change, also run a 24-worker smoke,
deterministic/stochastic diagnostics, and visual inspection before scaling.
