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

## Phase 2: Add Zero-Default Structural Reward Terms

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

### Tests

- Reward components still sum exactly to total reward.
- Zero weights reproduce the current reward.
- Same-foot tapping earns no gait-progress reward.
- Alternating touchdown with positive pelvis progress earns a bounded reward.
- Contacting-foot slip is penalized; swing-foot motion is not.
- Reset clears all gait-tracker state.

## Phase 3: Focused Ablation

Use `velocity_sigma: 0.10`, the existing balance checkpoints, and three
continuation candidates:

- alternating-step progress only;
- stance-slip penalty only;
- combined step progress and stance-slip.

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

## Phase 4: Promotion Validation

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

