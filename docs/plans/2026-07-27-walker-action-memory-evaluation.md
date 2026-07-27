# Walker Action-Memory Evaluation Plan

## Evidence And Question

The 10M natural-velocity policies achieve real 1.0 m/s displacement, but
stochastic fall rates reach 60%, stance slip is 0.41–0.57 m/s, and the learned
gait uses rapid 5–6 Hz contact chatter. A bounded 0.5 slip cost reduced
seed-21 stochastic slip to 0.290 m/s without meeting the 0.18 m/s gate; five of
six observed falls were recovery failures.

The isolated question is whether the policy lacks short-term command context:
does observing the action applied on the preceding control step improve
stochastic recovery and permit longer swings and strides without a new reward?

## Phase 1 — Matched Seed-21 New Lineages

Train two policies from scratch for 10M steps:

1. `control`: the existing 35-D coordinate-free v2 observation.
2. `previous_action`: append the 10-D action that actually reached PD control
   after latency and action scaling.

The observation shapes are checkpoint-incompatible, so neither arm resumes an
old model. Hold the 0.15→1.0 m/s ramp, PPO settings, reset/domain
randomization, physics, reward economics, and bounded 0.5 stance-slip cost
fixed. Evaluate 20 deterministic and stochastic episodes and render five
stochastic episodes per arm.

## Advancement Gate

The candidate advances only if its stochastic result has all of:

- fall rate ≤30%;
- mean episode length ≥600 control steps;
- stance slip ≤0.18 m/s;
- mean displacement ≥5 m;
- peak torso height <1.0 m;
- 1.5–8.5 alternating touchdowns per 100 steps;
- stride ≥0.10 m and foot clearance ≥0.02 m;
- progress ≥0.05 m per alternating touchdown;
- flight fraction ≤60% and same-foot sequence ≤5; and
- no launch, sliding, or contact-chatter exploit in five renders.

Runs below 10M steps per arm or 20 evaluation episodes are functional smokes
only. They cannot authorize confirmation.

## Phase 2 — Confirmation And Transfer

After explicit visual approval of a passing `previous_action` arm, train the
same candidate from scratch on seeds 22 and 23 for 10M steps. All three seeds
must pass every gate before flat/uneven/obstacle/push transfer evaluation.
Reward, speed, action-scale, or algorithm variants remain deferred.

## Implementation Status

`quality-study --study walker-action-memory` implements the resumable workflow,
gates, rendering, reporting, visual-approval hold, three-seed confirmation,
and conditional transfer. A seed-21 2,400-step functional smoke completed both
arms and remained correctly blocked from Phase 2. Artifacts:
`outputs/quality_studies_walker_action_memory_smoke_20260727`.
