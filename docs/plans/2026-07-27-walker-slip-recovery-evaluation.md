# Walker Slip And Recovery Evaluation Plan

## Evidence

The 10M natural-velocity baseline produces real 1.0 m/s locomotion: all three
deterministic policies move 9.02–11.26 m without height exploits. Reliability
is the blocker: stochastic fall rates are 25–60%, stance slip is 0.41–0.57 m/s,
and stride is 5.7–9.0 cm. The next experiments must improve contact control,
not seek more speed.

## Phase 0 — Mechanism measurements

For each final seed, render five stochastic rollouts and record the 0.5 s
before every fall: per-foot tangential velocity at touchdown and mid-stance,
normal contact force, slip direction, torso pitch/roll, action delta, and
target-vs-achieved velocity. Separate falls caused by touchdown overshoot from
late-stance push-off and recovery failures. No training changes in this phase.

## Phase 1 — One-factor 1M screens (seed 21)

Use the 10M configuration and its final seed-21 model as the common starting
point. Evaluate one factor at a time against a fixed continuation:

1. **Slip-cost sweep:** stance-slip penalty weights 0.25 and 0.5, with a
   calibrated clip from Phase 0.
2. **Contact damping:** reduce PD velocity gain only if telemetry identifies
   touchdown overshoot; compare 0.75 and 0.5 of the baseline gain.
3. **Recovery robustness:** increase reset yaw/position noise only after the
   preceding best candidate holds the slip gate, to test whether failure is
   state sensitivity rather than steady-state traction.

Run 1M steps per candidate, 20 stochastic and deterministic final episodes,
and five stochastic renders. Keep target-ramp, reward economics, action scale,
gamma, and exploration fixed.

## Advancement rules

Advance only if the candidate, relative to the fixed continuation, has all of:

- stochastic fall rate ≤30%;
- stance slip ≤0.18 m/s;
- mean displacement ≥5 m at 1.0 m/s;
- peak torso height <1.0 m; and
- no launch, sliding, or contact-chatter exploit in renders.

## Phase 2 — Confirmation and transfer

The sole Phase-1 winner runs seeds 21–23 × 3M. It must pass every seed, then
receives flat/uneven/obstacle/push transfer evaluation. Do not run competing
reward, speed, action-scale, or algorithm variants before Phase 2 clears.
