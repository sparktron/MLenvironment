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

## Implementation And Result

The resumable `quality-study --study walker-slip-recovery` workflow implements
the plan. Walker steps now expose per-foot planar velocity vectors, normal
contact force, target/achieved velocity, and action delta. The study retains
the raw 0.5 s (30-control-step) window before each fall, classifies the window,
calibrates the slip penalty clip from the pooled contact-speed p90, and writes
five 20 fps stochastic GIFs per checkpoint. A positive
`stance_slip_penalty_clip` caps the reward contribution without clipping the
reported telemetry; zero preserves the previous unbounded behavior.

Phase 0 completed seeds 21–23 × five stochastic episodes. Across 7,144 contact
samples and 2,839 touchdowns, the pooled slip p90 was 1.3398 m/s and touchdown
normal-force p90 was 1,452.4 N. Six falls were recorded: five recovery failures
and one touchdown overshoot. No late-stance push-off fall was identified.
Because touchdown overshoot was not the dominant mechanism, the conditional
PD damping variants were not run.

The fixed seed-21 1M continuation and both slip-cost screens completed:

| Variant | Stochastic falls | Displacement | Stance slip | Alt. touchdowns / 100 | Stride |
|---|---:|---:|---:|---:|---:|
| control | 20% | 11.20 m | 0.500 m/s | 19.24 | 7.47 cm |
| slip cost 0.25 | 15% | 11.28 m | 0.348 m/s | 21.47 | 6.91 cm |
| slip cost 0.5 | 25% | 10.85 m | 0.290 m/s | 23.05 | 6.73 cm |

Both penalties reduced slip while retaining displacement, but neither reached
0.18 m/s. All three variants also retained rapid, short-stride contact chatter;
the 15 Phase-1 renders showed no launch, but confirmed the shuffle/contact
pathology and sampled six falls. No candidate held the slip gate, so reset-noise
recovery testing, three-seed confirmation, and transfer evaluation were not
run. Artifacts:
`outputs/quality_studies_walker_slip_recovery_20260727`.

Because this plan produced no Phase-1 winner, its Phase 2 remains closed. The
implemented next evaluation is the matched, checkpoint-incompatible
[`walker action-memory plan`](2026-07-27-walker-action-memory-evaluation.md);
it tests recovery state awareness without promoting either rejected slip
checkpoint or relaxing the frozen gait gates.
