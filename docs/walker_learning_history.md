# Walker Learning Experiment History

This is the canonical, append-only account of how the walker training process
evolves. It records what was changed, what worked, what failed, and what each
result justified doing next. The purpose is not only reproducibility; it is to
preserve enough reasoning to teach the process after robust walking is solved.

Detailed commands, study definitions, and complete tables remain in
[`learning_quality_studies.md`](learning_quality_studies.md). Phase-specific
design details live in the
[`walker gait-coordination plan`](plans/2026-07-26-walker-gait-coordination.md).
Generated `state.json`, `report.json`, checkpoints, TensorBoard logs, and videos
live under the ignored `outputs/` tree.

## Definition Of Success

A preset is not considered solved because one rollout looks good or an
aggregate mean is high. The final gate is frozen:

- at least 760 mean episode steps;
- at most 10% falls;
- at least 3 m forward displacement;
- peak torso height below 1 m;
- legitimate locomotion rather than standing, sliding, launching, going prone,
  or exploiting contact telemetry;
- all required seeds and flat, uneven, obstacle, and push-recovery evaluations
  pass.

Evidence readiness and behavioral promotion are separate. Completing the
requested seeds and budgets makes a study evidence-ready; only passing every
behavioral gate makes it promotion-ready.

## Experimental Method That Emerged

1. Establish a zero-action baseline and evaluate both deterministic and
   stochastic policy modes.
2. Render representative rollouts. Metrics alone cannot distinguish walking
   from standing, sliding, launching, or lying on terrain.
3. Instrument the suspected bottleneck before shaping it.
4. Change one causal factor at a time and keep unrelated physics, seeds,
   rollout size, optimizer schedule, and gates fixed.
5. Use a short single-seed run only as a rejection screen. It can stop a bad
   idea but cannot promote one.
6. Replicate a survivor across at least three fixed seeds.
7. Reject the candidate if any seed fails; do not hide failures in the mean.
8. Extend training and run transfer environments only after the fixed
   multi-seed gate passes.
9. Change a shipped preset only after the final promotion gate passes.

## Chronological Decision Ledger

### 2026-07-19 — Validate the study machinery

**Question:** Can the framework run resumable, multi-seed walker comparisons
and produce diagnostics suitable for promotion decisions?

**Change:** Added the `quality-study` workflow with saved identity, atomic
state, deterministic/stochastic evaluation, a zero-action baseline, transfer
environments, and separate evidence and promotion readiness.

**Observed result:** The deliberately small 10k preliminary study completed,
but 53 of 60 terrain evaluations were `untrained_equivalent`.

**What worked:** The experimental machinery produced comparable, resumable
evidence and correctly refused to promote an under-budget result.

**What failed:** No candidate learned meaningful behavior at the smoke budget.

**Decision:** Keep the workflow, not the policies. Run promotion-scale evidence
before changing defaults.

### 2026-07-25 — First promotion-scale walker matrix

**Question:** Do the existing reward and terrain curricula produce robust
walking at 300k steps?

**Change:** Trained five candidates over seeds 0–2 and evaluated all
terrain/seed combinations with 20 deterministic and stochastic episodes.

**Observed result:** All 15 runs and 60 diagnostics completed, but every
diagnostic was `untrained_equivalent`. The leading `rebalanced_flat` candidate
averaged 117.9 stochastic steps, 78.3% falls, and 1.18 m displacement. Zero
action survived 800 steps. One rendered seed moved 3.01 m in 223 steps and then
fell.

**What worked:** Forward stepping was physically possible, and peak height near
0.71 m ruled out a launch exploit.

**What failed:** Movement was far less stable than doing nothing. The study
also compared candidates with very different PPO rollout/update counts:
single-env runs received hundreds of updates while 24-env candidates received
about ten.

**Decision:** Do not increase the budget or compare algorithms yet. First
equalize rollout size and optimizer exposure.

**Lesson:** A completed large run can reveal a methodological failure rather
than a policy winner. Fix the comparison before interpreting rankings.

### 2026-07-26 — Equalize PPO exposure

**Question:** Was the poor result mainly caused by unequal PPO rollout and
update schedules?

**Change:** Restricted the focused matrix to plausible candidates and gave all
of them 24 environments, 128 steps per environment, and batch size 256.

**Observed result:** The comparison became fair, but no candidate passed.
`curriculum_flat` led with 177.6 deterministic steps, 73.3% falls, and 0.59 m
displacement; its stochastic policy averaged 107.8 steps and 75% falls.

**What worked:** The confound was removed, making later conclusions credible.

**What failed:** Equalized PPO exposure alone did not produce robust
locomotion.

**Decision:** Stop treating more training of the same unstable movement policy
as the next step. Learn balance first.

### 2026-07-26 — Close a termination loophole and isolate balance

**Question:** Can the walker first learn a stable upright policy under the
explicit Atlas-class physics?

**Change:** Rendering exposed a policy lying on raised terrain without
terminating. Torso contact was extended from the ground plane to all terrain
bodies. The balance-first preset then used the explicit 28 kg torso, a
restricted action scale, zero target velocity, stronger balance/fall terms,
and behavior-gated curriculum advancement. Reward, contact, action, tilt, and
termination telemetry were added.

**Observed result:** Deterministic policies stood for 800 steps, but stochastic
evaluation remained poor: the base candidate averaged 275 steps with 100%
falls, and the conservative candidate averaged 405 steps with 85% falls.

**What worked:** The false survival exploit was removed. The environment could
now measure real upright balance, and deterministic standing became reliable.

**What failed:** Deterministic evaluation hid severe stochastic instability.

**Decision:** Keep the environment fix and telemetry. Investigate exploration
variance and action range before adding forward motion.

**Lesson:** Always render, and always evaluate both action modes. A stable
deterministic mean does not imply a usable stochastic PPO policy.

### 2026-07-26 — Exploration-noise balance screen

**Question:** Is excessive stochastic action variance preventing the learned
balance policy from surviving?

**Change:** Compared initial PPO log standard deviation, entropy coefficient,
and restricted initial action scale while freezing curriculum progression.

**Observed result:** The `balance_low_std_conservative` combination
(`log_std_init: -1.5`, `ent_coef: 0`, initial action scale 0.15) reached 800
stochastic steps with 0% falls. The higher-variance candidates remained
unstable.

**What worked:** Lower initial exploration variance plus a conservative action
range solved the balance stage without changing physics.

**What failed:** Balance did not itself create locomotion; the winning policy
moved -0.25 m.

**Decision:** Promote this checkpoint only to a separate low-velocity walking
stage, not to a shipped locomotion preset.

**Lesson:** Stage incompatible objectives. First make exploration safe enough
to retain balance, then introduce movement.

### 2026-07-26 — Low-velocity continuation and independent replication

**Question:** Can stable balance be converted into legitimate forward motion
with a small velocity target?

**Change:** Continued the balance winner with a 0.15 m/s target. After a
successful seed-13 screen, repeated the complete balance-plus-motion process
independently for seeds 21–23.

**Observed result:** Seed 13 reached 675 stochastic steps, 25% falls, and
1.12 m. The independent seeds averaged 700 steps, 18.3% falls, and 0.93 m:
seed 21 moved 0.76 m, seed 22 moved 1.60 m, and seed 23 moved 0.44 m. Rendering
showed a real small alternating-foot shuffle, not sliding or launching.

**What worked:** The staged method converted stable standing into legitimate
forward motion, and balance generalized across seeds.

**What failed:** Displacement was strongly seed-sensitive.

**Decision:** Preserve the staged balance method. Test narrower velocity
reward discrimination using seed-matched continuation checkpoints.

### 2026-07-26 — Velocity-reward continuation matrix

**Question:** Will a narrower velocity reward or stronger forward reward make
the shuffle reproduce across seeds?

**Change:** Compared velocity sigmas 0.15 and 0.10, plus sigma 0.15 with the
forward-velocity weight increased from 1.0 to 1.5.

**Observed result:** Sigma 0.10 ranked first at 703 mean steps, 20% falls, and
0.90 m, but only seed 22 passed. The stronger forward weight reduced mean
displacement to 0.73 m.

**What worked:** Narrowing sigma to 0.10 was directionally better and provided
the source checkpoints for the gait study.

**What failed:** More scalar forward-reward weight made locomotion worse, and
no velocity variant generalized.

**Decision:** Do not keep increasing forward-reward weight. Instrument gait
structure and test contact-based shaping.

### 2026-07-26 — Gait telemetry and structural shaping

**Question:** Is the policy failing because its compact shuffle lacks
repeatable alternating-foot structure?

**Change:** Added support fractions, debounced touchdowns, alternating
touchdowns, pelvis progress per alternating touchdown, stance-foot slip,
flight, same-foot sequence, and action-delta telemetry. Added zero-default
alternating-progress and stance-slip reward terms. Frozen gait gates included
at least 15 alternating touchdowns per 100 steps, at least 0.003 m progress per
alternating touchdown, at most 0.18 m/s stance slip, at most 22% flight, and no
mean same-foot sequence above 5.

**Observed result:** All four variants survived the seed-22 50k rejection
screen, but the apparent differences were too small for selection. In the
three-seed 150k matrix, slip-only was closest:

| Seed | Steps | Falls | Displacement | Stance slip | Result |
|---:|---:|---:|---:|---:|---|
| 21 | 733 | 10% | 1.52 m | 0.190 m/s | failed slip |
| 22 | 793 | 5% | 2.00 m | 0.140 m/s | passed |
| 23 | 738 | 10% | 1.26 m | 0.178 m/s | passed |

Five-rollout stochastic rendering of seeds 21 and 22 showed upright, compact
alternating motion without a visible exploit. The selected rollouts survived
800 steps and moved 2.04 m and 2.01 m. Seed 22 also fell after 90 and 121 steps
in two samples, while its deterministic policy averaged only 361 steps with
70% falls.

**What worked:** Gait telemetry made the compact shuffle measurable. Slip
weight 0.5 produced the closest multi-seed candidate, and two seeds passed the
complete intermediate gate.

**What failed:** No structural variant passed every seed. Alternating-progress
shaping did not create long strides, and the strongest stochastic seed had a
collapsed deterministic mean.

**Decision:** Do not promote or start transfer evaluation. Use seed 21 as the
rejection screen for slightly stronger slip penalties while holding the
0.18 m/s ceiling fixed.

### 2026-07-26 — Stronger stance-slip rejection screen

**Question:** Can a modestly stronger slip penalty bring seed 21 under the
frozen slip ceiling without losing forward progress?

**Change:** Tested weights 0.75 and 1.0 for 50k continuation steps on seed 21.
No gate or threshold changed after observing the results.

| Weight | Steps | Falls | Displacement | Stance slip | Result |
|---:|---:|---:|---:|---:|---|
| 0.75 | 682 | 20% | 0.967 m | 0.1794 m/s | failed displacement |
| 1.0 | 729 | 20% | 1.093 m | 0.18065 m/s | failed slip |

**What worked:** The rejection screen cheaply eliminated both candidates
before a three-seed run. Weight 0.75 reduced mean slip below the ceiling.

**What failed:** The improvement came with insufficient displacement. Weight
1.0 missed the slip ceiling even though it was close; rounding or loosening
the frozen threshold would have biased selection after the fact.

**Decision:** Reject both. Do not run the conditional seeds 21–23 × 150k,
300k extension, or transfer suite. Do not test still larger slip penalties
without a new measured mechanism.

**Lesson:** A near miss is still a miss when the threshold was frozen in
advance. Rejection screens save compute only when their failures are honored.

## Current State Of Knowledge

### Changes supported by evidence

- Terrain-aware torso contact prevents false survival and must remain.
- Equal rollout/update exposure is required for fair PPO comparisons.
- Deterministic and stochastic evaluation plus rendering are both necessary.
- Low initial PPO variance and a conservative action range produce robust
  balance.
- Staging balance before velocity produces legitimate forward shuffling.
- Velocity sigma 0.10 is better than the tested 0.15 alternatives.
- Stance-slip weight 0.5 is the closest tested structural candidate, but it is
  not promotion-worthy.
- Fixed per-seed gates and rejection screens prevent attractive outliers from
  becoming defaults.

### Approaches rejected by current evidence

- More training of the original unstable curricula without fixing comparison
  or stability problems.
- Selecting policies from deterministic evaluation alone.
- Treating one good seed, one long rollout, or an aggregate mean as promotion.
- Increasing forward-velocity weight to 1.5.
- Increasing stance-slip weight to 0.75 or 1.0.
- Loosening the 0.18 m/s slip threshold after a near miss.
- Running transfer or algorithm comparisons before multi-seed locomotion
  passes.

### Next hypothesis, not yet evidence

At 60 Hz, 800 steps last about 13.3 seconds. The current 0.15 m/s target
corresponds to about 2.0 m over a full episode, matching the observed ceiling;
3 m requires approximately 0.225 m/s. The next isolated test should therefore
keep slip weight 0.5 and the frozen gait gates while ramping the target velocity
from 0.15 toward 0.225–0.25 m/s. Before adding more gait reward, telemetry
should distinguish true strides from rapid contact chatter by measuring swing
duration, foot clearance, cadence, and stride length.

This is a hypothesis and planned diagnostic direction, not a successful
change. It belongs in the supported list only after passing the same rejection,
multi-seed, and transfer process.

## 2026-07-26 — Velocity Ramp And Qualified-Swing Telemetry

**Question:** Can a higher staged velocity target improve displacement without
breaking the frozen slip and gait limits, and does any gain reflect longer
steps rather than contact chatter?

**Change:** Added reward-agnostic stride length, swing-duration, swing-clearance,
qualified-touchdown, and cadence telemetry. Added a seed-matched rejection
study that holds the 0.5 stance-slip penalty and every other training setting
fixed, comparing 0.15 m/s with a linear 0.15--0.25 m/s ramp.

**Decision rule:** Reject the ramp if it misses any existing behavior or gait
gate, including the 0.18 m/s slip ceiling, or if it gains less than 0.25 m of
mean stochastic displacement over control. The planned 50k seed-21 run is a
screen only. A survivor, and only a survivor, proceeds to seeds 21--23 at
150k; sustained-swing touchdown progress is deferred unless telemetry shows
faster shuffling rather than longer strides.

**Observed result:** The corrected 50k screen rejected the candidate. At its
0.25 m/s final target, stochastic displacement rose from 0.104 m (control) to
0.200 m, a +0.097 m change below the frozen +0.25 m materiality rule. The ramp
also averaged 666 steps and 35% falls, missing the 1 m displacement and 30%
fall gates. Slip was compliant (0.156 m/s), but stride length (10.88 mm), swing
duration (41.5 ms), and clearance (0.66 mm) were not above control (10.94 mm,
43.4 ms, and 0.70 mm).

**Decision:** Reject the ramp. Do not run seeds 21--23 × 150k, 300k, or
transfer testing; do not add the conditional swing-gated touchdown reward.

## 2026-07-26 — Sustained-Swing Touchdown Progress Screen

**Question:** Does rewarding forward progress only at an alternating touchdown
after a sufficiently long, clear airborne swing turn the compact shuffle into
longer forward stepping?

**Calibration and change:** Collected 3,155 individual stochastic touchdown
events from the seed-21 fixed-target control. The event-level p75 thresholds
were 50 ms duration and 0.492 mm clearance; the candidate used 50 ms and 0.5
mm. It retained target velocity 0.15 m/s, stance-slip weight 0.5, physics,
optimizer, and all gates. The new bounded progress term fired only when both
thresholds and ordered touchdown alternation were satisfied.

**Observed result:** At 50k continuation steps, stochastic displacement was
0.260 m versus 0.104 m for the paired control (+0.157 m). The candidate had
736 steps, 25% falls, 0.159 m/s stance slip, 17.01 alternating touchdowns per
100 steps, and 0.00093 m progress per alternating touchdown. It missed the
frozen +0.25 m paired improvement, 1 m displacement, and 0.003 m progress
floors.

**Decision:** Reject. Do not run seeds 21--23, 300k, transfer, or algorithm
comparisons. Artifacts: `outputs/quality_studies_walker_swing_touchdown_20260726`.

## 2026-07-26 — Touchdown Placement Telemetry

**Question:** Does a sustained touchdown landing farther ahead of the pelvis
predict progress before the next touchdown, enough to justify a placement term?

**Observed result:** The seed-21 fixed-target control produced 578 calibrated
sustained-event pairs. Foot lead and pelvis progress to the next touchdown had
Pearson correlation -0.054 (mean lead -1.16 mm; mean next-touchdown progress
7.21 mm).

**Decision:** Reject the mechanism before training. A foot-placement reward is
not justified; no 50k candidate was run.

## 2026-07-26 — Stance-Phase Telemetry

**Question:** Do longer, low-slip touchdown-to-liftoff stances with greater
pelvis advance predict displacement strongly enough to justify a push-off term?

**Observed result:** Across 3,985 completed stances in 20 seed-21 fixed-target
control episodes, episode-average stance advance correlated with displacement
at 0.705. Duration correlated at -0.492 and mean stance-foot slip at 0.308;
the low-slip, longer-advance condition was not met.

**Decision:** Reject before training. Stance advance alone is too close to the
outcome to serve as a new mechanism, and the remaining predictors contradict
the proposed reward direction. No 50k candidate was run.

## 2026-07-26 — Correction: The Frozen Gait Gate Certified Contact Chatter

This is a dated correction to the gait thresholds frozen earlier on 2026-07-26,
filed under the rule that a wrong historical value is corrected by appending
rather than by rewriting. No earlier observation is retracted.

**What was wrong:** the gait gate was calibrated as "at least as good as the
measured baseline" (at least 15 alternating touchdowns per 100 steps, at most
22% flight). The baseline it was calibrated from was itself the pathology. One
gait cycle produces two alternating touchdowns, so at 60 Hz control the rate is
`3.33 × cycle_frequency_Hz`. The baseline's 17–20 touchdowns per 100 steps is
therefore a **5–6 Hz gait cycle** — roughly five times human cadence — paired
with 10.9 mm strides and 0.7 mm of foot clearance. That is contact chatter, not
stepping. A floor of 15 demanded chatter, and a genuine 1 Hz gait produces about
3.3 touchdowns per 100 steps, so real walking would have **failed** the gate by
roughly a factor of five.

**What this does and does not invalidate.** Every rejection in the preceding
five entries was decided on displacement, fall rate, stance slip, the paired
+0.25 m improvement rule, or a correlation analysis — **none was decided by the
cadence floor**. Those decisions stand, and the placement (r = −0.054) and
stance-phase (r = 0.705 / −0.492 / 0.308) analyses are independent of the gate
entirely. The defect is that the gate never *discriminated*: every candidate
chattered, so every candidate satisfied it. It reported that gait structure was
acceptable while stride length sat at 11 mm, which pointed five consecutive
experiments at displacement and slip shaping instead of at the stride collapse.
The variant ranking made this worse by scoring `+ cadence / 100`, so a faster
shuffle out-ranked longer stepping by construction.

**Supporting measurement (open loop, no learning):** a scripted sinusoidal hip
and knee gait was commanded at this study's own `action_scale: 0.25` and
`position_gain: 0.1`.

| action_scale | position_gain | foot clearance |
|---:|---:|---:|
| 0.25 | 0.1 | 33.7 mm |
| 0.5 | 0.3 | 55.2 mm |
| 1.0 | 1.0 | 170.9 mm |

All rollouts fell within about a second, as an open-loop gait on a 65.9 kg
biped should — this measures actuator capability, not balance. The conclusion is
narrow and important: the actuator lifts a foot **33.7 mm at the exact settings
where the learned policy achieves 0.7 mm**, a factor of 48. Control authority
is not the constraint. Flight fractions of 0.17–0.56 were also observed, so the
former 22% ceiling was below what a commanded gait naturally produces.

**Change:** `WALKER_GAIT_GATE` in `quality_study.py` now requires a cadence
*band* of 1.5–8.5 alternating touchdowns per 100 steps (cycle frequency
0.45–2.5 Hz), a mean stride of at least 0.10 m, mean foot clearance of at least
0.02 m, at least 0.05 m of progress per alternating touchdown, and at most 60%
flight. The ranking score drops its cadence term in favour of stride length and
clearance, which are monotonic in walking quality. `max_stance_slip_speed`
remains **unchanged at 0.18 m/s** because it was measured in the 0.1 m/s regime
and no faster reference gait has been measured; it must be recalibrated from
measurement, not raised by guess, before it gates a faster candidate.

**Decision:** treat gait *structure* as the open problem and stride length as
the primary quantity. Do not resume reward-structure ablations under the old
thresholds, and do not read the previous five entries as evidence that gait
structure was adequate.

**Lesson:** a threshold calibrated as "no worse than what we currently do" can
only ever certify the current behavior. Freeze thresholds against the mechanism
being sought — here, gait mechanics and demonstrated actuator capability — not
against the run in front of you. A gate that every candidate passes is not a
gate; it is a measurement you have stopped taking.

## 2026-07-26 — Measured Throughput And Budget Scale

**Question:** are the experiment budgets used above large enough to distinguish
candidates from seed noise?

**Observed result:** the real training pipeline sustains **5,970 fps** on this
machine (60k steps in 10 s, `walker_low_velocity_candidate`, 24 envs). The
50k rejection screens therefore cost about **8 seconds**, and a three-seed 150k
matrix costs about **75 seconds**. A 10M-step run costs 28 minutes; 50M costs
2.3 hours.

**What this implies:** the ledger's conclusion that four gait variants were
"too small for selection" (1.72–1.86 m) is the expected outcome of comparing
policies at well under 1% of the budget a 3D biped needs. The rejection-screen
method is sound; it has been spent adjudicating noise.

**Decision:** re-baseline at 5–10M steps per seed before running further
variant comparisons. This is a measurement, not yet a successful change.

## 2026-07-27 — Three-Seed 10M Natural-Velocity Re-Baseline

**Question:** does the combined rebalanced, reopened-exploration configuration
learn stable walking at the dynamically natural 1.0 m/s target?

**Protocol:** seeds 21–23 each resumed their matched `velocity_sigma_010`
checkpoint for 10M continuation steps. The fixed configuration used 0.5 action
scale, `gamma: 0.995`, `log_std_init: -1.0`, `ent_coef: 1e-3`, low survival/fall
economics, and a 0.15→1.0 m/s ramp over 3M steps. Final models were evaluated
for 20 deterministic and 20 stochastic episodes at 1.0 m/s.

**Observed result:** deterministic displacement was 9.78, 11.26, and 9.02 m
for seeds 21–23; peak torso height never exceeded 0.71 m. Stochastic
displacement was 9.86, 10.37, and 5.67 m, respectively. The mechanism is not
yet robust: stochastic fall rates were 30%, 25%, and 60%, while stance slip was
0.48, 0.57, and 0.41 m/s. Stride length increased to 5.7–9.0 cm but remains
below the 10 cm gate.

**Decision:** retain the natural-speed direction but do not promote it. Next,
isolate slip/recovery mechanisms with a fixed baseline: (1) measure per-foot
slip, touchdown speed, and fall lead-up from renders/telemetry; (2) screen only
one conservative slip-control intervention at a time for 1M on seed 21; (3)
advance only candidates meeting ≤30% falls and ≤0.18 m/s slip to seeds 21–23 ×
3M; and (4) require all existing gait floors before transfer testing. Artifacts:
`outputs/walker_low_velocity_candidate/seed_{21,22,23}/scale_evaluation_1ms.json`.

## 2026-07-27 — Slip And Recovery Mechanism Evaluation

**Question:** are high-speed falls primarily touchdown overshoot, late-stance
push-off, or recovery failure, and can a bounded stance-slip cost fix the
measured mechanism without sacrificing 1.0 m/s locomotion?

**Source and constants:** the final seeds 21–23 10M natural-velocity
checkpoints. Phase 0 used five stochastic rollouts per seed with no training
changes. Phase 1 resumed the seed-21 checkpoint for a fixed 1M continuation,
kept the 1.0 m/s target, reward economics, action scale, gamma, and exploration
fixed, and compared only slip weights 0.25 and 0.5. Each final policy received
20 deterministic, 20 stochastic, and five rendered stochastic episodes.

**Mechanism measurement:** the 0.5 s before each fall retained per-foot
tangential velocity at touchdown and mid-stance, normal force, slip direction,
torso roll/pitch, action delta, and target/achieved velocity. Across 7,144
contact samples and 2,839 touchdowns, pooled p90 slip was 1.3398 m/s and
touchdown-force p90 was 1,452.4 N. Five of six falls were classified as recovery
failure and one as touchdown overshoot; none was late-stance push-off.

**Observed result:**

| Variant | Det. falls / slip | Stoch. falls / slip | Stoch. displacement | Cadence / stride |
|---|---:|---:|---:|---:|
| control | 15% / 0.487 m/s | 20% / 0.500 m/s | 11.20 m | 19.24 / 7.47 cm |
| slip 0.25 | 15% / 0.278 m/s | 15% / 0.348 m/s | 11.28 m | 21.47 / 6.91 cm |
| slip 0.5 | 5% / 0.182 m/s | 25% / 0.290 m/s | 10.85 m | 23.05 / 6.73 cm |

**Visual check:** all 15 Phase-1 renders retained rapid, short-stride foot
contacts. They showed no launch, but confirmed the same contact-chatter/shuffle
pathology and included six sampled falls. Paths:
`outputs/quality_studies_walker_slip_recovery_20260727/videos/phase1`.

**What worked:** both slip weights materially reduced measured slip without
destroying displacement or height safety. The bounded reward used the
measurement-derived p90 rather than an invented clip.

**What failed:** neither candidate reached the frozen stochastic 0.18 m/s slip
gate. The larger weight lowered deterministic slip to 0.182 m/s but did not
transfer that control to stochastic actions, and both weights increased contact
cadence while shortening stride.

**Decision:** reject both candidates. Touchdown overshoot was not dominant, so
do not run PD damping. No candidate held the slip gate, so do not run increased
reset noise, three-seed confirmation, or transfer. The next proposal must
address recovery state sensitivity and the short-stride/contact-chatter
mechanism without loosening the gates. Artifacts:
`outputs/quality_studies_walker_slip_recovery_20260727`.

## 2026-07-26 — Reward Economics And Natural-Velocity Ramp

**Question:** can reducing survival/fall distortion and ramping from 0.15 to
1.0 m/s over 3M continuation steps produce useful walking?

**Source and change:** seed 21 resumed the 150k `velocity_sigma_010`
checkpoint. Both arms used `alive_bonus: 0.125`, `fall_penalty: 2.0`, and slip
weight 0.5; the control retained 0.15 m/s, while the candidate used the fixed
three-million-step `VelocityTargetRampCallback`.

**Observed result:** across 20 stochastic episodes, the control achieved 800
mean steps, no falls, 1.97 m displacement, and 0.056 m/s stance slip. The ramp
achieved 7.70 m displacement, but only 594 mean steps, a 40% fall rate, and
0.317 m/s stance slip. Peak height remained safe at 0.689 m.

**Decision:** reject the candidate despite its +5.73 m displacement gain: it
fails the frozen 30% fall and 0.18 m/s slip limits. Artifacts:
`outputs/quality_studies_walker_velocity_ramp_3m_20260726`.

## 2026-07-27 — Action-Memory Evaluation Implementation

**Question:** can short-term action context make the stochastic policy less
state-sensitive during recovery and replace rapid contact chatter with longer
swings and strides?

**Source and change:** the previous 35-D checkpoints cannot be resumed after an
observation-shape change. The new `walker-action-memory` study therefore trains
matched seed-21 lineages from scratch: a 35-D coordinate-free v2 control and a
45-D candidate that appends the command actually applied to PD control on the
preceding step.

**Constants:** both arms retain the 0.15→1.0 m/s ramp, PPO settings, physics,
randomization, bounded 0.5 stance-slip intervention with its measured 1.3398
m/s clip, and every frozen fall/slip/displacement/height and real-gait gate.
There is no new reward term.

**Budget and decision rule:** the screen requires 10M steps per arm, 20
deterministic and stochastic episodes, and five stochastic renders. The
candidate advances only if every metrics gate passes and the renders receive
explicit approval. Confirmation trains seeds 22–23 at 10M; transfer remains
conditional on all three seeds.

**Implementation validation:** a 2,400-step smoke completed both observation
shapes, training, evaluation, fall telemetry, GIF rendering, report writing,
and the non-advancement path. Both sampled stochastic episodes fell and neither
arm met locomotion gates, which is expected at 0.024% of the evidence budget.

**Decision:** the workflow is ready, but no learning claim or candidate
selection follows from the smoke. Run the full seed-21 screen before deciding
whether previous-action context addresses recovery or gait structure.
Artifacts:
`outputs/quality_studies_walker_action_memory_smoke_20260727`.

## How To Append A Future Entry

Do not rewrite an old failure to make a later result look inevitable. Add a
dated entry containing:

1. **Question:** the specific uncertainty being tested.
2. **Source:** exact checkpoint family, seeds, and prior stage.
3. **Change:** configuration differences from the control.
4. **Constants:** physics, gates, rollout schedule, and other held parameters.
5. **Budget:** smoke, rejection, multi-seed, and transfer budgets actually run.
6. **Observed result:** deterministic and stochastic metrics, per seed.
7. **Visual check:** what the policy physically did and the video path.
8. **What worked:** improvements directly supported by evidence.
9. **What failed:** gate failures, exploits, variance, or methodological flaws.
10. **Decision:** reject, repeat, advance, or promote, with the exact reason.
11. **Artifacts:** output directory, report, relevant config, and commit.

If a historical value is later found to be wrong, append a dated correction
instead of silently replacing the original interpretation. Keep observation
separate from inference, and label proposed work as a hypothesis until it has
run.

## 2026-07-27 — Reward Structure: The Gate Measures Gait, The Reward Never Paid For It

**Question:** the preceding eight entries all shaped displacement, slip, or
touchdown proxies and all failed to lengthen the stride. Is the stride collapse
caused by the reward being *indifferent* to gait structure rather than by
insufficient budget, actuator authority, or exploration?

**Diagnostic measurements (before any training):**

Three candidate explanations were tested and *rejected*:

1. *No lateral degree of freedom.* All ten joints rotate about Y
   (`linkJointAxis=[[0,1,0]]*10`), and in single support the COM sits 35 mm
   outside the stance-foot support polygon. But sampling 300 actions in single
   support, 60% produced a **restoring** roll change with up to 8 rad/s of
   authority against 1.1 rad/s of accumulated drift. Lateral stabilisation is
   momentum- and contact-mediated, not impossible.
2. *PD gains suppress stride-scale motion.* With the torso pinned, the shipped
   `position_gain: 0.1 / velocity_gain: 1.0` tracks a 1 Hz hip swing at **0.97**
   achieved/commanded amplitude. Combined with the open-loop 33.7 mm clearance
   already recorded above, actuation is not the constraint.
3. *Budget.* Rejected previously; 10M costs ~30 min at the measured 5,500 fps.

The reward decomposition at the measured operating point (v = 0.889 m/s, from
4,193 stochastic steps on the seed-21 10M checkpoint) is:

| term | value/step |
|---|---:|
| alive | +0.750 |
| velocity | +0.278 |
| stance_slip | −0.250 |
| orientation | −0.100 |
| action | −0.013 |
| gait_step_progress | +0.007 |

Survival is **2.7× the velocity term** and one fall costs **144 steps** of
velocity reward, so the optimal policy minimises single-support time while
drifting forward — which is the chatter. Three specific defects:

- **The Gaussian velocity reward has no gradient away from its target.** At
  `velocity_sigma: 0.10` with `target_velocity: 1.0` it pays 3.3e-4 at 0.6 m/s
  and 1.3e-14 at 0.2 m/s. Sigma was calibrated in the 0.15 m/s regime and
  `VelocityTargetRampCallback` rescales only the target, so for most of the 3M
  ramp the velocity signal is under 1% of the alive bonus.
- **`gait_step_progress` is a duplicate of the velocity reward.** It pays pelvis
  advance between alternating touchdowns, which telescopes to total displacement
  and is therefore cadence-invariant. It is structurally incapable of preferring
  a long stride to a short one at the same speed — which is why weight 40.0
  produced no measurable effect.
- **`torque_penalty_weight` charges action magnitude, not rate.** It taxes a
  held stride and leaves high-frequency chatter free.

Measured directly: **at the same forward speed the control reward prefers a
1 Hz walk to the measured chatter by only +0.20/step**, and that margin comes
entirely from the slip term — whose own measured effect is to *shorten* contact.

**Change (candidate only):** `alive_bonus` 0.75 → 0.10, `fall_penalty` 40 → 2,
`velocity_mode` gaussian → `clipped_linear`, `stance_slip_penalty_weight`
0.5 → 0, plus three terms weighted against the measured signals:
`action_rate_penalty_weight: 0.10` (‖a_t − a_t−1‖² averages 1.45),
`swing_clearance_weight: 8.0` at a 0.05 m target (mean lift was 5.2 mm), and
`touchdown_rate_penalty_weight: 1.0` above 2.5 Hz (measured cycle frequency was
9.8 Hz). Under the candidate reward the same walk-vs-chatter comparison is
**+0.729/step**.

**Constants:** identical physics, PPO settings, seed, budget, randomization, and
0.15→1.0 ramp in both arms. Both trained **from scratch** — every prior walking
policy was a continuation of a balance checkpoint trained with
`log_std_init: -1.5, ent_coef: 0.0` to stand still.

**Budget:** paired single-seed (21) screen, 10M steps per arm, 20 deterministic
and 20 stochastic evaluation episodes at target 1.0 m/s.

**Observed result (stochastic):**

| metric | control | candidate | delta |
|---|---:|---:|---:|
| episode steps | 763 | 800 | +37 |
| displacement | 12.33 m | 16.07 m | +3.74 m |
| fall rate | 5% | 0% | −5 pts |
| stride length | 0.117 m | 0.413 m | **×3.5** |
| foot clearance | 7.6 mm | 63.4 mm | **×8.3** |
| swing duration | 0.083 s | 0.263 s | ×3.2 |
| cadence /100 steps | 11.90 | 9.09 | −2.81 |
| stance slip | 0.223 m/s | 0.280 m/s | +0.057 |
| peak torso height | 0.701 m | 0.723 m | +0.022 |

Frozen `WALKER_GAIT_GATE`: control **8/12**, candidate **10/12**. The candidate
newly passes foot clearance (0.0634 ≥ 0.02) and progress per alternating
touchdown (0.2147 ≥ 0.05). Both still fail the cadence ceiling (9.09 > 8.5) and
the slip ceiling (0.280 > 0.18).

**Visual check:** `walker_reward_v2_{control,candidate}.gif` and
`gait_signature.png`. The control's contact trace is dense bilateral tapping at
~5 Hz with 25–60 mm foot excursions — the documented chatter. The candidate
shows clearly separated contact blocks with real swing arcs peaking at 137 mm.
Peak torso height 0.72 m rules out launching in both. **However**, over 300
steps the candidate runs 30.7% right-foot contact against 16.0% left, mean
stance 0.090 s right against 0.057 s left, **0.7% double support and 54%
flight**. That is an asymmetric bounding run, not a walk.

**What worked:** the reward diagnosis is supported. Removing the survival
distortion, giving the velocity term a gradient, and paying directly for the
gated quantities moved stride and clearance by 3.5× and 8.3× — the first
material movement in either quantity across nine entries — while eliminating
falls. Training from scratch also beat the continuation lineage on its own
terms: the from-scratch *control* reached 0.117 m stride and 7.6 mm clearance
against the resumed lineage's 0.075 m and 0.7 mm, supporting the hypothesis that
continuing from an exploration-collapsed stand-still checkpoint suppresses gait
discovery.

**What failed:** the candidate does not pass every frozen gate, so it is not
promotion-ready. Cadence remains above the 2.5 Hz band, and removing the slip
penalty raised stance slip as expected. More importantly the resulting gait is
an asymmetric run: the reward now pays for clearance, stride, and cadence but
still contains **no term that prefers a walk to a run** — nothing rewards double
support or penalises flight, and nothing penalises left/right asymmetry.

**Decision:** do not promote, and do not run seeds 22–23, the 3M extension, or
transfer. Keep the reward changes as the new baseline for the next isolated
screen. The next change should add a double-support/flight term and a gait
symmetry term, and should be screened against this candidate as the paired
control. `max_stance_slip_speed: 0.18` still needs recalibration from a faster
reference gait before it gates anything at 1.2 m/s — the source comment already
says so, and this result does not license raising it.

**Lesson:** a reward that is *indifferent* between two behaviours cannot be
fixed by shaping a proxy for the one you want. Before adding another shaping
term, measure whether the reward already distinguishes the target behaviour from
the pathology at equal task performance. Here it differed by +0.20/step across
nine experiments' worth of tuning.

**Artifacts:** `outputs/quality_studies_walker_reward_v2_20260727/`,
configs `walker_reward_v2_control.yaml` and `walker_reward_v2_candidate.yaml`,
commit `cb2b4bc`.

## 2026-07-28 — Walk-vs-Run Screen: Symmetry Solved, Double Support Did Not Appear

**Question:** the reward_v2 candidate reached 0.41 m strides and 63 mm clearance
by *running* — 0.7% double support, 54% flight, 30.7% right-foot contact against
16.0% left. Do a flight penalty and a gait-symmetry penalty convert that into a
walk?

**Source:** `walker_reward_v2_candidate` seed-21 10M checkpoint serves as the
paired control directly; it is already a from-scratch run under identical
settings, so only the new arm was trained.

**Change:** two terms, and nothing else. `flight_penalty_weight: 0.5` charged per
step with neither foot supported; `gait_symmetry_penalty_weight: 0.5` charged in
proportion to the running left/right stance duty imbalance. Weights sized against
the measured control (0.54 flight fraction, 0.315 duty imbalance) so each lands
near 0.2–0.3/step against a ~1.0/step velocity term.

**Constants:** identical physics, PPO settings, seed 21, 10M budget,
randomization, 0.15→1.0 ramp, and every other reward weight.

**Pre-registered prediction:** measured against the control's own behaviour, the
v2 reward preferred a symmetric 1.2 Hz walk to the bounding run by only
+0.041/step — near-indifference. Under v3 the same comparison is +0.458/step.

**Observed result (stochastic, 20 episodes):**

| metric | v2 (ran) | v3 | |
|---|---:|---:|---|
| duty imbalance | 0.291 | **0.014** | symmetry solved |
| flight fraction | 57.6% | **29.0%** | halved |
| foot clearance | 63.4 mm | 81.8 mm | improved |
| **double support** | 0.1% | **0.4%** | **unchanged** |
| stance slip | 0.280 m/s | **0.880 m/s** | 3.1× worse |
| stride | 0.413 m | 0.252 m | worse |
| displacement | 16.07 m | 13.12 m | worse |
| fall rate | 0% | 5% | worse |

Frozen `WALKER_GAIT_GATE`: **10/12 for both arms** — no net gate movement. v3
newly passes nothing; it improves flight (0.29 vs 0.58, both already passing) and
still fails the cadence ceiling (9.58 > 8.5) and slip ceiling (0.880 > 0.18).

**Visual check:** `v3_surviving.png`, `v3_surviving.gif`. The first sampled
rollout fell at 71 steps and was discarded as unrepresentative; the second ran
799 steps at 1.03 m/s. It shows clean, near-perfectly alternating contacts
(imbalance 0.014) with 150–200 mm foot peaks — a genuine symmetric jog. But the
contact trace never overlaps: the robot is always either on one foot or
airborne.

**What worked:** the symmetry term is an unambiguous success and should be
retained — duty imbalance fell from 0.291 to 0.014, and the asymmetric limp
visible in the v2 renders is gone. The flight penalty also did what it was
weighted to do, halving flight fraction.

**What failed:** the screen did not produce a walk, and it cost displacement,
stride, and fall rate to get there. The mechanism is now clear and was a design
error on my part: **penalising flight is not the same as rewarding double
support.** Forced to keep a foot down, the policy satisfied the constraint by
extending *single* support (42.3% → 70.6%) rather than by overlapping the feet,
so double support stayed at ~0. Extending single-support contact at 1 m/s with
no slip penalty — removed in v2 because it was manufacturing chatter — is also
exactly what produced the 3.1× slip regression.

A second, procedural flaw: this screen changed **two** terms at once, violating
the one-causal-factor rule. Their effects happened to be separable in telemetry
(imbalance attributes to symmetry, flight fraction and slip to the flight
penalty), so the conclusions stand, but the separation was luck rather than
design.

**Decision:** reject v3 as a promotion candidate; do not run seeds 22–23 or
transfer. **Retain the symmetry term** — it is independently supported and cost
nothing attributable. **Drop or reduce the flight penalty** in favour of a direct
double-support reward, which is the term that actually encodes the target
behaviour. Reinstate a clipped stance-slip penalty at the same time: the chatter
mechanism that made it counterproductive in the v1 studies is now blocked by
`touchdown_rate_penalty_weight`, so its old failure mode no longer applies.

**Lesson:** a penalty on the complement of a behaviour is not a reward for that
behaviour when a third option exists. Flight, single support, and double support
are three states, not two; charging one of them let the policy escape into the
second rather than the intended third. Before pricing a behaviour by penalising
its opposite, enumerate every state the agent can occupy.

**Artifacts:** `outputs/quality_studies_walker_reward_v3_20260728/`,
config `walker_reward_v3_candidate.yaml`, commit `f51d15b`.

## 2026-07-28 — Double Support Screen: Best Gate Score Yet, And A Farmable Clearance Reward

**Question:** does pricing double support directly — rather than penalising its
complement — produce the double-support phase the v3 screen failed to create,
and does reinstating a clipped slip penalty control the slip regression?

**Source:** `walker_reward_v3_candidate` seed-21 10M checkpoint as paired
control. Only the new arm was trained.

**Change:** two coupled additions and nothing else.
`double_support_reward_weight: 0.5` and `stance_slip_penalty_weight: 0.35` with
`stance_slip_penalty_clip: 1.5`. The flight penalty was retained at 0.5 — it did
halve flight, and its failure was the missing double-support term.

**Calibration:** both weights were measured, not guessed. The double-support
weight is bounded above by a standing exploit (standing scores 100% double
support): a 1 m/s walk beats standing by +0.757/step at weight 0.5, +0.007 at
1.5, and **loses by −0.368 at 2.0**. That bound is now a test. The slip clip is
deliberately **not** the p90: v3 slip is heavy-tailed (p50 0.219, p75 0.565,
p90 3.60, max 7.47), so a p90 clip would hand the gradient to touchdown-impulse
spikes. 1.5 sits above the bulk and below the spikes.

**Observed result (stochastic, 20 episodes):**

| metric | v3 | v4 | |
|---|---:|---:|---|
| **double support** | 0.4% | **5.2%** | first movement in this quantity |
| **cadence /100** | 9.58 | **7.30** | **now inside the 1.5–8.5 band** |
| **stance slip** | 0.880 m/s | **0.287 m/s** | 3.1× better, still fails 0.18 |
| displacement | 13.12 m | 13.50 m | + |
| flight | 29.0% | 36.4% | worse |
| foot clearance | 81.8 mm | 41.5 mm | worse |
| stride | 0.252 m | 0.218 m | worse |
| duty imbalance | 0.014 | ~0.30 | **regressed** |

Frozen `WALKER_GAIT_GATE`: **11/12 — the best score any arm has reached.** The
cadence ceiling passes for the first time. The only remaining failure is
`stance_slip <= 0.18` at 0.287.

**Visual check:** `v4_surviving.png`, `v4_surviving.gif`, 799 steps at 1.06 m/s.
The robot **hops on its right leg while the left leg does all the swinging** —
right foot peaks at 50–80 mm with visible contact chatter, left foot swings
150–270 mm. Right foot is in contact 47.2% against the left's 23.7%.

**Root cause of the asymmetry — a defect in the v2 reward implementation.** The
env feeds `swing_clearance` from `GaitStep.swing_clearance_now`, which is the
**max over currently-swinging feet**. One high-swinging leg therefore collects
full clearance credit while the other drags. The telemetry proves the farm:

| | v3 | v4 |
|---|---:|---:|
| `reward_swing_clearance_mean` (earned) | 0.360 | 0.257 (−29%) |
| `gait_foot_clearance_mean` (measured, gated) | 0.0818 | 0.0415 (−49%) |

The policy retained 71% of the reward while real per-foot clearance halved, and
it is *paying* a 4.8× larger symmetry penalty (−0.028 → −0.133) to do so because
the farm is worth more than the penalty. The gate telemetry is honest — it
averages per-touchdown clearances per foot — so only the reward term was wrong.
This defect was latent in v2 and v3; it only became profitable once symmetry and
double-support pressure made single-leg specialisation attractive.

**What worked:** pricing the target state directly moved it. Double support went
0.4% → 5.2% where the flight penalty alone had moved it 0.1% → 0.4%. The
reinstated slip penalty cut slip 3.1× without reproducing the chatter it caused
in the v1 studies, confirming that `touchdown_rate_penalty_weight` closes that
escape route. Cadence entered the frozen band.

**What failed:** 5.2% double support is still far from the ~25% of a walk, and
the arm regressed clearance, stride, flight, and symmetry. It is not
promotion-ready at 11/12.

**Change made in response:** added `touchdown_clearance_weight`, which pays
per-foot on the step a foot lands, proportional to the clearance that foot
actually achieved — the same quantity `gait_foot_clearance` gates on, so it
cannot be farmed by one leg. `swing_clearance_weight` is retained at zero
default with a warning so v2–v4 stay reproducible.

**Decision:** reject v4 as a promotion candidate; do not run seeds 22–23 or
transfer. Do **not** run another reward screen until the clearance term is
switched to the per-foot form — every clearance result from v2 onward was
measured against a farmable signal. The next screen should be v4 with
`swing_clearance_weight: 0` and `touchdown_clearance_weight` sized to match the
dense term's expected value, and should re-check whether the symmetry regression
disappears once the farm is closed.

**Lesson:** when a reward aggregates over interchangeable body parts, check
whether the aggregation is farmable by specialising one of them. `max` over feet
rewards having *a* good leg; the gate measures *both* legs. Whenever a reward
term and its gate compute the same-sounding quantity differently, the policy will
find the gap — here it was worth more than the penalty explicitly designed to
prevent it.

**Artifacts:** `outputs/quality_studies_walker_reward_v4_20260728/`,
config `walker_reward_v4_candidate.yaml`, commit `0b4442f`.

## 2026-07-28 — Clearance-Farm Screen: Hypothesis Confirmed, And The Fourth Hack

**Question:** the v4 arm hopped on one leg because `swing_clearance` is the max
over swinging feet, letting one high leg collect full credit. Does closing that
farm restore the symmetry v3 had achieved?

**Source:** `walker_reward_v4_candidate` seed-21 10M checkpoint as paired
control. Only the new arm was trained.

**Change:** exactly one. `swing_clearance_weight` 8.0 -> 0 and
`bilateral_clearance_weight` 0 -> 8.0, same 0.05 m target. The bilateral term
pays the LOWER of the two feet's most recent swing clearances, densely, with a
stale value counted as zero. It was chosen over the per-touchdown form added in
`db2d21f` because that form scales with touchdown COUNT and would fight the
cadence ceiling v4 had just brought inside its band.

**Observed result (stochastic, 20 episodes):**

| metric | v4 | v5 | |
|---|---:|---:|---|
| stance duty imbalance | 0.303 | **0.015** | hypothesis confirmed |
| **double support** | 5.2% | **31.4%** | into the walk range |
| single support | 58.5% | **13.9%** | collapsed |
| flight | 36.4% | 54.7% | worse |
| cadence /100 | 7.30 | 4.20 | still in band |
| progress/touchdown | 0.1049 | 0.0463 | now FAILS |
| fall rate | 5% | 15% | worse |
| episode length | 763 | 688 | worse |
| mid-stance slip | 0.184 | 0.244 | worse |

Frozen `WALKER_GAIT_GATE`: **10/12**, down from v4's 11/12. Newly failing:
progress per alternating touchdown.

**Visual check:** `v5_surviving.png`, `v5_surviving.gif`, 799 steps at 1.16 m/s.
The two foot-height traces are **nearly superimposed** and the contact bands are
aligned: both feet lift together to ~90-100 mm, land together, and leave
together. The robot is **bunny-hopping on both legs**.

**What worked:** the clearance farm was indeed the cause of v4's asymmetry.
Closing it restored symmetry outright — stance duty imbalance 0.303 -> 0.015 —
**without touching `gait_symmetry_penalty_weight`**. That was the pre-registered
prediction and it held. Double support also finally reached the walk range.

**What failed:** the gait is not a walk. Double support 31.4% plus flight 54.7%
leaves only 13.9% single support, against roughly 75% for a real walk. Every
term is satisfied by hopping: two feet moving in phase are perfectly symmetric,
land together (double support), lift together (bilateral clearance), and do so
within the cadence band.

**The pattern across four screens.** Each added property term has been satisfied
by a new non-walking gait:

| screen | added | resulting gait |
|---|---|---|
| v2 | clearance, stride, action-rate, cadence | asymmetric bound |
| v3 | flight penalty, symmetry | symmetric jog |
| v4 | double support, slip | one-legged hop |
| v5 | bilateral clearance | two-footed bunny hop |

The reward now specifies the *marginal properties* of a walk — clearance,
cadence, symmetry, support-state fractions — but never the one thing that
defines it: **the legs must be in anti-phase.** Symmetry is satisfied by
in-phase motion just as well as by alternation, because equal duty says nothing
about ordering. No per-step marginal can express a phase relationship, which is
why each new term has been absorbed rather than resisted.

**Decision:** reject v5 as a promotion candidate. Retain the bilateral clearance
term — it fixed the farm and restored symmetry, both confirmed. Do **not** add a
fifth marginal property term. The next change should encode the gait *cycle*
directly: a phase clock in the observation plus a reward on matching a periodic
reference contact schedule (the standard approach in the sim-to-real bipedal
literature), which forces the R -> RL -> L -> LR ordering that marginals cannot.
This is also the observation work deferred on 2026-07-27; it is
checkpoint-incompatible and should be bundled with the torso-frame velocity and
heading fixes so the obs change is paid for once.

**Note on the slip gate.** v4 remains the best arm at 11/12 and its only failure
was `stance_slip <= 0.18` at 0.287 — which the new `gait_mid_stance_slip_speed`
telemetry shows is dominated by landing impact, not sliding (0.153 mid-stance,
under the ceiling). Whether the gate should measure mid-stance or whole-stance
slip is an open decision that must be made on mechanism, not on v4's numbers.
No threshold has been changed.

**Lesson:** properties are not structure. A set of per-step marginal constraints
can be jointly satisfied by a behaviour that violates the relationship between
them — here, four independent gait properties were all satisfied by hopping.
When the target behaviour is defined by a temporal relationship between parts,
reward the relationship, not a checklist of its symptoms.

**Artifacts:** `outputs/quality_studies_walker_reward_v5_20260728/`,
config `walker_reward_v5_candidate.yaml`, commit `7c9eb68`.

## 2026-07-28 — Gait Phase Screen: Anti-Phase Achieved, Schedule Kinematically Unreachable

**Question:** four screens of per-step property terms were each absorbed by a
non-walking gait, ending with v5 hopping on both legs in phase. Does a
phase-indexed reference contact schedule produce anti-phase alternation?

**Source:** `walker_reward_v5_candidate` seed-21 10M checkpoint as paired
control. Both arms trained from scratch (v6's observation is 37-D, not 35-D).

**Change:** `observation.include_gait_phase` appends (sin, cos) of a gait clock;
`phase_contact_weight: 1.0` pays the fraction of feet matching the schedule
(right leg in stance over phase [0, 0.6), left the same window half a cycle
later). `double_support_reward_weight` 0.5 -> 0, since it is subsumed and in
exact conflict: standing on both feet during a single-support window trades 0.5
of phase match for 0.5 of bonus, leaving double-support timing unconstrained.

**Observed result (stochastic, 20 episodes):**

| metric | v5 | v6 | |
|---|---:|---:|---|
| **single support** | 13.9% | **62.1%** | the target |
| right-only / left-only | 7.5 / 6.4% | **30.0 / 32.1%** | balanced |
| double support | 31.4% | 3.5% | |
| flight | 54.7% | 34.4% | |
| phase match reward | 0.000 | 0.616 | of a possible 1.0 |
| displacement | 13.32 m | 14.92 m | + |
| episode length | 688 | 782 | + |
| fall rate | 15% | 10% | + |
| cadence /100 | 4.20 | 9.82 | now FAILS the 8.5 ceiling |
| foot clearance | 33.4 mm | 16.4 mm | now FAILS the 20 mm floor |

Frozen `WALKER_GAIT_GATE`: **9/12**, down from v5's 10/12 and v4's 11/12.

**Visual check:** `v6_surviving.png`, `v6_surviving.gif`, 799 steps at 1.17 m/s.
The two foot-height traces are **no longer superimposed** — they alternate. The
bunny hop is gone and duty imbalance is 0.059. Contact bursts are irregular
rather than a clean rhythm.

**What worked:** the phase clock did exactly what the four preceding marginal
terms could not. Single support went 13.9% -> 62.1% with the two legs
near-perfectly balanced, and anti-phase alternation appeared for the first time
in this ledger. Displacement, episode length and fall rate all improved.

**What failed — a specification conflict I introduced.** The schedule period and
the velocity target are mutually unreachable on this morphology:

| quantity | value |
|---|---:|
| leg length, hip to ankle | 0.480 m |
| hip excursion at `action_scale: 0.5` | ±0.600 rad |
| kinematic step ceiling, `2 L sin(theta)` | 0.542 m |
| kinematic stride ceiling | 1.084 m |
| **top speed at a 1 Hz cycle** | **1.08 m/s** |
| **speed the policy actually runs** | **1.14 m/s** |

A 1 Hz cycle at 1.14 m/s demands a 1.14 m stride, which exceeds the ceiling even
at full hip excursion. The policy cannot satisfy both, and the velocity term
(1.0/step) outweighs the phase term, so it kept the speed and took partial phase
credit (0.616). Stepping at 2.95 Hz against a 1 Hz clock leaves roughly 0.17 s
per swing, which is why clearance halved to 16.4 mm and failed its gate; the
achieved 0.21 m stride accounts for only part of the distance, with stance slip
at 0.347 m/s making up the rest. The policy uses 77% of the available hip range,
so this is a schedule/target inconsistency, not an actuation limit.

**Decision:** retain the phase clock and schedule — the mechanism is validated.
Reject v6's *parameters*. The next screen changes one number:
`gait.cycle_period_s` 1.0 -> 0.55, which at the 1.0 m/s target demands a 0.55 m
stride, 51% of the kinematic ceiling, and implies 6.06 alternating touchdowns per
100 steps, inside the frozen 1.5-8.5 band. Nothing else changes.

**Lesson:** a reference schedule is a kinematic claim, not just a reward shape.
`v = stride x cycle_frequency` ties the period to the velocity target and the
leg geometry; choosing a period without checking that product asks the policy to
satisfy two constraints that cannot both hold, and it will drop the cheaper one.
Check the arithmetic of a reference trajectory against the morphology before
training against it.

**Artifacts:** `outputs/quality_studies_walker_reward_v6_20260728/`,
config `walker_reward_v6_candidate.yaml`.

## 2026-07-28 — Correction: Swing Segmentation Was Not Debounced

A dated correction to every clearance, swing-duration and stride figure recorded
above, filed under the rule that a wrong historical value is corrected by
appending. No earlier observation is retracted; the measurements were real, the
instrument was not.

**What was wrong.** `WalkerGaitTracker` debounced *touchdowns*
(`step - _last_touchdown_steps[side] >= touchdown_debounce_steps`) but reset a
swing's reference height on **any** raw contact break. The same contact stream
was therefore filtered two different ways, and a single swing interrupted by a
momentary re-contact was recorded as several small swings.

**Evidence.** On the reward_v7 checkpoint, re-deriving clearance while merging
contact gaps shorter than N steps:

| merge gaps <= | swings counted | mean clearance |
|---:|---:|---:|
| 0 (what the tracker did) | 150 | 7.2 mm |
| 3 | 85 | 14.2 mm |
| 5 | 45 | 20.8 mm |

Cadence independently implied ~45 swings (5.90 alternating touchdowns per 100
steps over 762 steps), so the raw signal over-counted by 3.2x. Raw foot-height
range over the same episode was 60 mm against a reported 7.2 mm clearance.

**Why it mattered beyond the metric.** `bilateral_clearance_weight` is fed from
this signal, and on v7 it earned **0.003 of a possible 0.400** — 27% of the
reward left untaken with no gradient to climb, because the quantity it paid for
was being measured as noise.

**Change.** Event segmentation (swings, stances, touchdowns) now runs on a
debounced contact signal; a swing is dated from the *raw* break so the
confirmation lag does not eat the early part of the lift; slip and the
support-occupancy fractions stay on the raw signal, being instantaneous
measures rather than events.

**Corrected re-evaluation, all six arms, 20 stochastic episodes:**

| | v2 | v3 | v4 | v5 | v6 | v7 |
|---|---:|---:|---:|---:|---:|---:|
| stride (m) | 0.460 | 0.334 | 0.295 | 0.235 | 0.342 | **0.527** |
| clearance (m) | 0.069 | **0.107** | 0.055 | 0.038 | 0.021 | 0.018 |
| single support | 42.3% | 70.6% | 58.5% | 13.9% | 62.1% | **74.9%** |
| flight | 57.6% | 29.0% | 36.4% | 54.7% | 34.4% | **20.7%** |
| stance slip | 0.280 | 0.880 | 0.287 | 0.340 | 0.347 | **0.218** |
| mid-stance slip | 0.319 | 0.196 | 0.208 | 0.251 | 0.249 | **0.188** |
| phase match | — | — | — | — | 0.616 | **0.802** |
| **gates** | 11/12 | 10/12 | 10/12 | 11/12 | **11/12** | 10/12 |

The ranking moves: v4 drops 11/12 -> 10/12 (cadence 8.56 now exceeds the 8.5
ceiling) and v6 rises 9/12 -> 11/12 (clearance and cadence both pass). Clearance
was never as catastrophic as the broken instrument suggested — it was 38-107 mm
in v2-v5, not 7-33 mm — so the "clearance collapsed" reading in the v6 and v7
entries above overstated a real but smaller decline.

**What the corrected data says.** v7 is structurally the closest to a walk this
project has produced: 74.9% single support, 20.7% flight, 0.527 m stride at
1.77 Hz, legs balanced, contact phase-locked at 0.802. Its two remaining gate
failures are narrow — clearance 0.0176 against 0.020, and stance slip 0.218
against 0.180 (mid-stance 0.188, essentially at the line).

**On the slip gate.** `stance_slip <= 0.18` now fails in **all six arms**, best
0.218, across gaits ranging from a bound to a hop to a phase-locked walk. A
threshold measured in the 0.1 m/s regime that no gait at ~1.1 m/s has come
within 20% of is more likely mis-scaled than universally violated. It still has
not been changed, and should not be changed to make a candidate pass; it needs
recalibration from a reference gait measured at the target speed, as the source
comment on `WALKER_GAIT_GATE` already requires.

**Lesson.** When two metrics derived from the same signal disagree by a constant
factor, suspect the derivation before the behaviour. Cadence said 45 swings and
clearance said 150; that ratio was visible in the data for five entries before
anyone divided one by the other. A reward term reading near-zero while its
theoretical maximum is 27% of total reward is the same signal in a different
form — an untaken gradient that large is usually a broken measurement, not a
stubborn policy.

**Artifacts:** `outputs/quality_studies_walker_recheck_20260728/`.

## 2026-07-28 — Correction: Stance Slip Measured The Link Origin, Not The Contact

A second dated correction to the instrument, filed like the one above.
`stance_slip <= 0.18 m/s` had failed in all six arms; this explains part of why,
and — more importantly — shows why relaxing it would be the wrong move.

**What was wrong.** `_foot_slip_speeds` returns the velocity of the foot **link
origin**. A foot rotating about a stationary contact edge — ordinary heel-off
and toe-off — moves its origin while nothing slides. Slip proper is the velocity
of the material point at the contact:

    v_contact = v_com + omega x (r_contact - r_com)

**Measured across all six checkpoints (20 stochastic episodes each):**

| | v2 | v3 | v4 | v5 | v6 | v7 |
|---|---:|---:|---:|---:|---:|---:|
| link origin (gated) | 0.280 | 0.880 | 0.287 | 0.340 | 0.347 | 0.218 |
| **contact point (true slip)** | **0.128** | **0.713** | 0.190 | 0.206 | 0.227 | **0.163** |
| rotation share | 55% | 18% | 37% | 39% | 37% | 34% |

Rotation accounts for 18-55% of the gated value, and the share grows with how
much the foot pivots. v7's median contact-point slip is 0.010 m/s — the foot is
planted to within a centimetre per second half the time.

**Validity check.** A slip metric that merely flatters candidates would be
useless. This one still separates v3 — the arm rendering showed visibly skidding
— at 0.713, three to six times every other arm. It discriminates.

**Caveat on the estimator.** The value taken is the *slowest* contact point, on
the argument that it is the one anchoring the foot. That is the generous choice;
a mean over contact points would read higher. The v3 separation holds either way.

**Why this does NOT license relaxing the gate.** Under contact-point slip, v2
passes stance slip — and stance slip was its *only* failure, so **v2 would score
12/12**. v2 is the asymmetric bound: 57.6% flight, 0.1% double support, right
foot in contact 27.3% against the left's 15.0%, and its render shows a bounding
run. A gate set that awards a perfect score to that is not measuring walking.
The frozen set has no double-support floor and a 60% flight ceiling, neither of
which a bound violates.

**Decision:** `gait_contact_point_slip_speed` is added as **telemetry only**.
`max_stance_slip_speed` is unchanged, still 0.18, and still evaluated on the
link-origin metric. Switching the gate to the physically correct quantity is
justified on its own terms, but doing it in isolation would certify a bound as a
walk, so it must be paired with a double-support floor and a tighter flight
ceiling. That is a gate-design decision, not a measurement one, and it is left
open rather than made with candidates in view.

**Lesson.** Fixing a measurement can be correct and still be the wrong thing to
act on alone. The right instrument revealed that the *gate set*, not the
threshold, was the weaker link: v2 was one metric definition away from a perfect
score on a gait nobody would call walking. When correcting an instrument moves a
candidate across a line, check what else is now on the passing side.

**Artifacts:** `outputs/quality_studies_walker_recheck_20260728/`.

## 2026-07-29 — Gait Gate V2 Frozen Before The Corrected V7 Rerun

**Question:** how can the gate use the physically correct contact-point slip
metric without certifying the v2 bound as a walk?

**Calibration rule:** support thresholds are derived from gait semantics and
the configured reference schedule, not from the six candidate measurements.
For an anti-phase schedule with `stance_duty: 0.6`, each foot is planted for
60% of a cycle, which implies 20% nominal double support and zero flight. Gate
v2 retains half that nominal overlap as its floor and permits an equal
raw-contact tolerance in the opposite direction:

- double support at least 10%;
- flight at most 10%.

The slip metric switches from foot-link-origin speed to
`gait_contact_point_slip_speed_mean`, the planar velocity of the slowest
material contact point. The numeric ceiling remains 0.18 m/s; correcting the
instrument is not used as a reason to loosen it. Cadence, progress, stride,
clearance, survival, fall, displacement, height, and same-foot-sequence limits
are unchanged.

**Freeze:** `WALKER_GAIT_GATE["gate_version"] == 2` and regression tests pin
the metric and all three thresholds. The gate was committed before producing a
new v7 training result. The next experiment is a from-scratch seed-21 rerun of
the otherwise unchanged v7 policy. Unlike the previous checkpoint, it will
train against the corrected debounced bilateral-clearance signal. Seeds 22–23
remain blocked unless seed 21 passes every v2 criterion and visual review.

## 2026-07-29 — Corrected V7 Rerun: Clearance Fixed, Support Gate Fails

**Question:** with the bilateral-clearance reward finally receiving correctly
debounced swing events, does an otherwise identical v7 run clear the gait gate?

**Constants:** seed 21, 10M PPO steps from scratch, 0.15→1.0 m/s velocity ramp,
0.55 s gait period, 0.6 stance duty, all reward weights, physics,
randomization, and optimizer settings. The only YAML difference is
`experiment_name`, so the original checkpoint is preserved. Gate v2 was
already committed as `ccce8fc`.

**Result:** training completed at 10,002,432 steps. The corrected reward was no
longer starved: its stochastic contribution increased from 0.100 to 0.218 per
step. Twenty deterministic and stochastic episodes then completed with no
falls.

| stochastic metric | original v7 | corrected rerun | gate |
|---|---:|---:|---:|
| episode steps | 762.25 | **800.00** | ≥600 |
| displacement | 13.72 m | **13.87 m** | ≥5 m |
| contact-point slip | 0.163 m/s | **0.124 m/s** | ≤0.18 m/s |
| cadence / 100 steps | 5.90 | **5.99** | 1.5–8.5 |
| progress / touchdown | 0.273 m | **0.284 m** | ≥0.05 m |
| stride | 0.527 m | **0.569 m** | ≥0.10 m |
| clearance | 17.6 mm | **32.5 mm** | ≥20 mm |
| double support | 4.42% | **4.91%** | ≥10% |
| flight | 20.66% | **12.96%** | ≤10% |
| longest same-foot sequence | 2.10 | **1.65** | ≤5 |

**Decision:** reject for promotion at 10/12. The clearance mechanism worked,
and flight moved materially toward walking, but double support remained below
half the frozen floor and flight remained 2.96 percentage points above its
ceiling. Because the metrics gate failed, visual approval is not reached and
seeds 22–23 and transfer remain blocked. Do not rerun v7 again for seed luck.
The next hypothesis, if pursued, must target support occupancy explicitly while
holding gate v2 and the corrected bilateral signal fixed.

**Artifacts:**
`outputs/walker_reward_v7_corrected_rerun/seed_21/checkpoints/final_model.zip`
and
`outputs/quality_studies_walker_reward_v7_corrected_20260729/comparison.json`.

## 2026-07-29 — V8 Scheduled Support Overlap: Support Recovered, Flight Fails

**Question:** can a reward paid only for scheduled double support recover the
gait gate's support occupancy without the standing and in-phase-hop exploits of
an unconditional double-support bonus?

**Constants:** v7's corrected bilateral-clearance signal, 0.55 s cycle, 0.6
stance duty, seed 21, 10M steps, PPO schedule, physics, randomization, and
frozen gait gate v2. **Only intervention:**
`phase_double_support_reward_weight: 0.5`, which pays when both feet contact
during the phase reference's 20% overlap, rather than whenever both feet happen
to contact.

**Result:** the run completed at 10,002,432 steps. Across 20 stochastic
episodes, double support rose from v7's 4.91% to **11.59%**, exceeding the 10%
floor. Contact-point slip remained compliant at **0.170 m/s**, displacement was
13.28 m, falls were 5%, stride was 0.535 m, and clearance was 34.5 mm. Flight
remained **22.04%**, however, above the frozen 10% ceiling. The 20 deterministic
episodes also showed 13.69% double support but 16.51% flight.

**Decision:** reject at 11/12 behavior criteria. The intervention validates
scheduled support as a real lever, but does not establish walking because flight
is still too frequent. No visual approval, seeds 22–23, or transfer evaluation
is permitted. Preserve gate v2; any successor must lower flight while retaining
the recovered support and slip.

**Artifacts:** `outputs/walker_reward_v8_support_overlap/seed_21/checkpoints/final_model.zip`.

## 2026-08-02 — Slip Ceiling: Recalibration Attempted, Threshold Unchanged

**Question:** `max_slip_speed: 0.18` was frozen when `target_velocity` was
0.15 m/s and is now applied at 1.0 m/s. The source comment on
`WALKER_GAIT_GATE` has required recalibration "from a reference gait measured at
the target speed" since 2026-07-26. Can that be done before the next candidate?

**The intended method is unavailable, and the reason is worth recording.** The
clearance floor was calibrated from an open-loop scripted gait (33.7 mm at the
study's own action scale). That worked because clearance is an *actuator
capability* measure: it survives a rollout that falls within a second, because
falling does not change how high the actuator can lift a foot. Slip is not a
capability measure. It only means anything during sustained balanced locomotion
at speed, and the only policies that do that are the candidates themselves.
Calibrating a gate from candidate measurements is precisely the error that made
gate v1 certify contact chatter. There is no third source, so the threshold was
not moved.

**What the measurement did establish.** The 2026-07-28 correction fixed the
*rotation* artifact by switching from foot-link origin to contact point. It did
not fix the *landing transient*: `gait_contact_point_slip_speed` averages the
whole contact phase, including the step where the foot is still arriving. The
tracker's own comment already recorded that transient at 1.034 m/s against
0.153 m/s for the rest of stance on reward_v4. The two corrections are on
orthogonal axes, and the cross cell had never been measured:

| | whole-phase | mid-stance |
|---|---|---|
| link origin | `gait_stance_slip_speed` | `gait_mid_stance_slip_speed` |
| contact point | `gait_contact_point_slip_speed` ← **gate v2** | *did not exist* |

Twenty stochastic episodes per checkpoint, both at 1.04 m/s mean forward speed:

| | whole-phase | mid-stance | transient share |
|---|---:|---:|---:|
| v7 corrected, link origin | 0.1949 | 0.1378 | 29% |
| **v7 corrected, contact point** | **0.1206** | **0.0764** | **37%** |
| v8, link origin | 0.2712 | 0.2276 | 16% |
| **v8, contact point** | **0.1667** | **0.1503** | **10%** |

**Why this matters.** The landing transient carries roughly the body's forward
speed, so it scales with locomotion speed; ideal mid-stance slip is zero at any
speed. The gated metric therefore inherits a speed-proportional term that the
quantity it is meant to measure does not have — which is the real mechanism
behind the suspicion that a ceiling set at 0.15 m/s misbehaves at 1.0 m/s. Worse
for gate hygiene, the transient's share is *gait-dependent* (37% for v7, 10% for
v8), so two candidates can be separated by how they land rather than by how much
they slide.

Median contact-point slip is 0.0088 (v7) and 0.0149 (v8) m/s: the foot is
planted to within a centimetre or two per second half the time, and the mean is
set by a small number of transient samples. A standing robot holding rest pose
measures 0.0065 m/s mean, so the instrument's own floor is ~25x below the
ceiling; the ceiling is not absurd on noise-floor grounds.

**Decision:** add `gait_mid_stance_contact_point_slip_speed` as **telemetry
only**. `max_slip_speed` stays 0.18 and stays on the whole-phase contact-point
metric. This follows the 2026-07-28 precedent exactly: a corrected instrument is
not swapped into a live gate with a candidate in view, because the question of
what else lands on the passing side has to be asked separately from the question
of what is physically correct. Both v7 and v8 pass 0.18 on both metrics, so no
decision to date turned on the difference — but v8's 6% margin is thin, and a v9
that reduces flight will spend more time landing, so the telemetry is in place
before it is needed rather than after.

**Lesson.** Two instrument corrections on orthogonal axes do not compose by
themselves. Fixing rotation in 2026-07-28 made the metric physically better and
left a second, independent contaminant untouched — and because the first fix was
framed as "slip is now correct", the second was invisible for five days. When
correcting a measurement, name the axis being corrected, so the axes not being
corrected stay visible.

**Artifacts:** measurement script and raw output under the session scratchpad;
metric added in `src/rl_framework/envs/locomotion/gait.py`.

## 2026-08-02 — V9 Flight Cost: Intervention Selected, Duplicate Rejected

**Question:** v8's only failing criterion was flight (22.04% against a 10%
ceiling). What is the single intervention that lowers it without giving back the
scheduled double support v8 recovered?

**A proposed "unscheduled flight penalty" was rejected as a duplicate.** The
idea was the symmetric complement of v8's phase-gated double-support reward:
charge flight only where the reference schedule does not expect it. With
`stance_duty: 0.6`, `_expected_stance` puts at least one foot in stance at every
phase — verified numerically, the schedule's expected flight fraction is exactly
0.0000, and `walker_bullet.py` says so in the docstring ("the two windows
overlap, producing double support and no flight phase"). So "flight outside the
schedule" is identical to "flight", and `flight_penalty_weight` has been
charging exactly that, unconditionally, at 0.5 since v7. The new term would have
been mathematically equal to an existing active one. A duty cycle below 0.5
would make the two distinct, but that is a different gait, not a different
penalty.

**The trade v8 made, priced from its own recorded metrics:**

| | per step |
|---|---:|
| phase_double_support earned (0.5 × 11.59%) | +0.0580 |
| extra flight cost (0.5 × [22.04% − 12.96%]) | −0.0454 |
| **net** | **+0.0125** |

The policy bought scheduled overlap with airtime because airtime was cheap. The
weight at which that trade is neutral is 0.64.

**Intervention:** `flight_penalty_weight: 0.5 → 1.0`, ~1.6x break-even, so it is
a gradient rather than a tie-breaker. Every other value is byte-identical to v8,
including `phase_double_support_reward_weight: 0.5`. Config:
`walker_reward_v9_flight_cost.yaml`; the diff against v8 is two lines, one of
them `experiment_name`.

**Guard:** the obvious failure mode of a large flight penalty is chatter — never
lift a foot, never fly. Gate v2 already blocks it with the 1.5–8.5 cadence band
and the 0.10 m stride and 0.02 m clearance floors, none of which were touched.

**Status:** not yet run. Seed 21, 10M, gate v2 frozen. Seeds 22–23 and transfer
remain blocked unless seed 21 clears all twelve criteria and visual review.

## 2026-08-02 — V9 Flight Cost: Flight Recovered, Support Given Back

**Question:** v8 cleared the double-support floor but flew 22.04% against a 10%
ceiling. Does doubling the price of flight remove the excess airtime while
keeping the scheduled overlap v8 recovered?

**Constants:** seed 21, 10M PPO steps from scratch, 0.15→1.0 m/s ramp, 0.55 s
cycle, 0.6 stance duty, gate v2, and every other reward, physics, randomization
and optimizer setting. **Only intervention:** `flight_penalty_weight: 0.5 →
1.0`, chosen at ~1.6x the 0.64 break-even weight computed from v8's own reward
economics. The config diff against v8 is two lines.

**Result:** training completed at 10,002,432 steps. Twenty stochastic and 20
deterministic episodes completed with no falls.

| stochastic metric | v7 corrected | v8 | **v9** | gate |
|---|---:|---:|---:|---:|
| flight | 12.96% | 22.04% | **12.39%** | ≤10% |
| double support | 4.91% | 11.59% | **8.88%** | ≥10% |
| single support | 82.13% | 66.37% | **78.73%** | — |
| contact-point slip | 0.124 | 0.170 | **0.153** | ≤0.18 |
| displacement | 13.87 m | 13.28 m | **14.29 m** | ≥1 m |
| falls | 0% | 5% | **0%** | ≤30% |
| stride | 0.569 | 0.535 | **0.592 m** | ≥0.10 |
| clearance | 32.5 mm | 34.5 mm | **22.3 mm** | ≥20 mm |
| cadence /100 | 5.99 | — | **5.93** | 1.5–8.5 |

**Decision:** reject at 11/13. Deterministic scored 12/13 (flight 9.77%, under
the ceiling; double support 9.33% the sole miss), but the gate evaluates
stochastic, and there both support criteria fail.

**The intervention did exactly what it was priced to do, and that is the
finding.** It removed 9.65 points of flight — almost precisely the 9.08 that v8
had added — and gave back 2.71 points of double support. v8 and v9 now bracket
the target without either reaching it. Read through the support decomposition,
v8 converted 15.8 points of single support into +6.7 double and +9.1 flight;
v9 converted most of it back. A scalar on either term moves both.

**Where the shortfall actually is — not aiming, duration.** The phase telemetry
rules out the obvious explanation:

- `reward_phase_contact_mean` 0.869 at weight 1.0 → the policy matches the
  reference contact schedule on 86.9% of steps;
- `reward_phase_double_support_mean` 0.0422 at weight 0.5 → 8.44% of all steps
  are simultaneously in the overlap window *and* in double support;
- that is **95.0% of the policy's total double support**, so essentially all of
  the overlap it produces is already correctly scheduled.

The policy is not failing to aim at the window. It fills only **42.2% of the
20% window** the schedule provides, and would need 50% to reach the 10% gate.
The deficit is how long both feet stay down, not when they do.

**Watch item:** clearance fell 34.5 → 22.3 mm, now 2.3 mm above its floor. Part
of how v9 bought reduced airtime was by lifting less. Cadence (5.93) and stride
(0.592 m) are unharmed, so this is not yet chatter, but it is the chatter
direction, and a successor that pushes the same lever harder should be expected
to fail clearance before it delivers support.

**Telemetry check:** the new mid-stance contact-point slip read 0.105 against
the gated whole-phase 0.153 — a 31% transient share, inside the 10–37% range
measured on 2026-08-02. Slip passed on both metrics and remained non-binding,
as predicted when it was added.

**Lesson.** Pricing an unwanted behaviour out works, and buys nothing on its own
if the wanted behaviour shares a budget with it. Flight, single support and
double support are three states summing to one; v8 and v9 each moved mass
between them with a scalar and neither could reach a corner that constrains two
of the three at once. The measurement that mattered was not either failing
metric but the window-occupancy ratio, which separated "the policy does not know
where the overlap should be" (false — 95% correctly placed) from "the overlap
the schedule asks for is shorter than what the gate demands the policy realize"
(true).

**Artifacts:**
`outputs/walker_reward_v9_flight_cost/seed_21/checkpoints/final_model.zip` and
`outputs/walker_reward_v9_flight_cost/seed_21/eval_gate_v2.json`.

## 2026-08-02 — Gate v2 Support Floors Re-Grounded Absolutely (No Value Changed)

**Question:** the next candidate changes `stance_duty`, and gate v2's 10%
double-support floor was derived as *half the nominal overlap implied by
`stance_duty: 0.6`*. Does the floor move with the schedule?

**No, and the reason is the gate's purpose.** A floor indexed to the commanded
schedule can be passed by commanding a shorter one, so the gate would certify
progressively less walking as the schedule relaxed. That is the same failure
class as gate v1, which calibrated from whatever the system happened to produce.
What a promotion gate must certify is walking; the phase schedule is a training
device, not the definition of the target.

**Re-grounding, absolute:** human walking holds roughly 20% double support per
cycle at comfortable speed, declining toward zero at the walk-run transition.
The 10% floor is half that — conservative for a 65.9 kg Atlas-class biped at the
~1.0 m/s target, and still strictly excluding running, which has no double
support by definition. The 10% flight ceiling was never a schedule quantity at
all: walking has no flight phase, and the allowance exists for raw-contact
estimation noise.

**What changed:** the justifying comment, and a regression test
(`test_walker_support_thresholds_do_not_track_the_commanded_schedule`) that pins
v9's failing 8.88% as failing at commanded duties from 0.55 to 0.9, with a
passing 15% control so the assertion is not vacuous. **`gate_version` stays 2
and no threshold value moved**, so this is not a gate change with a candidate in
view — the accept/reject set is identical. Only the argument for it is now one
that survives a schedule change.

**Lesson.** A threshold derived from a configurable quantity inherits that
quantity's mutability whether or not anyone intended it. The derivation read as
principled ("half the nominal overlap") and was in fact parameterised by a knob
the next experiment was about to turn. When freezing a gate, check whether its
justification names anything a future candidate can edit.

## 2026-08-02 — V10 Stance Duty: Lengthen The Window The Policy Already Hits

**Question:** v9 places 95.0% of its double support correctly inside the
scheduled window but fills only 42.2% of that window. Does a longer window
produce more realized overlap, given the policy already tracks the schedule at
86.9%?

**Why not the reward weight.** Raising
`phase_double_support_reward_weight` pays more for a window the policy already
hits accurately — the deficit measured is duration, not aim — and v8 measured
what that pressure costs: +6.7 points of double support bought with +9.1 points
of flight. The mechanism-matched lever is the window.

**Constants:** seed 21, 10M steps from scratch, 0.15→1.0 m/s ramp, 0.55 s cycle,
gate v2, and every reward weight including `flight_penalty_weight: 1.0`.
**Only intervention:** `gait.stance_duty: 0.6 → 0.65`. The config diff against
v9 is two lines.

Nominal double support is `2 * stance_duty - 1`, verified numerically against
`_expected_stance`: 20% → 30%, with nominal flight still exactly 0.0000. At
v9's observed 42.2% window occupancy that projects to 12.7% realized overlap,
clearing the 10% floor with margin.

**Declared risk:** the swing window falls from 0.220 s to 0.193 s of the cycle.
v9's clearance was already 22.3 mm against a 20 mm floor, so clearance is the
expected failure mode if this misses. The correct response to that outcome is
*not* to lengthen `cycle_period_s` in the same run; that would be a second
intervention and would make the result unattributable.

**Status:** launched, not yet evaluated. Seeds 22–23, transfer, and visual
approval remain blocked pending all thirteen stochastic criteria.

## 2026-08-02 — V10 Stance Duty: Rejected, And The Projection Was Wrong

**Question:** v9 filled 42.2% of the 20% overlap window its schedule provided.
Does a 30% window (`stance_duty: 0.65`) yield proportionally more realized
double support?

**Result:** no. Rejected at **10/13**, worse than v9's 11/13. Training completed
at 10,002,432 steps; 20 stochastic and 20 deterministic episodes.

| stochastic metric | v9 | **v10** | gate |
|---|---:|---:|---:|
| double support | 8.88% | **10.00%** | ≥10% |
| flight | 12.39% | **27.38%** | ≤10% |
| single support | 78.73% | **62.62%** | — |
| foot clearance | 22.3 mm | **12.4 mm** | ≥20 mm |
| contact-point slip | 0.153 | **0.327** | ≤0.18 |
| stride | 0.592 m | **0.396 m** | ≥0.10 |
| displacement | 14.29 m | **11.36 m** | ≥1 m |
| falls | 0% | **20%** | ≤30% |
| episode steps | 800 | **691.9** | ≥600 |
| schedule match | 0.869 | **0.704** | — |

**The projection failed at its central assumption.** It treated the 42.2%
window occupancy as a property of the policy that would carry to a longer
window, projecting 12.7% realized overlap. Measured occupancy came out at
**22.4%**, and the absolute in-window double support *fell*, 8.44% → 6.73%.
Occupancy is not conserved across window lengths; it is not a policy constant.

**The double-support "pass" is not evidence the mechanism worked.** It reads
exactly 0.1000, on the threshold, and the scheduled share of that overlap fell
from **95.0% to 67.3%** — a third of v10's double support now occurs at phases
the schedule does not ask for. v9 produced less overlap but placed essentially
all of it correctly; v10 produces marginally more and places a third of it
wrong. On the quantity the intervention targeted — scheduled overlap — v10 is
strictly worse.

**Mechanism: stance duty and swing time are one knob, not two.** Commanding 65%
stance leaves 35% of the 0.55 s cycle for swing (0.193 s, from 0.220 s). The
declared risk was clearance, and clearance is what broke: the bilateral
clearance reward collapsed 0.143 → 0.058 per step and mean clearance halved to
12.4 mm, less than two thirds of its floor. Without lift there is no real step,
so single support collapsed 16.1 points — and that mass went to **flight**, not
to double support. Stride fell a third, slip doubled, and the policy lost the
schedule it had been tracking at 86.9%.

**Decision:** reject. Do not raise `stance_duty` further, and do not rescue this
by lengthening `cycle_period_s` — the pre-declared reason for not compensating
in the same run was to keep the result attributable, and it is: the failure is
specifically that swing time bought the overlap window.

**Lesson.** A ratio measured under one setting of a parameter is not a constant
to extrapolate along that parameter. The projection table in the v10 config
header multiplied a measured occupancy by three different nominal overlaps as
though the two factors were independent; they are coupled through swing time,
and the coupling is strong enough to reverse the sign of the intended effect.
The tell was available beforehand — the same header declared clearance as the
expected failure mode, which is to say it already contained the reason the
arithmetic could not hold. When a projection and its own risk note disagree,
the risk note is describing a coupling the projection omitted.

**Artifacts:**
`outputs/walker_reward_v10_stance_duty/seed_21/checkpoints/final_model.zip` and
`outputs/walker_reward_v10_stance_duty/seed_21/eval_gate_v2.json`.

**Standing after v10:** v9 remains the best candidate at 11/13, failing only the
two support criteria. Three levers have now been tried against them —
scheduled-overlap reward (v8), flight price (v9), and window length (v10) — and
each moved mass between flight, single and double support without reaching the
corner where both pass. No further single-scalar variant on this reward
structure is justified without a new measured mechanism.

## 2026-08-02 — Telemetry: What Terminates Double Support (No Training Budget)

**Question:** three levers have failed the same two support criteria — the
scheduled-overlap reward (v8), the flight price (v9), and the window length
(v10). Before a fourth, what actually ends a double-support interval? Following
the 2026-07-26 precedent, where foot-placement and stance-phase analyses
rejected reward candidates before training.

**Method:** the v9 checkpoint (best at 11/13), 20 stochastic episodes, 791
double-support (DS) intervals. A DS interval opens when the *leading* foot lands
while the *trailing* foot is still loaded, and closes when either foot breaks
contact. PD `POSITION_CONTROL` means the commanded target is known exactly, and
`getJointStates` exposes `appliedJointMotorTorque` against a known per-joint cap,
so "the policy lifted the foot" and "the leg could not hold" are separable.

### Finding 1 — double support is a 1–3 step impact transient, not a phase

| interval length | share |
|---|---:|
| 1 control step | 48.3% |
| 2 control steps | 43.2% |
| 3 control steps | 8.5% |
| 4+ | **0.0%** |

Not one interval in 791 exceeded 3 control steps (0.05 s). **The scheduled
overlap window is 6.6 control steps (0.11 s).** The
`phase_double_support_reward_weight` term has therefore been paying for a state
the gait cannot occupy, by a factor of 4.4 — this is a defect in the schedule's
target, not in the weight, and it explains why v8 and v10 both failed to convert
window length into realized overlap.

### Finding 2 — but the *gate* floor is nearly in reach

DS fraction decomposes exactly as cadence x mean interval length:

    5.931 touchdowns/100 steps  x  1.497 steps  =  8.88%
    to clear the 10% floor:        1.686 steps  =  +0.189 steps (+12.6%)

The 10% gate needs about a fifth of a control step more mean overlap. The 6.6
step schedule and the 10% gate are two very different asks, and they had been
treated as one.

### Finding 3 — the handover is an impact, and the trailing ankle absorbs it

Vertical load at DS onset, against 646 N body weight:

| | force | x BW |
|---|---:|---:|
| leading foot | 894.7 N | 1.38 |
| trailing foot | 486.7 N | 0.75 |
| **combined** | **1381.4 N** | **2.14** |

A walk shares weight across the overlap; this lands on it. Consistently, 19.8%
of intervals end with the *leading* foot lifting — the new foot bounces rather
than accepting load — against 70.7% ending normally with the trailing foot.

Torque saturation is **specific to double support, and specific to the ankle**:

| joint | cap | DS mean | SS mean | >99% cap in DS | >99% cap in SS |
|---|---:|---:|---:|---:|---:|
| hip | 190 | 0.726 | 0.748 | 47.6% | 46.4% |
| knee | 220 | 0.499 | 0.398 | 9.0% | 0.6% |
| **ankle** | **100** | **0.647** | **0.550** | **58.9%** | **21.2%** |

The hip is near-saturated in *all* stance, so its saturation says nothing about
double support. The ankle nearly triples its at-cap rate in DS.

### Finding 4 — two headline correlations were duration artifacts

`target_delta` and `sat_max` accumulate over an interval, so they rise with
duration mechanically. Recomputed per step, both reverse sign:

| | r vs duration (total) | r vs duration (per step) |
|---|---:|---:|
| `target_delta` | +0.633 | **−0.272** |
| `sat_max` | +0.376 | **−0.567** |

Per step, **more saturation means shorter double support** — the direction
consistent with a leg being unable to hold, and the opposite of what the raw
numbers suggested. Recorded because the raw +0.633 would have supported a
"policy chooses to lift" reading that the control refutes.

### Finding 5 — the torque probe is inconclusive, and its caveat was pre-declared

Running the unchanged v9 policy with more torque headroom (no retraining) was
meant to be decisive. It is not:

| variant | intervals | mean steps | max | DS fraction |
|---|---:|---:|---:|---:|
| baseline | 419 | 1.59 | 3 | 9.02% |
| ankle x2 | 158 | 1.18 | 3 | 3.42% |
| all x2 | 56 | 1.16 | 2 | 3.47% |
| all x4 | 17 | 1.00 | 1 | 3.32% |

More torque made double support *rarer and shorter*. But the interval count
collapsing 419 → 17 shows the gait itself fell apart: the policy was trained
under the baseline caps and is off-distribution under changed dynamics, so this
cannot separate "torque is not the constraint" from "this policy cannot exploit
extra torque." The script declared in advance that a null result would be weak
evidence, and it is. **It does not license raising the caps, and does not
exonerate them either.** A real test requires retraining.

### Decision

The reward *weights* are exhausted — no further single-scalar variant on this
structure is justified, which is what this study was run to determine. Two
distinct defects are now separable:

1. **The schedule asks for a state that cannot occur.** 6.6 steps of commanded
   overlap against a 3-step mechanical ceiling. Any reward keyed to that window
   is paying for an unreachable target.
2. **The handover is an impact.** 2.14x body weight at onset, a leading foot
   that bounces one time in five, and a trailing ankle at its cap three times as
   often as in single support.

Actuation is now the leading hypothesis, but it is a hypothesis, not a result.
The ankle's 100 N·m cap is sourced from `atlas_v3.urdf` effort values, so raising
it is a **modelling change, not a tuning knob**, and should be argued as such
rather than slipped in as a weight.

**Lesson.** The gate floor (10%) and the schedule's window (20% of cycle,
6.6 steps) had been treated as one target for three experiments. They differ by
4.4x in what they demand, and only one of them is reachable. Decomposing the
metric (`DS fraction = cadence x interval length`) showed the gate needs +0.19
control steps while the schedule needs +3.6 — a distinction no amount of weight
tuning would have surfaced, because both were being pushed by the same term.

**Artifacts:**
`outputs/walker_reward_v9_flight_cost/seed_21/double_support_telemetry.json`.

## 2026-08-02 — Telemetry Round 2: The Impact Hypothesis Dies, And The Corner May Not Exist

Three follow-up studies, no training budget. The first was proposed at the end
of the previous entry and is refuted by its own measurement.

### Study A — landing impact does not predict anything about double support

The previous entry found the handover is an impact (2.14x body weight) and
suggested a landing-softness reward. Per the 2026-07-26 precedent, a reward may
not be proposed from a predictor that has not been shown to predict, so this
tested it first. Interval length takes only three values, so Pearson r is the
wrong tool; both outcomes were treated as classification with rank-based AUC.

**Outcome A, does the interval last >=2 steps:**

| predictor | AUC |
|---|---:|
| `land_vz` (foot vertical velocity at touchdown) | 0.434 |
| `combined_bw` (the 2.14x impact itself) | 0.444 |
| `lead_force` | 0.483 |
| `pelvis_z` | **0.295** |
| `land_vx` | **0.698** |

Landing impact sits in the null band, and `land_vz` points the *wrong way* —
softer landings gave marginally shorter intervals. **A landing-softness reward
is not justified.** What does discriminate is pelvis height (lower pelvis, longer
overlap) and horizontal foot speed at touchdown.

**Outcome B, does the leading foot bounce (19.8% of intervals):**

| predictor | AUC |
|---|---:|
| `trail_force` | 0.776 |
| `pelvis_vz` | 0.743 |
| `trail_ankle_sat` | 0.733 |
| `land_vz` | 0.479 |

Bounces are predicted by the *trailing* leg's state, not by how hard the leading
foot lands. A bounce is a failure to unload, not a hard landing.

### Study B — the decomposition, corrected

The previous entry wrote `DS fraction = cadence x mean interval`. That is wrong
as stated: cadence (debounced alternating touchdowns, both feet) is not the same
quantity as the double-support interval *rate*, and the raw touchdown count
over-reports by ~3.5x. The identity that actually holds is

    DS fraction = (DS interval rate) x (mean interval length)

verified to err 0.0000 on all four checkpoints:

| run | duty | DS rate /100 | mean interval | DS fraction |
|---|---:|---:|---:|---:|
| v7corr | 0.60 | 4.06 | 1.21 | 4.90% |
| v8 | 0.60 | 6.19 | 1.98 | 12.24% |
| v9 | 0.60 | 5.67 | 1.59 | 9.02% |
| v10 | 0.65 | 7.92 | 1.35 | 10.73% |

Interval length does not shrink as rate rises (r = +0.222, n=4), so the rate
lever survives its screen — the failure mode that killed v10 does not obviously
repeat. But n=4 makes this a screen for a disqualifying negative, not a
confirmation.

### Study C — no policy has ever reached the corner

Twenty-four existing checkpoints (every 2M steps across four runs), six
stochastic episodes each, scored only on the two failing criteria:

**Zero of 24 satisfy flight <=10% and double support >=10% together.**

And the two are *positively* coupled across mature policies — the correlation
strengthens as training matures, so it is not an immaturity artifact:

| subset | r(DS, flight) | n |
|---|---:|---:|
| all checkpoints | +0.442 | 24 |
| mature (>=8M) | +0.554 | 12 |
| final only | **+0.636** | 8 |

Repeat measurements of the same policy (`ppo_model_9999960` vs `final_model`)
differ by 0.08–0.59pp on DS and 0.16–0.97pp on flight, so the between-run
spread is well outside sampling noise.

Closest five, by total violation (DS shortfall + flight excess):

| run | DS | flight | violation |
|---|---:|---:|---:|
| **v9** | 9.54% | 11.77% | **0.0223** |
| v9 | 9.14% | 12.74% | 0.0360 |
| v7corr | 4.96% | 12.23% | 0.0727 |
| v7corr | 5.04% | 13.10% | 0.0806 |
| v8 | 12.42% | 20.17% | 0.1017 |

### Decision

Reaching the gate requires moving double support up and flight down —
*opposite* directions — against a measured positive coupling among every
converged policy this project has produced. Combined with the mechanical
interval ceiling (1–3 control steps against a 6.6-step commanded window) and the
DS-specific ankle saturation, the evidence says the corner is not reachable by
reward shaping in this actuation regime. **Stop tuning rewards here.**

The honest limit of that claim: eight mature policies from four closely related
reward structures is not the space of all gaits, and a correlation over n=8 is
not a proof of infeasibility. It is strong enough to stop spending 28-minute
runs on reward weights, not strong enough to declare the problem impossible.

**Next is a modelling question, not a tuning one.** The ankle's 100 N·m cap is
sourced from `atlas_v3.urdf` effort values; it is the joint that saturates
specifically during double support (58.9% at cap vs 21.2% in single support) and
that predicts bounces (AUC 0.733). Raising it changes what robot is being
claimed, so it needs an explicit decision rather than a config edit — and the
no-retrain probe already showed it cannot be tested without retraining.

**Lesson.** Two entries ago the recommendation was a landing-softness reward,
argued from a real measurement (2.14x body weight) that turned out to predict
nothing. A number can be true, striking, and causally irrelevant at once. The
precedent that saved the run was procedural, not clever: measure that the
predictor predicts before proposing the reward.

**Artifacts:** scratchpad scripts; the 24-checkpoint sweep is reproducible from
existing outputs.

## 2026-08-02 — Gate v3: The Objective Changed, So The Support Criteria Did

**This is a gate change made with a candidate in view, which this project
otherwise forbids.** The justification has to be stated plainly enough that a
later reader can reject it if it does not hold.

**What changed:** the project owner redefined the target. The goal is a
framework that demonstrably improves a locomotion policy over iterations, and
running is an acceptable — explicitly preferred — outcome. Gate v2's
`min_double_support_fraction: 0.10` and `max_flight_fraction: 0.10` encoded
*walking specifically*. They were correct for the old objective and are the
wrong question for the new one.

**What did not change it:** v9's numbers. No v3 threshold was placed using
them. The test of whether this was a goalpost move is whether the new
thresholds were derived from degenerate cases or from a candidate — they were
derived from degenerate cases, and the anti-degenerate work the old criteria did
is preserved rather than dropped.

**The three changes:**

1. `min_double_support_fraction` **removed, not relaxed.** Running has zero
   double support by definition, so any floor forbids the preferred gait.
2. `max_flight_fraction` **0.10 → 0.60**, its pre-v2 value, which had an
   independent derivation the v2 tightening discarded: open-loop scripted gaits
   at this robot's own settings measured 17–56% flight. Human sprinting peaks
   near 40%. The ceiling now excludes only degenerate leaping.
3. `max_contact_duty_imbalance: 0.15` **added**, and this is what actually
   replaces the removed criteria. The 2026-07-28 entry recorded the real hazard:
   an asymmetric bound (right foot in contact 27.3% vs left 15.0%, 57.6% flight,
   0.1% double support) that a naive gate scores perfectly. The old flight
   ceiling excluded it only as a side effect, using flight as a proxy for
   asymmetry. A symmetry criterion targets the defect directly, so the flight
   ceiling no longer has to do two jobs.

   Threshold derived independently of any candidate: 0.15 means one leg carries
   57.5% of stance and the other 42.5% — the line past which a gait is
   meaningfully one-legged. Validity check, *not* calibration: the 2026-07-28
   bound measures 0.289 and is excluded; healthy alternating gaits measure ~0.02.

Cadence band, stride, clearance, progress, slip, survival, fall rate,
displacement, peak height and same-foot-sequence limits are **all unchanged**.
Every guard against contact chatter, self-launching, standing and sliding is
retained; the loosening is confined to the two walk-specific criteria.

`gait_contact_duty_imbalance` is added as an episode metric (the per-step value
already existed for the reward) and propagates to diagnostics and TensorBoard.

**Regression tests:** the frozen-contract test now asserts
`min_double_support_fraction` is *absent* rather than any value; the bound test
asserts the bound passes the loosened flight ceiling and is rejected by symmetry
instead, so symmetry is load-bearing rather than incidental; a new test asserts
a symmetric gait with 35% flight **passes**, guarding the other direction so the
loosening is not cosmetic; and the schedule-independence test was rewritten
against the surviving criteria with a passing control so it is not vacuous.

**Result:** v9 seed 21 scores **13/13 under gate v3** — 800 steps, 0% falls,
14.29 m at 1.072 m/s, 0.592 m strides, symmetry 0.0214 against the 0.15 limit.
Double support is 8.88% and is no longer gated.

**Lesson, and it is about process rather than gait.** Four training runs and two
telemetry rounds went into the 10%/10% corner. Gate v2's support criteria were
introduced on 2026-07-29 as a schedule-derived detail, re-grounded on 2026-08-02
in human walking biomechanics — each step locally defensible, and the net effect
was a stricter target than the plan's own stated objective ("repeatable
alternating-foot locomotion"), which v9 had already met. The failure was never
asking whether the gate still encoded the goal. A gate is a proxy; proxies drift
from objectives silently, and nothing in the measure-first discipline catches it,
because that discipline is about how to answer the question rather than whether
it is the right one.

## 2026-08-02 — V9 Across Three Seeds: 2/3 Pass Gate v3

**Question:** is v9's result robust across seeds, or was seed 21 lucky? Seeds
22 and 23 trained on the identical config — `--seed` overrides only the global
and environment seed, and `num_envs` was held at 24 (running the two in parallel
would have oversubscribed the machine, and halving workers would have changed
the effective batch size and broken comparability).

| seed | steps | falls | disp | speed | stride | clearance | slip (gated) | flight | symmetry | gate v3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 21 | 800 | 0% | 14.29 m | 1.07 m/s | 0.592 | 22.3 mm | 0.1534 | 12.4% | 0.021 | **13/13 PASS** |
| 22 | 800 | 0% | 13.99 m | 1.05 m/s | 0.580 | 27.8 mm | 0.1570 | 13.7% | 0.027 | **13/13 PASS** |
| 23 | 754 | 10% | 12.90 m | 1.03 m/s | 0.531 | 33.8 mm | 0.2178 | 21.9% | 0.023 | 12/13 fail: slip |

**All three learned to run.** Every seed reaches ~1.0 m/s over 12.9–14.3 m with
0.53–0.59 m strides, symmetric legs (0.021–0.027 against a 0.15 limit) and
proper alternation. Seed variation shows up as a spread in *style* — seed 23
lifts higher (33.8 mm vs 22.3) and flies more (21.9% vs 12.4%) — not as the
difference between locomotion and collapse. This is the first three-seed
walker result in the project where every seed produced real locomotion.

**Seed 23's single failure lands on the one threshold documented as
uncalibrated.** `max_slip_speed: 0.18` was frozen when `target_velocity` was
0.15 m/s and is applied here at 1.03 m/s, on a metric that averages the whole
contact phase including the landing transient. Seed 23's breakdown:

| | value |
|---|---:|
| whole-phase contact-point slip (gated) | 0.2178 — fails by +0.0378 |
| mid-stance contact-point slip (telemetry) | 0.1283 — would pass with 0.0517 spare |
| landing-transient share | **41.1%** |

That is the largest transient share measured (previous range 10–37%), and it is
coherent with seed 23's own gait: it lifts higher and flies more, so it lands
harder, so the transient is larger. The metric penalises it for how it lands,
not for how much its feet slide.

**Decision: the threshold is NOT moved and the metric is NOT swapped.** The
telemetry added on 2026-08-02 exists precisely for this case, and it says seed
23 would pass on the uncontaminated measure — which is exactly why changing
either now would be indefensible. The 2026-07-28 precedent is explicit: a
corrected instrument is not swapped into a live gate with a candidate in view,
because the question of what else lands on the passing side has to be asked
separately from the question of what is physically correct. Seed 23 *is* the
candidate in view. Gate v3 was changed hours earlier on the strength of an
objective change from the project owner; doing it again on the strength of an
inconvenient result is a different act entirely, and the distinction is the
whole reason the first one was defensible.

**Open, and it should be settled without a candidate in view:** switching the
slip gate from `gait_contact_point_slip_speed_mean` to
`gait_mid_stance_contact_point_slip_speed_mean` while keeping 0.18 is arguable
on physical grounds alone — the transient is not sliding, it scales with speed,
and its share varies 10–41% by gait, so the gated metric can separate two
policies on landing dynamics rather than on sliding. That is the same shape of
argument as the 2026-07-28 link-origin → contact-point correction, which was
made without changing the number. It is left open here deliberately.

**Result:** v9 is a three-seed result with 2/3 passing. Seeds 21 and 22 are
eligible for visual review and transfer evaluation.

## 2026-08-02 — Gate v4, Visual Review, And Transfer: Flat Is Solved, Terrain Is Not

### Gate v4 — the slip criterion moves to the mid-stance metric

**Decided by the project owner**, on the grounds that the whole-phase metric was
not providing the feedback it was introduced to provide. That reading is
correct: the criterion exists to catch a foot *sliding under load*, and 10–41%
of what it reported was a landing transient — the foot still moving as it
arrives, which is not sliding, scales with locomotion speed, and varies by gait.

`max_slip_speed` stays **0.18**. Correcting an instrument is not a reason to
move a threshold, exactly as in the 2026-07-28 link-origin → contact-point
change. Against the corrected metric 0.18 is a *stricter* test of sliding, since
the contamination only ever inflated the reported value.

**Disclosure and the audit the precedent requires.** This change moves a
candidate across a line: v9 seed 23 goes 12/13 → 13/13. The 2026-07-28
requirement is not "never switch" but "when correcting an instrument moves a
candidate across a line, check what else is now on the passing side". Every
checkpoint with a stored evaluation was re-scored under both metrics:

| run | whole-phase | mid-stance | other criteria | v3 | v4 |
|---|---:|---:|---|---|---|
| v10 seed 21 | 0.3269 | 0.2701 | **fail** | . | . |
| v9 seed 21 | 0.1534 | 0.1053 | pass | P | P |
| v9 seed 22 | 0.1570 | 0.0890 | pass | P | P |
| v9 seed 23 | 0.2178 | 0.1283 | pass | . | **P** |

**Exactly one crossing, and it is the intended one.** v10 still fails — it also
fails clearance, so slip was never its deciding criterion. v7corr (0.1206 /
0.0764) and v8 (0.1667 / 0.1503) pass under either metric and do not move.

### Visual review — all three seeds pass

Rendered as GIFs via imageio/PIL rather than `RecordVideo`, which needs MoviePy;
adding a dependency for a review artifact is not warranted.

Seed 21/22/23 flat rollouts all ran the full 800 steps for 13.97–14.19 m with
peak torso height 0.683–0.693 against the 1.0 limit. A cropped filmstrip over
one full gait cycle shows legs scissoring in clear alternation, knees flexing,
feet lifting clear of the ground, torso upright, and arms counter-swinging.
**None of the tracked exploits is present**: no prone dragging, no
sliding-without-stepping, no launching, no same-foot tapping.

### Transfer suite — flat 3/3, everything else 0/3

Twenty stochastic episodes per terrain per seed:

| terrain | seeds passing | steps | falls | displacement |
|---|---|---:|---:|---:|
| flat | **3/3** | 754–800 | 0–10% | 12.90–14.29 m |
| uneven (2.5 cm) | 0/3 | 170–178 | 95–100% | 1.70–1.95 m |
| obstacles (10 cm) | 0/3 | 111–122 | 100% | 1.62–1.65 m |
| push recovery (180 N) | 0/3 | 404–486 | 100% | 6.65–7.82 m |

**This is a flat-ground specialist, and that is the expected result rather than
a defect**: every one of these policies trained with `terrain.preset: flat` and
no curriculum, so this measures zero-shot transfer, which nothing in training
asked for. It is a baseline, and now a measured one.

Push recovery is the informative case. Per-push recovery is **98.2–100%** and
episodes last 404–486 steps covering 6.6–7.8 m, yet 100% of episodes eventually
end in a fall. The policy absorbs individual 180 N pushes and survives them by
well over the 30-step recovery window; what it lacks is the ability to keep
doing so indefinitely. That is a much smaller gap than uneven or obstacles,
where it falls within ~2–3 seconds.

**Standing:** v9 is a three-seed, visually-approved, flat-terrain result at
13/13 on gate v4. The next iteration target is terrain, not reward: train with
the existing terrain curriculum rather than flat-only, and re-run this same
suite. That is a direct test of the stated objective — a framework that improves
a policy over iterations — with a measured 0/3 baseline to beat.

**Artifacts:** `outputs/walker_reward_v9_flight_cost/seed_*/transfer_suite.json`,
`outputs/walker_reward_v9_flight_cost/visual_review/`.
