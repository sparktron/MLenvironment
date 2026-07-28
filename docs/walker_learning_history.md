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
