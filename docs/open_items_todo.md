# Development Roadmap

Last updated: 2026-08-02

This is the active roadmap. Historical review findings have been folded into
the completed summary below so completed work is not presented as pending.
The append-only
[`walker_learning_history.md`](walker_learning_history.md) is the canonical
teaching narrative and decision ledger for walker experiments; new studies
must record both successful changes and failed hypotheses there.

## Priority 1: Runtime And GUI Performance

- All tracked Priority 1 items are complete (see Completed Foundations).

## Priority 2: Evaluation And Operations

- All tracked Priority 2 items are complete (see Completed Foundations).

## Priority 3: Learning Quality And Features

- (2026-08-02) **OBJECTIVE RESTATED: the goal is a framework that demonstrably
  improves a locomotion policy over iterations. Running is acceptable and
  preferred over walking.** Everything below dated 2026-07-29 to 2026-08-02 that
  chases the 10% double-support floor and 10% flight ceiling was pursuing
  *walking specifically*, which is no longer the target. Read those entries as
  history, not as open work.
- (2026-08-02) **Gate v3 supersedes v2.** `min_double_support_fraction` is
  REMOVED (running has none by definition, so any floor forbids the preferred
  gait); `max_flight_fraction` returns to **0.60** (its pre-v2 value, derived
  from open-loop commanded gaits measuring 17–56%); and
  `max_contact_duty_imbalance: 0.15` is ADDED to do the anti-degenerate work the
  support criteria were doing implicitly — it excludes the 2026-07-28 asymmetric
  bound (0.289) directly rather than using flight as a proxy for asymmetry.
  0.15 means one leg carries 57.5% of stance; healthy gaits measure ~0.02. All
  chatter, launch, standing and sliding guards are unchanged. This was a gate
  change made with a candidate in view, permitted only because the *objective*
  changed; no threshold was placed using v9's numbers.
- (2026-08-02) **v9 is a three-seed result: 2/3 pass gate v3, and all three
  learned to run.** Identical config, `--seed` only.

  | seed | steps | falls | disp | speed | clearance | slip | flight | gate v3 |
  |---:|---:|---:|---:|---:|---:|---:|---:|---|
  | 21 | 800 | 0% | 14.29 m | 1.07 m/s | 22.3 mm | 0.1534 | 12.4% | **13/13 PASS** |
  | 22 | 800 | 0% | 13.99 m | 1.05 m/s | 27.8 mm | 0.1570 | 13.7% | **13/13 PASS** |
  | 23 | 754 | 10% | 12.90 m | 1.03 m/s | 33.8 mm | 0.2178 | 21.9% | 12/13 fail: slip |

  Seed variation is a spread in style (seed 23 lifts higher and flies more), not
  the difference between locomotion and collapse. All three are symmetric
  (0.021–0.027 vs a 0.15 limit). First three-seed walker result where every seed
  produced real locomotion. **Seeds 21 and 22 are eligible for visual review and
  transfer evaluation.**
- (2026-08-02) **Seed 23 fails only on the slip threshold that is documented as
  uncalibrated, and it was NOT moved.** `max_slip_speed: 0.18` was frozen at a
  0.15 m/s target and is applied at 1.03 m/s to a metric that includes the
  landing transient. Seed 23: whole-phase 0.2178 (fails by 0.0378), mid-stance
  0.1283 (would pass with 0.0517 spare), transient share **41.1%** — the largest
  measured, and coherent with its higher-lifting, higher-flight gait. Neither the
  threshold nor the metric was changed: the 2026-07-28 precedent forbids swapping
  a corrected instrument into a live gate with a candidate in view, and seed 23
  is that candidate. **Open question to settle with no candidate in view:**
  switching the slip gate to `gait_mid_stance_contact_point_slip_speed_mean`
  while keeping 0.18 is arguable on physical grounds alone (the transient is not
  sliding, scales with speed, and varies 10–41% by gait), the same shape of
  argument as the 2026-07-28 link-origin → contact-point correction.

- (2026-08-02) **`walker_reward_v9_flight_cost` completed and is rejected at
  11/13.** The single intervention was `flight_penalty_weight: 0.5 → 1.0`, set
  at ~1.6x the 0.64 break-even weight computed from v8's reward economics. (A
  proposed "unscheduled flight penalty" was rejected before implementation as a
  duplicate: at `stance_duty: 0.6` the reference schedule expects a foot in
  stance at every phase — measured expected flight fraction exactly 0.0000 — so
  unscheduled flight *is* flight, already charged unconditionally.) Stochastic
  results: flight 22.04% → **12.39%**, but double support 11.59% → **8.88%**;
  both support criteria fail. Displacement 14.29 m, 0% falls, slip 0.153 m/s,
  stride 0.592 m, cadence 5.93 all pass. Deterministic scored 12/13 (flight
  9.77% clears; double support 9.33% is the only miss), but the gate evaluates
  stochastic. v8 and v9 now **bracket** the target: a scalar on either term
  moves both, because flight, single support and double support are three
  states summing to one. Seeds 22–23, transfer, and visual approval remain
  blocked.
- (2026-08-02) **`walker_reward_v10_stance_duty` completed and is rejected at
  10/13 — worse than v9, and the projection behind it was wrong.** The single
  intervention was `gait.stance_duty: 0.6 → 0.65`, taking nominal overlap 20% →
  30%. It projected 12.7% realized double support by assuming v9's 42.2% window
  occupancy was a policy property that would carry to a longer window. Measured
  occupancy was **22.4%**, and absolute in-window overlap *fell* (8.44% →
  6.73%). Double support reads exactly 0.1000 (a bare pass), but the scheduled
  share of it dropped 95.0% → 67.3%, so on the targeted quantity — correctly
  scheduled overlap — v10 is strictly worse. Everything else regressed: flight
  27.38%, clearance 12.4 mm (fail), slip 0.327 m/s (fail), stride 0.396 m, falls
  20%, schedule match 0.704. **Mechanism: stance duty and swing time are one
  knob.** 65% stance leaves 0.193 s of swing; the bilateral clearance reward
  collapsed 0.143 → 0.058 and clearance halved, so single support fell 16.1
  points and that mass went to flight, not double support. Do not raise
  `stance_duty` further and do not rescue it with `cycle_period_s`.
- (2026-08-02) **STOP TUNING REWARDS FOR THE SUPPORT CRITERIA. Zero of 24
  existing checkpoints ever satisfied flight ≤10% and double support ≥10%
  together, and the two are positively coupled among converged policies**
  (r = +0.442 over all 24, +0.554 at ≥8M, **+0.636 over the 8 final
  checkpoints** — it strengthens with maturity, so it is not a training
  artifact). Reaching the gate needs the two moved in opposite directions,
  against that coupling. Repeat measurements of the same policy differ by only
  0.08–0.59pp (DS) and 0.16–0.97pp (flight), so between-run spread is real.
  Closest is **v9 at violation 0.0223** (DS 9.54%, flight 11.77%). Honest limit:
  8 mature policies from 4 related reward structures is not the space of all
  gaits, so this justifies stopping 28-minute reward runs, not declaring the
  problem impossible.
- (2026-08-02) **A landing-softness reward is NOT justified — the hypothesis was
  refuted by its own measurement.** Following the 2026-07-26 precedent (measure
  that the predictor predicts before proposing the reward), landing impact was
  tested against two outcomes with rank-based AUC, since interval length takes
  only three values and Pearson r is meaningless on it. Landing impact does not
  discriminate interval length: `land_vz` AUC 0.434, `combined_bw` 0.444,
  `lead_force` 0.483 — in the null band, with `land_vz` pointing the wrong way.
  What predicts a bounce is the *trailing* leg (`trail_force` 0.776,
  `trail_ankle_sat` 0.733), not the landing (`land_vz` 0.479): a bounce is
  failed unloading, not a hard landing. What predicts longer intervals is pelvis
  height (AUC 0.295, lower is longer) and horizontal foot speed (0.698).
- (2026-08-02) **Correction to the decomposition recorded earlier the same day.**
  `DS fraction = cadence × mean interval` is wrong: cadence (debounced
  alternating touchdowns) is not the double-support interval *rate*, and a raw
  touchdown count over-reports by ~3.5×. The identity that holds, verified to
  err 0.0000 on four checkpoints, is `DS fraction = DS interval rate × mean
  interval length`. Interval length does not shrink as rate rises (r = +0.222,
  n=4), so the rate lever survives its screen — but n=4 is a screen for a
  disqualifying negative, not a confirmation.
- (2026-08-02) **Double-support telemetry study completed (no training budget);
  it separates two defects that had been treated as one.** 791 DS intervals from
  the v9 checkpoint over 20 stochastic episodes. (1) **DS is a 1–3 control-step
  impact transient** — 48.3% last one step, 43.2% two, 8.5% three, **0% four or
  more**, against a scheduled overlap window of **6.6 steps**. The
  `phase_double_support_reward_weight` term has been paying for a state the gait
  cannot occupy by a factor of 4.4, which explains why v8 and v10 could not
  convert window length into realized overlap. (2) **The gate floor is nearly in
  reach**: `DS fraction = cadence × mean interval` = 5.931/100 × 1.497 = 8.88%,
  and the 10% floor needs 1.686 steps — **+0.19 steps, +12.6%**. The 6.6-step
  schedule and the 10% gate are different asks by 4.4×. (3) **The handover is an
  impact**: 2.14× body weight combined at onset (leading 1.38×, trailing 0.75×),
  19.8% of intervals end with the *leading* foot bouncing off, and trailing-ankle
  torque saturation is DS-specific (58.9% at cap in DS vs 21.2% in single
  support; hip is ~47% in both, so hip saturation is not informative).
  (4) Two headline correlations were duration artifacts and reverse per step:
  `target_delta` +0.633 → −0.272, `sat_max` +0.376 → **−0.567** (more
  saturation → shorter DS). **Conclusion: reward weights are exhausted; do not
  run another single-scalar variant on this structure.**
- (2026-08-02) **The no-retrain torque probe is inconclusive — do not cite it
  either way.** Running the unchanged v9 policy at raised caps made DS rarer and
  shorter (baseline 419 intervals / 9.02% DS → all×4 17 intervals / 3.32%), but
  the interval collapse shows the policy is off-distribution under changed
  dynamics. It cannot separate "torque is not the constraint" from "this policy
  cannot exploit extra torque"; the script pre-declared that a null result would
  be weak. A real actuation test requires **retraining** at a raised cap (~28
  min). Note the ankle's 100 N·m cap is sourced from `atlas_v3.urdf` effort
  values, so raising it is a **modelling change, not a tuning knob**, and must be
  argued as such.
- (2026-08-02) **Standing after v10: v9 is the best candidate at 11/13**,
  failing only double support (8.88%) and flight (12.39%). Three levers have now
  been tried against those two criteria — scheduled-overlap reward (v8), flight
  price (v9), window length (v10) — and each redistributed mass among flight,
  single and double support without reaching the corner where both pass. **No
  further single-scalar variant on this reward structure is justified without a
  new measured mechanism.** The next work should be a telemetry-only study (the
  2026-07-26 precedent, which rejected two reward candidates before training) of
  what actually terminates double support: measure, per cycle, the time between
  trailing-foot loading and trailing-foot liftoff, and what predicts an early
  liftoff. If nothing predicts it, the reward structure is not the constraint and
  the PD/actuation limits should be examined instead.
- (2026-08-02) **The v9 shortfall is window duration, not aiming — superseded in
  part by the v10 result above, which shows a longer window does not convert.** Phase telemetry:
  `reward_phase_contact_mean` 0.869 (86.9% schedule match) and
  `reward_phase_double_support_mean` 0.0422 at weight 0.5, i.e. 8.44% of steps
  are in-window *and* in double support — **95.0% of all the double support the
  policy produces is already correctly scheduled.** The policy fills only
  **42.2% of the 20% overlap window** `stance_duty: 0.6` provides and would need
  50% to clear the 10% floor. Raising the incentive to aim at a window it
  already hits is therefore the wrong lever; lengthening the window is the
  mechanism-matched one. At v9's observed 42% occupancy, `stance_duty: 0.65`
  projects ~12.7% realized double support and `0.7` ~16.9%. **Gate coupling that
  must be decided first:** gate v2's 10% double-support floor was *derived* as
  half the nominal overlap at duty 0.6; changing the duty breaks that derivation
  and turns 10% into an absolute threshold needing its own justification
  (human walking is ~20% double support at comfortable speed, so 10% remains
  conservative). Settle that as a gate-design question before running a duty
  candidate, not after.
- (2026-08-02) **Watch item: clearance is trending toward the chatter
  direction.** v8 34.5 mm → v9 22.3 mm, against a 20 mm floor. Part of how v9
  bought reduced airtime was lifting less. Cadence and stride are unharmed, so
  it is not chatter yet, but a successor pushing the flight penalty harder
  should be expected to fail clearance before it delivers support.
- (2026-08-02) **Slip-ceiling recalibration attempted; threshold deliberately
  unchanged.** The method that calibrated the clearance floor (an open-loop
  scripted gait) does not transfer: clearance is an actuator-capability measure
  that survives a falling rollout, while slip requires sustained balanced
  locomotion at speed, and the only policies that provide it are the candidates
  themselves — calibrating from those is what made gate v1 certify chatter. The
  measurement did find a real defect: the gate metric
  `gait_contact_point_slip_speed` fixed the rotation artifact but still averages
  the whole contact phase, so it carries the landing transient, which scales
  with locomotion speed and is gait-dependent (37% of v7's gated value, 10% of
  v8's). `gait_mid_stance_contact_point_slip_speed` is added as **telemetry
  only**, per the 2026-07-28 precedent; `max_slip_speed` stays 0.18 on the
  whole-phase metric. Both v7 and v8 pass on both metrics, so no past decision
  turned on it, but v8's margin is 6%. Revisit the gate only as a deliberate
  gate-design change, without a candidate in view.
- (2026-07-29) **Walker gait gate v2 is frozen and the corrected v7 rerun is
  rejected.** The gate uses
  `gait_contact_point_slip_speed_mean` with the unchanged 0.18 m/s ceiling,
  requires double support ≥10%, and tightens flight from ≤60% to ≤10%. The
  support limits were derived independently from the configured anti-phase
  reference: `stance_duty: 0.6` produces 20% nominal double support and no
  flight, so the gate retains half the nominal overlap and allows a symmetric
  10% raw-contact tolerance. The from-scratch seed-21 10M rerun proved the
  debounced bilateral-clearance signal works: stochastic clearance improved
  17.6→32.5 mm, true slip 0.163→0.124 m/s, flight 20.66→12.96%, falls 10→0%,
  and reward contribution 0.100→0.218 per step. It passed 10/12 criteria but
  missed double support (4.91% <10%) and flight (12.96% >10%), so no visual
  approval, seeds 22–23, or transfer is allowed. Any next candidate must target
  support occupancy explicitly while retaining gate v2 and the corrected
  bilateral signal; do not rerun v7 again for seed luck. The next isolated
  seed-21 candidate is `walker_reward_v8_support_overlap`: it adds a 0.5
  phase-gated double-support reward, paid only during the reference schedule's
  20% overlap, so standing and in-phase hopping cannot farm it. Its completed
  10M seed-21 evaluation clears the support floor (11.59%) and slip ceiling
  (0.170 m/s), but fails flight (22.04% >10%); reject it for promotion and do
  not run seeds 22–23 or transfer. Any successor must reduce flight without
  sacrificing the recovered scheduled overlap.
- The 2026-07-25 promotion-scale walker and arena matrices completed, but no
  candidate earned behavioral promotion. The report readiness fields are true
  because the seed and budget gates were satisfied; every walker diagnostic
  remained `untrained_equivalent`, and arena matches timed out almost
  universally. Detailed evidence is in
  [`learning_quality_studies.md`](learning_quality_studies.md).
- The stochastic-balance and low-velocity phases now pass their one-seed
  screens. Independent staged seeds 21–23 (100k balance + 200k low velocity)
  averaged 700 stochastic steps, 18.3% falls, and 0.93 m displacement. Only
  seed 22 produced clear forward locomotion (1.60 m), so locomotion remains
  seed-sensitive. The resumable `walker-velocity` matrix completed nine 150k
  continuations. `velocity_sigma_010` ranked first, but only seed 22 passed;
  stronger forward weighting was worse. The next walker work should target
  gait structure or phase coordination rather than more scalar velocity-reward
  pressure. Phase 1 of the
  [`walker gait-coordination plan`](plans/2026-07-26-walker-gait-coordination.md)
  is complete: contact/strike/slip telemetry established three-seed baselines,
  zero-default structural rewards and gait gates are tested, and the four
  variants completed seeds 21–23 × 150k. Slip-only came closest but seed 21
  exceeded the frozen 0.18 m/s slip ceiling; displacement remained only
  1.26–2.00 m across seeds. Renders of slip-only seeds 21 and 22 confirmed an
  upright compact alternating shuffle without an obvious exploit. A seed-21
  50k screen then rejected weight 0.75 on displacement (0.967 m) and weight
  1.0 on slip (0.18065 m/s). Do not advance either weight or loosen the slip
  threshold. The seed-21 50k velocity-ramp screen is rejected: retaining slip
  weight 0.5, the 0.15--0.25 m/s ramp gained only +0.097 m (0.200 m vs 0.104 m
  control), missed the 1 m and 30%-fall gates, and showed no longer stride or
  swing. Do not advance it to seeds 21--23 or add a sustained-swing touchdown
  reward without a new measured mechanism. New qualified-swing telemetry
  measures stride length, swing duration, clearance, and cadence to determine
  whether any gain is longer stepping rather than contact chatter. Keep
  PPO/SAC/TD3 comparisons deferred until the final 3 m transfer gate clears.
- (2026-07-26) The seed-21 50k sustained-swing touchdown screen is rejected.
  Calibration used 3,155 individual stochastic touchdown events from the
  fixed-target control (p75: 50 ms airborne duration, 0.5 mm clearance), then
  retained the 0.15 m/s target, 0.5 slip weight, and every frozen gate. The
  qualified-progress candidate reached 0.260 m versus 0.104 m control
  (+0.157 m), below the required +0.25 m; it also missed the 1 m displacement
  and 0.003 m alternating-touchdown-progress floors. Do not advance to seeds
  21--23, 300k, transfer, or algorithm comparisons.
- (2026-07-26) A telemetry-only seed-21 placement analysis rejects
  foot-placement shaping before training: 578 sustained touchdown pairs had
  a lead-to-next-touchdown-progress correlation of -0.054. Do not create a
  50k foot-placement reward candidate from this non-predictive signal.
- (2026-07-26) A telemetry-only seed-21 stance-phase analysis also rejects a
  reward candidate before training. Across 3,985 completed stances in 20
  episodes, stance advance correlated with displacement (r=0.705), but duration
  correlated negatively (r=-0.492) and slip positively (r=0.308). The required
  low-slip, longer-advance mechanism is not present; do not run a stance reward
  screen.
- (2026-07-26) **Gait gate corrected; it previously certified contact chatter.**
  The frozen "at least 15 alternating touchdowns per 100 steps" floor was
  calibrated from the chattering baseline. At 60 Hz control the rate is
  `3.33 × cycle_frequency_Hz`, so 17–20 per 100 steps is a 5–6 Hz cycle with
  10.9 mm strides and 0.7 mm clearance, and a genuine 1 Hz gait (~3.3 per 100
  steps) would have failed. None of the five prior rejections was decided by
  that floor — they turned on displacement, slip, fall rate, or correlation — but
  the gate never discriminated, and the ranking score's `+ cadence / 100` term
  actively favoured a faster shuffle. The gate is now a cadence band of 1.5–8.5
  plus stride (≥0.10 m), clearance (≥0.02 m), and progress (≥0.05 m) floors.
  Gate v2 supersedes its provisional support/slip terms as recorded above.
  Open-loop measurement showed the actuator lifts a foot 33.7 mm at the study's
  own settings versus the learned 0.7 mm, so control authority is not the
  bottleneck.
- (2026-07-26) **Budget is the other blocker.** Measured throughput is 5,970 fps,
  so the 50k rejection screens cost ~8 s and a three-seed 150k matrix ~75 s,
  while 10M steps costs 28 min. Re-baseline at 5–10M steps per seed before
  running further variant comparisons; deltas at 150k are seed noise. Remaining
  candidate levers, in order: alive-bonus/fall-penalty economics and target
  velocity have now been rebalanced in `walker_low_velocity_candidate`
  (`alive_bonus: 0.125`, `fall_penalty: 2.0`, and a 0.15→1.0 m/s ramp over 3M
  continuation steps). The staged candidate now also reopens exploration with
  `log_std_init: -1.0`, `ent_coef: 1e-3`, a 0.5 action scale, and `gamma:
  0.995`; evaluate the combined schedule before further reward changes.
- (2026-07-26) The seed-21 3M economics/natural-velocity screen completed.
  The 0.15→1.0 m/s ramp improved stochastic displacement from 1.97 m to
  7.70 m, but failed the frozen gait gate: fall rate was 40% (maximum 30%) and
  stance slip was 0.317 m/s (maximum 0.18 m/s). Do not promote the ramp; the
  speed increase needs a mechanism that controls slip and falls.
- (2026-07-26) Re-baseline before any further A/B work: run the single
  `walker_low_velocity_candidate` configuration for 10M steps each on seeds
  21, 22, and 23. Its native training path includes the 0.15→1.0 m/s
  three-million-step ramp; do not introduce another variant until it walks
  reliably across all three seeds.
- (2026-07-27) The three 10M runs completed. At the final 1.0 m/s target,
  deterministic displacement was 9.02–11.26 m and peak height remained below
  0.71 m, confirming real forward locomotion rather than a launch. But stance
  slip was 0.41–0.57 m/s (far above the 0.18 m/s gate) and seed 23's stochastic
  fall rate was 60%. Do not add a faster target. First evaluate slip/recovery
  mechanisms with a fixed baseline, one-seed 1M screens, then three-seed 3M
  confirmation only if every screen meets the existing fall/slip limits.
- (2026-07-27) The slip/recovery evaluation completed. Phase 0 recorded six
  falls across 15 stochastic renders: five recovery failures and one touchdown
  overshoot. The pooled p90 clip was 1.3398 m/s, so touchdown overshoot did not
  justify the conditional damping screen. Seed-21 1M slip weights 0.25 and 0.5
  reduced stochastic slip from 0.500 m/s to 0.348 and 0.290 m/s, but neither
  reached 0.18 m/s. Cadence remained 19–23 alternating touchdowns per 100 steps
  with 6.7–7.5 cm strides, and renders confirmed contact chatter. Do not run
  recovery-noise, Phase 2, transfer, or competing speed/algorithm variants from
  these checkpoints. The next proposal must target longer swing/stride and
  recovery state sensitivity without relaxing the frozen gates.
- (2026-07-27) Corrected the walker observation documentation: coordinate-free
  v2 has 35 values including two foot contacts, not the old position-based
  layout. Previous applied action is now available as an opt-in 10-D v2
  extension (45 values total); enable it only for a new training lineage.
- (2026-07-27) Implemented that new-lineage evaluation as
  `walker-action-memory`. It holds the bounded 0.5 slip intervention and all
  natural-speed/gait gates fixed while comparing matched seed-21 35-D control
  and 45-D previous-action policies from scratch. A 2,400-step smoke completed
  both arms but is deliberately ineligible for advancement. Run the full 10M
  screen; only a metrics-passing and visually approved candidate may proceed
  to seeds 22–23 and transfer.
- Run the feasible-combat arena candidates at a short one-seed budget. Advance
  to three seeds × 30k only when attack hits/damage are nonzero and timeout
  falls below 90%; the report now enforces those thresholds.

## Retained Limitations

- The GUI permits one active in-process training run.
- Shared-policy arena training remains single-process through SuperSuit; use
  self-play for native parallel arena rollouts.
- Observation changes are checkpoint-incompatible by design: walker v1/v2 and
  arena 8D/13D checkpoints must not be mixed.

## Completed Foundations

- (2026-07-31) Separated immutable registry execution identities from reusable
  sweep/morphology output variant IDs. Direct training now generates a fresh
  opaque registry key, GUI orchestration passes its pre-registered key
  explicitly, registry reads expose `variant_id`, and conflicting identity
  reuse fails instead of updating an unrelated run.
- (2026-07-31) Corrected morphology selection so all candidates train before a
  single final tournament under the original common environment config. Every
  compared Elo now comes from the same competitor field and evaluation
  protocol.
- (2026-07-26) Walker evaluation now fans episodes out across process-backed
  vector environments for the `eval` command, quality-study checkpoint
  diagnostics, and periodic best-model selection. `evaluation.workers` caps
  parallelism, auto mode never exceeds the episode count, and strided seed
  streams preserve deterministic, non-overlapping episode coverage.
- (2026-07-26) Established an append-only walker learning history that records
  hypotheses, controlled changes, per-seed evidence, visual checks, failures,
  decisions, and lessons. Future experiments have a standard entry template so
  the eventual successful process can be taught without losing rejected paths.
- (2026-07-26) Completed the gait-coordination 150k matrix and stronger
  stance-slip rejection screen. Slip-only was the closest three-seed variant,
  but not every seed passed. Five-rollout stochastic renders checked its
  strongest and slip-gate-failing seeds. Neither 0.75 nor 1.0 passed the fixed
  seed-21 screen, so no candidate advanced to 300k or transfer testing.
- (2026-07-26) Started the gait-coordination gate. Walker episodes now report
  support fractions, debounced touchdowns, valid alternation, progress per
  alternating strike, stance slip, same-foot sequences, and action deltas.
  Diagnostics and TensorBoard propagate the metrics. Zero-default alternating
  progress and stance-slip reward terms were calibrated from seeds 21–23 and
  completed a four-variant 24-worker smoke plus a 50k rejection screen without
  changing checkpoint shapes.
- (2026-07-26) Added a seed-matched velocity-reward continuation study. It
  resumes balance checkpoints, compares two tighter velocity sigmas plus a
  higher forward-weight candidate, and rejects aggregate-only successes by
  requiring every seed to pass the stochastic locomotion gate. The completed
  seeds 21–23 run selected sigma 0.10 but promoted nothing.
- (2026-07-26) Completed the exploration-noise and staged low-velocity plan.
  PPO exposes validated initial action variance; four 100k candidates isolated
  variance, entropy, and action range; the winner advanced through a 150k
  continuation and an independent three-seed 300k staged replication. Balance
  generalized, while locomotion remained inconsistent across seeds.
- (2026-07-26) Added the stochastic-balance exploration screen. PPO training
  now accepts a validated `log_std_init`, the GUI exposes it, and walker
  reports distinguish a one-seed balance gate from the final multi-seed
  locomotion promotion gate.
- (2026-07-26) Implemented and screened the balance-first walker iteration.
  Rendering found and the environment closed a raised-terrain torso-contact
  loophole. Walker telemetry now decomposes rewards and logs terminal behavior;
  curricula support conjunctive min/max metric gates and live action scaling.
  The explicit 28 kg preset plus three study ablations completed a one-seed
  100k screen and learned stable deterministic standing, but no candidate met
  the short-run locomotion threshold.
- (2026-07-26) Completed the focused walker promotion study: nine 300k-step
  trainings plus 20-episode deterministic/stochastic diagnostics. The
  equalized 24×128 rollout schedule removed the prior comparison confound, but
  all behavioral pass rates were zero. `curriculum_flat` ranked first and no
  preset was promoted.
- (2026-07-26) Implemented the post-promotion learning plan. Walker quality
  candidates now share a 24×128 PPO rollout schedule and must clear explicit
  episode-length, displacement, fall-rate, and launch-height gates. Arena
  studies now compare feasible combat with and without distance-progress
  shaping, report per-variant timeouts, and gate promotion on
  hits/damage/timeouts. Dense damage reward annealing waits for a configurable
  number of non-timeout outcomes; the GUI and shipped self-play template expose
  the gate.
- (2026-07-25) Completed promotion-scale learning validation: 15 walker runs
  at 300k steps and 18 arena runs at 30k steps, all with three seeds. Rendered
  walker rollouts confirmed partial stepping without launch/slide exploits but
  persistent falls. Arena probes showed that policies never closed to attack
  range and that baseline linear-falloff combat cannot produce a food-free KO
  within the energy budget, establishing the next mechanics/shaping work.
- (2026-07-25) The complete `src/` tree passes
  `mypy src --ignore-missing-imports`. Gymnasium, PettingZoo, SB3 vector-env,
  heterogeneous result-dict, callback, CLI, and config-loading boundaries now
  carry explicit types or runtime narrowing guards.
- (2026-07-25) End-to-end GUI validation now rejects whitespace-only
  experiment names before saving or launching, and a failed template save
  prevents the training request instead of being ignored.
- (2026-07-25) Runtime smoke validation now covers parallel GUI frame capture:
  headless workers advertise SB3-compatible render metadata while only worker
  0 renders. Evaluation checkpoint loading also honors `training.device`,
  avoiding accidental GPU selection for CPU-configured MLP policies, and
  configs without an optional `evaluation` section use the five-episode
  default instead of failing.
- (2026-07-19) Added resumable Priority 3 learning-quality studies. Walker
  studies compare reward and terrain curricula across seeds using zero-action,
  deterministic/stochastic, transfer, launch-height, and push-recovery
  diagnostics. Arena studies tournament resource variants before optional
  contested-food and body-collision candidates, with native-regime episode
  metrics kept distinct from common-arena Elo. Algorithm studies compare PPO,
  SAC, and TD3 with both equal-step and equal-wall-clock budgets. Readiness
  gates prevent short smoke results from changing shipped defaults.
- Deterministic/reproducible training, strict resume provenance, pinned direct
  dependencies, CPU-first MLP presets, and resumable sweeps/benchmarks.
- Durable SQLite run registry, GUI tuning queue, best-checkpoint evaluation,
  registry-backed run analysis, replay jobs, and league Elo jobs.
- (2026-07-19) Added registry operations tooling and broader run comparisons:
  the `registry` CLI inspects status/table counts, exports every SQLite table,
  and filter-prunes runs, analysis jobs, and missing artifact index rows with
  dry-run and unfiltered-prune safeguards. The GUI Analysis tab now charts
  persisted metric history and filters runs by text, experiment, status,
  algorithm, and environment.
- (2026-07-19) Registry-persisted analysis job lifecycles with GUI startup
  recovery:
  orphaned running jobs are marked interrupted, and recent jobs (with results)
  survive restarts and render in the Analysis tab.
- (2026-07-19) Completed arena wizard schema coverage: the `resources` group
  (all eight `ResourceRules` fields) plus `sim.collision_radius` and
  `sim.speed_size_exponent` are now authorable from the New Experiment wizard,
  so a resource-tuned arena no longer requires hand-edited YAML.
- Walker v2 observations, terrain curricula, reward rebalance, SAC/TD3
  baselines, and CPU throughput presets.
- N-agent arena evaluation/tournament/replay support, tournament-Elo morphology
  scoring, collision/resource mechanics, and warning-free shared-policy vector
  metadata handling.
- (2026-07-20) GUI frame capture renders environment 0 only (`env_method`,
  not the tiling `VecEnv.render()`) with a wall-clock capture throttle, and
  only worker 0 retains `render_mode: rgb_array`; other rollout workers and
  the best-model evaluation env stay headless. Headless rollout workers
  advertise uniform render metadata to satisfy SB3's vector-env contract
  without enabling their underlying renderers. GUI
  wizard schema gained `self_play`/`reward_annealing`/`curriculum` groups and
  arena `sensing_radius`/`attack_falloff` fields (resource settings remain
  open — see Priority 1), and `create_config` rejects an empty/whitespace
  `experiment_name`. `RewardAnnealingCallback` now pushes its scale once per
  rollout instead of every vector step. `run_multi_seed` defaults to
  sequential execution, with a warning, when a seed's own training already
  parallelizes via `num_envs > 1`. `WalkerBulletEnv` warns on unknown
  `reward`/`termination` keys, matching the arena's existing `battle_rules`
  warning.
- (2026-07-17) Retired mirror-return morphology scoring: `run_morphology_search`
  now defaults to `tournament_elo` and rejects an explicit
  `morphology_search.scoring: mean_return` with a clear error, since morphology
  search only targets the zero-sum `organism_arena_parallel` env, where a
  shared policy's mean return sums to ~0 by construction and ranked trials on
  noise rather than skill.

## Validation Standard

- Bug fixes: focused regression tests plus the narrowest relevant smoke test.
- Environment or observation changes: API tests and a short train/eval/replay
  smoke.
- Pipeline changes: full tests, Ruff, repository policy, and a CLI smoke.
- GUI workflow changes: API tests, frontend syntax check, and browser checks
  at desktop and mobile widths.
