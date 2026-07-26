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
