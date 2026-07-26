# Development Roadmap

Last updated: 2026-07-26

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
