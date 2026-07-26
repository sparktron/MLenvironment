# Development Roadmap

Last updated: 2026-07-26

This is the active roadmap. Historical review findings have been folded into
the completed summary below so completed work is not presented as pending.

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
- Run the new focused walker matrix at a short one-seed budget, then at three
  seeds × 300k only if episode length and displacement improve. The code now
  equalizes PPO rollout/update settings and enforces the locomotion gate. Keep
  PPO/SAC/TD3 comparisons deferred until this clears.
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
