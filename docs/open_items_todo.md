# Development Roadmap

Last updated: 2026-07-12

This is the active roadmap. Historical review findings have been folded into
the completed summary below so completed work is not presented as pending.

## Priority 1: Runtime And GUI Performance

- **Capture GUI frames from one environment on a wall-clock budget.** GUI
  training currently enables RGB rendering for every worker and captures a
  tiled vector render. Render environment 0 only, cap capture frequency at
  roughly one frame per second, and only enable rendering when the dashboard
  frame view is requested.
- **Finish GUI schema coverage.** Expose `self_play`, `reward_annealing`, and
  `curriculum` sections, plus arena `sensing_radius`, `attack_falloff`, and
  resource settings. The backend accepts loaded templates today, but the
  wizard cannot author every supported option directly.
- **Persist analysis jobs.** Replay and league-rating jobs are intentionally
  lightweight/process-local. Store their lifecycle in the run registry if GUI
  restart recovery or multi-process GUI deployment becomes necessary.

## Priority 2: Evaluation And Operations

- **Retire mirror-return morphology scoring.** `morphology_search.scoring:
  tournament_elo` is now the meaningful arena score. Deprecate or reject the
  legacy shared-policy `mean_return` mode for arena searches, because zero-sum
  cancellation makes it uninformative.
- **Add registry maintenance commands.** Provide CLI support to inspect,
  export, and prune registry records/artifacts, including stale analysis jobs.
- **Broaden analysis comparisons.** Add metric history and configurable run
  filters to the GUI comparison view; current cards show latest persisted
  metrics and artifacts only.

## Priority 3: Learning Quality And Features

- **Empirically tune walker reward/curriculum quality.** The rebalance and
  terrain curricula are implemented; compare learned gait, recovery, and
  transfer metrics across seeds before changing the shipped defaults again.
- **Balance arena resources.** Collision, food, energy, and size/speed
  tradeoffs are implemented. Run self-play/tournament studies to tune resource
  costs, respawn cadence, and food density, then promote measured presets.
- **Arena strategic depth.** Consider body collision damage, richer food
  placement, or territory/objective mechanics only after resource balance data
  demonstrates a clear need.
- **Algorithm comparison.** Benchmark PPO, SAC, and TD3 v2 walker baselines
  on equal step and wall-clock budgets, including deterministic evaluation.

## Retained Limitations

- The GUI permits one active in-process training run.
- Shared-policy arena training remains single-process through SuperSuit; use
  self-play for native parallel arena rollouts.
- Observation changes are checkpoint-incompatible by design: walker v1/v2 and
  arena 8D/13D checkpoints must not be mixed.

## Completed Foundations

- Deterministic/reproducible training, strict resume provenance, pinned direct
  dependencies, CPU-first MLP presets, and resumable sweeps/benchmarks.
- Durable SQLite run registry, GUI tuning queue, best-checkpoint evaluation,
  registry-backed run analysis, replay jobs, and league Elo jobs.
- Walker v2 observations, terrain curricula, reward rebalance, SAC/TD3
  baselines, and CPU throughput presets.
- N-agent arena evaluation/tournament/replay support, tournament-Elo morphology
  scoring, collision/resource mechanics, and warning-free shared-policy vector
  metadata handling.

## Validation Standard

- Bug fixes: focused regression tests plus the narrowest relevant smoke test.
- Environment or observation changes: API tests and a short train/eval/replay
  smoke.
- Pipeline changes: full tests, Ruff, repository policy, and a CLI smoke.
- GUI workflow changes: API tests, frontend syntax check, and browser checks
  at desktop and mobile widths.
