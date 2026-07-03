# Open Items / TODO Plan

## Near-term (small/medium)
- ~~Add strict resume compatibility checks for model + `vecnormalize.pkl` provenance.~~ **Done** — `reproducibility.check_resume_provenance` compares a resuming run against the source checkpoint's `run_metadata.json` and flags silent drift SB3's space check misses (changed `environment` block, flipped `normalize_observations`, different policy/gamma). Strict mode hard-fails on drift or an unverifiable manifest; non-strict warns. Wired into `train()`.
- ~~Standardize all model path handling through one helper module.~~ **Done** — `utils/checkpoint.py` now owns `model_zip_path`, `vecnormalize_path_for_model`, `legacy_vecnormalize_path_for_model`, `find_vecnormalize_path_for_model`, and `validate_resume_path`. `sb3_runner` (private re-export aliases), `eval_runner`, `reproducibility` (`_as_zip_path`), and `morphology_search` (`_as_model_zip_path`) all delegate to it instead of re-implementing the `.zip`/sidecar resolution chain.
- ~~Persist run manifest (`git SHA`, resolved config, dependency snapshot, runtime device info).~~ **Done** — `reproducibility.write_run_metadata` writes `run_metadata.json` per run (git commit/branch/full worktree dirty state including staged and untracked files, full config + sha256, lockfile sha256, python/platform/host; device via persisted config). Wired into `train()`, with a strict mode that hard-fails on missing provenance.
- ~~Audit June arena evaluation timeout accounting.~~ **Done** — `eval_runner.evaluate()` now recognizes arena `episode_outcome: timeout` as a truncation and reuses the arena VecEnv adapter so SuperSuit's `uint8` done arrays cannot skew VecNormalize masking.
- ~~Track repository-specific agent workflow guidance.~~ **Done** — `AGENTS.md` is now part of the repo and documents local setup, CLI conventions, environment quirks, GUI behavior, and commit/push workflow for agent-assisted maintenance.
- ~~Add pytest coverage tooling to dev dependencies.~~ **Done** — `pytest-cov` is declared in `pyproject.toml`, pinned in `requirements-lock.txt`, and supports the documented `--cov=src/rl_framework --cov-fail-under=60` check. The dev pytest constraint excludes the vulnerable 9.0.2 release.
- ~~Add JSON output mode for CLI commands for automation-safe parsing.~~ **Done** — every non-GUI subcommand returns a result dict; `--json` emits it as a single JSON line to stdout (suppressing human output) and `--json-out <path>` writes the payload to a file. CLI-level coverage in `tests/test_cli_json_output.py`.
- ~~Add CSV schema stability checks/versioning for metrics files.~~ **Done** — `append_metrics_csv` now reads the existing header and raises `CsvSchemaError` on any column add/remove/reorder instead of silently dropping new keys (`extrasaction="ignore"`) or padding missing ones (`restval=""`). Covered in `tests/test_logging_utils.py`.

## Larger changes (separate session)
- ~~Rework experiment storage layout to avoid name mutation in multi-seed and sweep orchestration.~~ **Done** — variants now route through `output.run_id` (`<experiment>/runs/<run_id>/seed_<seed>/`); multi-seed no longer mutates the name. See `create_experiment_paths`.
- Replace file-based GUI tuning/status IPC with atomic event stream (SSE/WebSocket or durable queue).
- ~~Work through the GUI correctness and layout plan in `docs/ui_roadmap.md`.~~ **Done** — fixed stale walker metadata, template-to-environment state reset, dashboard disabled/empty states, variant run labels, compact mobile wizard progress, and denser desktop dashboard layout. The roadmap file now records implementation status and the browser checklist.
- Introduce end-to-end reproducibility mode (deterministic settings + enforcement + metadata).
- ~~Add CI pipeline for lint/test/type checks and lockfile validation.~~ **Done** — `.github/workflows/ci.yml` runs pytest+coverage, ruff, advisory mypy, security audit, and `check_repo_policy.py` (lockfile completeness). Checkout/setup-python use Node 24-compatible action majors.
- ~~Remove tracked generated artifacts (`__pycache__`, `.egg-info`) and enforce clean repository hygiene.~~ **Done** — `.egg-info` untracked; `check_repo_policy.py` now fails on any tracked `__pycache__`/`.pyc`/`.egg-info`/`.venv` unconditionally (was gated behind `STRICT_REPO_CLEAN`).

## Known limitations currently retained
- Single active GUI run policy.
- Floating dependency ranges in `pyproject.toml` (not lockfile-backed yet).
- No first-class run registry for comparing experiments by immutable metadata.

## Next development roadmap (2026-06-30)

This section is the current execution roadmap. The older review sections below
remain as evidence and background, but new work should flow through these
priority groups unless a fresh bug review changes the order.

### Priority 0: unblock known correctness bugs
- ~~Fix `validate_experiment_config()` for arena self-play parallelism.~~
  **Done (2026-06-30)** — validation now allows `training.num_envs > 1` only
  when `self_play.enabled: true`, while preserving the shared-policy SuperSuit
  guard. Regression tests cover the rejected shared-policy case, the accepted
  self-play case, and the bundled `organisms_fight_arena.yaml` config.
- ~~Align GUI arena defaults with the same rule.~~ **Done (2026-06-30)** — the
  wizard still defaults arena `num_envs` to 1 for shared-policy safety, but no
  longer caps the field at 1 so loaded self-play templates can keep their
  parallel worker count.
- ~~Refresh stale README schema facts while touching docs.~~ **Done
  (2026-06-30)** — README now points at the current walker dimensions, current
  arena replay/support limitations, and the self-play parallel arena path.

### Priority 1: walker training stability and learning quality
- ~~Fix the NaN PPO action mean at high `num_envs`.~~ **Done (2026-07-02)** —
  root cause was `WalkerBulletEnv.step()` feeding raw, un-sanitized
  `pos`/`quat`/`lin_vel` from PyBullet into reward and termination (only the
  observation copy was `nan_to_num`-guarded). A rare solver-divergence frame
  (more likely to be hit per wall-clock second at higher parallelism) could
  emit a `NaN` reward that survived uncaught until `max_steps` truncation,
  poisoning GAE for that episode and NaN-ing the next PPO gradient update.
  `step()` now detects non-finite physics reads, sanitizes them, and treats
  divergence as an immediate fall (torso-contact-equivalent) so it can never
  linger. Verified with a smoke run reproducing the original report exactly
  (`num_envs: 24`, one 49,152-step rollout, `training.check_nans: true`) —
  first PPO update now completes cleanly.
- ~~Establish a stable high-throughput walker preset before changing
  defaults.~~ **Done (2026-07-03)** — with the NaN-source fixed, benchmarked 5
  presets at `num_envs: 24` (this machine's CPU-saturating value) for 4 PPO
  rollouts each, `training.check_nans: true`, verifying the saved policy's
  action-distribution mean stayed finite after every update (not just
  `explained_variance`/loss sanity):

  | preset | n_steps | batch_size | lr | ent_coef | clip_range | FPS | stable |
  |---|---|---|---|---|---|---|---|
  | current_default | 2048 | 512 | 3e-4 | 0.005 | 0.2 | **5605** | yes |
  | lower_lr_tighter_clip | 2048 | 512 | 1e-4 | 0.005 | 0.2 | 4948 | yes |
  | hybrid_1024_256 | 1024 | 256 | 3e-4 | 0.005 | 0.2 | 4592 | yes |
  | pybullet_locomotion | 512 | 128 | 3e-4 | 0.0 | 0.2 | 3728 | yes |
  | rlzoo_bipedal | 2048 | 64 | 3e-4 | 0.0 | 0.18 | 3159 | yes |

  All five were numerically stable (the NaN fix holds across hyperparameter
  choices, not just the one config originally reported). `batch_size` is the
  dominant throughput lever on this CPU-only setup: SB3's reported FPS
  includes the PPO update, and small batches (64, 128) mean far more
  minibatch gradient steps per rollout (RL-Zoo's `batch_size: 64` runs ~7,680
  backprop steps per rollout here vs. `current_default`'s 960), so they lose
  on wall-clock throughput despite identical env-step cost. `current_default`
  (already the shipped hyperparameters, just previously capped at
  `num_envs: 8`) was both the fastest and stable, so `robot_walk_basic.yaml`
  and `my_walker.yaml` now ship with `num_envs: 24` instead of `8` — no other
  hyperparameters changed. RL-Zoo/PyBullet-style small-batch presets remain
  documented above as options, not defaults, given the throughput cost on
  this hardware.
- Add a best-checkpoint evaluation path. Long walker runs should save the best
  policy according to a separately normalized eval env, not only periodic and
  final checkpoints. Include resume/eval sidecar behavior in the regression
  tests.
- Rebalance the default walker reward so standing still is not overpaid. Test
  lower `alive_bonus`, stronger forward-progress incentives, energy/torque
  costs, and curriculum schedules that ramp target velocity and perturbations
  after balance is learned.
- Add an observation-version plan before changing policy inputs. Candidate v2
  signals are foot-contact indicators and a coordinate-free mode that removes
  absolute x/y while retaining height and velocities. Treat this as checkpoint
  incompatible and document migration expectations.

### Priority 2: throughput and experiment operations
- Produce documented local training presets for this 24-core machine: quick
  smoke, reliable overnight walker, high-throughput walker, arena self-play, and
  multi-seed evaluation. Record expected FPS, memory footprint, and failure
  modes next to the config recommendations.
- Add optional `training.torch_num_threads` and `training.worker_start_method`
  controls if benchmarks show they improve stability or CPU utilization. The
  runner already caps BLAS thread env vars for subprocesses; PyTorch update
  threading is still implicit.
- Replace file-based GUI tuning/status IPC with an atomic event stream
  (SSE/WebSocket) or a durable queue. Preserve the current simple polling API as
  a fallback until browser and API tests cover no-run, running, stopped, and
  completed states.
- Add a first-class run registry that records immutable run identity, config
  hash, seed, algorithm, checkpoint paths, VecNormalize sidecars, metrics CSVs,
  and parent/resume relationships. Use it to power comparisons, cleanup, and GUI
  output discovery instead of directory-name inference.
- Make sweeps and benchmark matrices resumable. Each variant should have a
  machine-readable manifest, completion marker, and partial-result summary so a
  failed late regime does not require recomputing successful regimes.

### Priority 3: feature additions
- Walker curricula: add terrain presets (`flat`, `uneven`, `obstacle/stump`,
  `push_recovery`) and example configs that progress from balance to locomotion
  to perturbation recovery.
- Algorithm baselines: add optional SAC and TD3 experiment configs for the
  continuous-control walker. Keep PPO as the default runner until the abstraction
  cost is justified by working configs and tests.
- Arena richness: add body collision, energy/food mechanics, and speed/size
  tradeoffs so organism morphology has strategic pressure beyond damage and
  health scaling.
- Arena tooling: extend tournament/eval/replay support beyond head-to-head
  where it makes sense for N-agent arenas, and let `morph-search` score
  candidates by tournament Elo rather than a single opponent result.
- GUI analysis views: add run comparison, best-checkpoint surfacing, league
  snapshot ratings, and replay launch links once the run registry exists.

### Validation expectations for roadmap work
- Bug fixes get focused regression tests plus the narrowest relevant CLI smoke
  check.
- Walker environment, reward, termination, or observation changes get
  `pytest`, a reset/step smoke test, and a short training or replay smoke.
- Training-pipeline or benchmark changes get `pytest`, `ruff`, repo policy, and
  one short command-line run that exercises the changed path.
- GUI workflow changes get API tests plus browser verification for the affected
  desktop and mobile flows.

## Bipedal walker training review plan (2026-06-29)

### Confirmed bugs / correctness gaps
- ~~Scale checkpoint cadence by `training.num_envs`.~~ **Done** — SB3 callback calls advance by vector-env step, so `training.checkpoint_every` must be converted from environment timesteps to callback calls. Without this, `robot_walk_basic` with `num_envs: 8` saved every 400k env steps instead of every 50k.
- ~~Load `VecNormalize` statistics in `render-replay` for `walker_bullet`, matching `eval_runner.evaluate()`.~~ **Done** — Gymnasium replay now runs through `DummyVecEnv` and loads the model-specific/legacy VecNormalize sidecar when present before `model.predict()`.
- ~~Penalize all terminal fall modes consistently.~~ **Done** — low-height, torso-contact, and max-height terminal paths now pass `fell=True` to `WalkerReward.compute()`; pure truncation remains unpenalized.
- ~~Make action clipping explicit in `WalkerBulletEnv.step()` before both control and reward accounting.~~ **Done** — env step clips to `[-1, 1]` before action-latency buffering, dynamics, and reward calculation.
- ~~Either wire or remove `environment.sim.body_half_extents`.~~ **Done** — removed the ignored knob from GUI schema, bundled walker YAMLs, README, and tests; old configs that contain it still load because unknown `sim` keys are tolerated.
- ~~Add NaN diagnostics for walker training.~~ **Done** — optional `training.check_nans: true` wraps the vector env in SB3 `VecCheckNan` and fails fast on NaN/Inf values without changing defaults.
- ~~Add regression tests for reward/termination edge cases, clipped action penalty, and normalized replay.~~ **Done** — targeted tests cover low-height/max-height terminal penalties, reward-side clipped actions, normalized walker replay, hidden geometry schema, and `training.check_nans` validation.

### Training efficiency / operator defaults
- Benchmark before raising `robot_walk_basic.training.num_envs` toward the local CPU-saturating value. A smoke run with `num_envs: 24` collected one 49,152-step rollout at ~3,100 FPS but produced NaN PPO action means on the first update, so higher parallelism needs optimizer/reward-scale validation before becoming the default.
- Benchmark `num_envs` x `n_steps` x `batch_size` on this machine with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, then record recommended presets. Compare current `24 x 2048 x 512` against RL Zoo-style BipedalWalker PPO settings (`n_envs: 32`, `n_steps: 2048`, `batch_size: 64`, `gamma: 0.999`, `ent_coef: 0.0`, `clip_range: 0.18`) and PyBullet locomotion defaults (`n_envs: 16`, `n_steps: 512`, `batch_size: 128`, ReLU 256x256 policy, `use_sde: true`).
- Add optional `training.torch_num_threads` / `training.worker_start_method` controls after benchmarking. The runner sets BLAS env vars for subprocesses, but PyTorch CPU update threading is still implicit.
- Add an `EvalCallback`/best-checkpoint path with a separately normalized eval env so long walker runs keep the best policy, not only periodic and final checkpoints.

### Walker environment and learning features
- Add foot contact indicators to the observation. Gymnasium BipedalWalker exposes leg ground contact, and contact phase is useful for gait learning; this will require an observation shape/version change and compatibility notes for old checkpoints.
- Add a configurable observation mode that can remove absolute x/y position from policy input while retaining velocity and height. Gymnasium BipedalWalker deliberately omits coordinates; keeping unbounded position in a flat-plane task may encourage time/progress overfitting and makes normalization drift with long episodes.
- Add terrain variation presets: flat, uneven, obstacle/stump, and push-recovery perturbations. This aligns the custom PyBullet walker more closely with normal/hardcore BipedalWalker training curricula.
- Rebalance the default reward so survival does not dominate locomotion. Current `alive_bonus: 5.0` over 800 steps can make standing still more attractive than learning gait; compare against forward-progress-plus-energy-cost reward structures and use curriculum to ramp target speed and posture penalties.
- Add curriculum examples for walker: start with balance/short horizon/low target velocity, then increase `target_velocity`, horizon, terrain difficulty, perturbations, and domain randomization.
- Add optional SAC/TD3 experiment configs for the continuous-control walker baseline. PPO remains the default, but off-policy algorithms are useful comparison points for sample efficiency.
