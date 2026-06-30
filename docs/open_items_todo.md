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
