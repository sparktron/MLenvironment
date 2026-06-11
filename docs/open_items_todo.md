# Open Items / TODO Plan

## Near-term (small/medium)
- ~~Add strict resume compatibility checks for model + `vecnormalize.pkl` provenance.~~ **Done** — `reproducibility.check_resume_provenance` compares a resuming run against the source checkpoint's `run_metadata.json` and flags silent drift SB3's space check misses (changed `environment` block, flipped `normalize_observations`, different policy/gamma). Strict mode hard-fails on drift or an unverifiable manifest; non-strict warns. Wired into `train()`.
- Standardize all model path handling through one helper module.
- ~~Persist run manifest (`git SHA`, resolved config, dependency snapshot, runtime device info).~~ **Done** — `reproducibility.write_run_metadata` writes `run_metadata.json` per run (git commit/branch/full worktree dirty state including staged and untracked files, full config + sha256, lockfile sha256, python/platform/host; device via persisted config). Wired into `train()`, with a strict mode that hard-fails on missing provenance.
- ~~Audit June arena evaluation timeout accounting.~~ **Done** — `eval_runner.evaluate()` now recognizes arena `episode_outcome: timeout` as a truncation and reuses the arena VecEnv adapter so SuperSuit's `uint8` done arrays cannot skew VecNormalize masking.
- ~~Track repository-specific agent workflow guidance.~~ **Done** — `AGENTS.md` is now part of the repo and documents local setup, CLI conventions, environment quirks, GUI behavior, and commit/push workflow for agent-assisted maintenance.
- ~~Add pytest coverage tooling to dev dependencies.~~ **Done** — `pytest-cov` is declared in `pyproject.toml`, pinned in `requirements-lock.txt`, and supports the documented `--cov=src/rl_framework --cov-fail-under=60` check.
- Add JSON output mode for CLI commands for automation-safe parsing.
- Add CSV schema stability checks/versioning for metrics files.

## Larger changes (separate session)
- ~~Rework experiment storage layout to avoid name mutation in multi-seed and sweep orchestration.~~ **Done** — variants now route through `output.run_id` (`<experiment>/runs/<run_id>/seed_<seed>/`); multi-seed no longer mutates the name. See `create_experiment_paths`.
- Replace file-based GUI tuning/status IPC with atomic event stream (SSE/WebSocket or durable queue).
- Introduce end-to-end reproducibility mode (deterministic settings + enforcement + metadata).
- ~~Add CI pipeline for lint/test/type checks and lockfile validation.~~ **Done** — `.github/workflows/ci.yml` runs pytest+coverage, ruff, mypy (non-blocking), and `check_repo_policy.py` (lockfile completeness).
- ~~Remove tracked generated artifacts (`__pycache__`, `.egg-info`) and enforce clean repository hygiene.~~ **Done** — `.egg-info` untracked; `check_repo_policy.py` now fails on any tracked `__pycache__`/`.pyc`/`.egg-info`/`.venv` unconditionally (was gated behind `STRICT_REPO_CLEAN`).

## Known limitations currently retained
- Single active GUI run policy.
- Floating dependency ranges in `pyproject.toml` (not lockfile-backed yet).
- No first-class run registry for comparing experiments by immutable metadata.
