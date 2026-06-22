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
- Introduce end-to-end reproducibility mode (deterministic settings + enforcement + metadata).
- ~~Add CI pipeline for lint/test/type checks and lockfile validation.~~ **Done** — `.github/workflows/ci.yml` runs pytest+coverage, ruff, advisory mypy, security audit, and `check_repo_policy.py` (lockfile completeness). Checkout/setup-python use Node 24-compatible action majors.
- ~~Remove tracked generated artifacts (`__pycache__`, `.egg-info`) and enforce clean repository hygiene.~~ **Done** — `.egg-info` untracked; `check_repo_policy.py` now fails on any tracked `__pycache__`/`.pyc`/`.egg-info`/`.venv` unconditionally (was gated behind `STRICT_REPO_CLEAN`).

## Known limitations currently retained
- Single active GUI run policy.
- Floating dependency ranges in `pyproject.toml` (not lockfile-backed yet).
- No first-class run registry for comparing experiments by immutable metadata.
