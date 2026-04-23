# Open Items / TODO Plan

## Near-term (small/medium)
- Add strict resume compatibility checks for model + `vecnormalize.pkl` provenance.
- Standardize all model path handling through one helper module.
- Persist run manifest (`git SHA`, resolved config, dependency snapshot, runtime device info).
- Add JSON output mode for CLI commands for automation-safe parsing.
- Add CSV schema stability checks/versioning for metrics files.

## Larger changes (separate session)
- Rework experiment storage layout to avoid name mutation in multi-seed and sweep orchestration.
- Replace file-based GUI tuning/status IPC with atomic event stream (SSE/WebSocket or durable queue).
- Introduce end-to-end reproducibility mode (deterministic settings + enforcement + metadata).
- Add CI pipeline for lint/test/type checks and lockfile validation.
- Remove tracked generated artifacts (`__pycache__`, `.egg-info`) from git history and enforce clean repository hygiene.

## Known limitations currently retained
- Single active GUI run policy.
- Floating dependency ranges in `pyproject.toml` (not lockfile-backed yet).
- No first-class run registry for comparing experiments by immutable metadata.
