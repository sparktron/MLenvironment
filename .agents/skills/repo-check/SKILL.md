---
name: repo-check
description: Run the full pre-push sanity check on this repo — pytest with coverage gate, ruff lint, and scripts/check_repo_policy.py. Use before pushing to master, or whenever the user asks "is this ready to push" or "run the full check".
---

# repo-check

Single command that runs everything CI runs (minus the non-blocking mypy and pip-audit).

## Procedure

Run these three commands in sequence. Stop at the first failure and report it; otherwise report all green.

```bash
source .venv/bin/activate

# 1. Test suite with coverage gate (CI fails if coverage < 60%)
pytest -q --cov=src/rl_framework --cov-fail-under=60 --ignore=tests/test_training_manager_streaming.py

# 2. Lint
ruff check src tests scripts

# 3. Repo policy (no tracked __pycache__/.venv/.egg-info, lockfile completeness)
python scripts/check_repo_policy.py
```

Output a compact pass/fail summary per step, e.g.:

```
✓ pytest: 96 passed, coverage 72%
✓ ruff: clean
✓ repo-policy: clean
→ ready to push
```

## Gotchas

- The streaming test (`test_training_manager_streaming.py`) is slow / order-dependent; exclude it for sanity checks (CI runs it separately).
- `check_repo_policy.py` enforces that every direct dep in `pyproject.toml` appears in `requirements-lock.txt`. If it fails on lockfile, the fix is usually `pip freeze > requirements-lock.txt` from a clean venv (consult with user before doing this — they may want a specific pin strategy).
- If pytest fails, do **not** push. Report the failure and ask the user how to proceed.
