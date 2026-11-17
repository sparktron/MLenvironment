# ADR-004: Install the src layout editable; drop the root `rl_framework/` shim

**Status:** Accepted · **Date:** 2026-11-17

## Context

The package lives at `src/rl_framework/`, but a second `rl_framework/` folder at
the repo root held only an `__init__.py` that pointed `rl_framework.*` at the
src tree. That let `python -m rl_framework.cli.main` run from the checkout
without an install, and it hid packaging bugs: `configs/experiments/*.yaml` and
the GUI static/template files were never declared as package data, so an
installed wheel would have shipped without them.

ADR-003 recorded the shim as unchanged because removing it required an
editable install, which conflicted with the install policy then in force:
"install from `requirements-lock.txt`, not `pip install -e .`". That policy's
purpose was pinned dependencies, not avoiding the editable install itself.

## Decision

1. Install in two steps, in this order, in CI and in every doc:

   ```
   pip install -r requirements-lock.txt
   pip install --no-deps -e .
   ```

   The lockfile stays the only source of dependency versions; `--no-deps` stops
   the editable install from re-resolving them.
2. Delete the root `rl_framework/` shim.
3. Declare the runtime assets as `[tool.setuptools.package-data]`
   (`configs/experiments/*.yaml`, `gui/static/*`, `gui/templates/*`).
4. Remove the `sys.path` inserts in `scripts/` and the `pythonpath = ["."]`
   pytest setting; they only worked around the missing install.
5. The default `--config-dir` is now package-relative (`DEFAULT_CONFIG_DIR` in
   `core/config/config.py`). It was the string
   `src/rl_framework/configs/experiments`, which only resolved from the repo root.
6. `scripts/check_repo_policy.py` fails if a root `rl_framework/` folder
   reappears.

## Consequences

- `import rl_framework` works from any directory after the install, and resolves
  into `src/`.
- Running a script from a bare checkout without installing now fails with
  `ModuleNotFoundError` by design.
- Checkpoints, observation shapes and env behaviour are unaffected: this is
  packaging only.
