"""Compatibility package for running src-layout modules without installation.

This package maps `rl_framework.*` imports to `src/rl_framework/*` so commands
like `python -m rl_framework.cli.main` work directly from the repository root.
"""

from __future__ import annotations

from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_src_pkg = _repo_root / "src" / "rl_framework"

# Explicit package search path for submodules (e.g., rl_framework.cli.main).
__path__ = [str(_src_pkg)]
