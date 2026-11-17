#!/usr/bin/env python3
"""Thin entry point; the implementation lives in ``rl_framework.benchmark_device_matrix``.

Kept so ``python scripts/benchmark_device_matrix.py`` still works from a checkout
without an editable install (the package depends on nothing in ``scripts/``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl_framework.benchmark_device_matrix import main  # noqa: E402

if __name__ == "__main__":
    main()
