from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentPaths:
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    videos_dir: Path


_UNSAFE_RUN_ID = re.compile(r"[^A-Za-z0-9._=+-]")


def sanitize_run_id(run_id: str) -> str:
    """Make an orchestration variant id safe to use as a single path segment.

    Sweep combos and morphology trial ids are generated from config values
    (which can contain ``/`` or other separators), so they are not trusted to
    be valid filenames. Reject empty/dotty ids and replace anything outside a
    conservative allowlist with ``_``.
    """
    cleaned = _UNSAFE_RUN_ID.sub("_", str(run_id))
    if not cleaned or set(cleaned) <= {".", "_"}:
        raise ValueError(f"run_id reduces to an unsafe path segment: {run_id!r}")
    return cleaned


def create_experiment_paths(
    base_dir: str, experiment_name: str, seed: int, run_id: str | None = None
) -> ExperimentPaths:
    """Build the on-disk layout for one run.

    The base layout is ``<base_dir>/<experiment_name>/seed_<seed>/``. When
    ``run_id`` is given (sweep combo or morphology trial), the run is nested
    one level deeper under a ``runs/`` group so orchestration variants stay
    distinct from plain single/multi-seed runs and from summary directories:
    ``<base_dir>/<experiment_name>/runs/<run_id>/seed_<seed>/``.
    """
    run_root = Path(base_dir) / experiment_name
    if run_id is not None:
        run_root = run_root / "runs" / sanitize_run_id(run_id)
    run_dir = run_root / f"seed_{seed}"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    videos_dir = run_dir / "videos"
    for p in [run_dir, checkpoints_dir, logs_dir, videos_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        videos_dir=videos_dir,
    )


def append_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    # Determine whether we need to write a header row. Check actual file
    # content rather than just file existence, so a pre-created empty file
    # still gets a header on first write.
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(metrics.keys()),
            extrasaction="ignore",  # silently drop keys not in fieldnames
            restval="",  # fill missing keys with empty string
        )
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)
