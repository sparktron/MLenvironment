from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CsvSchemaError(Exception):
    """Raised when appending to a CSV whose header doesn't match the new data.

    This catches silent data corruption from unintentional schema drift: new
    columns would be silently dropped and missing columns would be filled with
    empty strings if we didn't check first.
    """


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
    """Append one row of *metrics* to the CSV at *path*.

    On the first write (empty or absent file) the header is derived from
    ``metrics.keys()``. On subsequent writes the existing header is read and
    validated: if the keys differ a :class:`CsvSchemaError` is raised rather
    than silently dropping new columns or padding missing ones with empty
    strings.
    """
    is_new = not path.exists() or path.stat().st_size == 0
    if not is_new:
        with path.open(newline="", encoding="utf-8") as fh:
            existing_fields = next(csv.reader(fh), None)
        if existing_fields is not None:
            new_keys = list(metrics.keys())
            if existing_fields != new_keys:
                added = [k for k in new_keys if k not in existing_fields]
                removed = [k for k in existing_fields if k not in new_keys]
                parts = []
                if added:
                    parts.append(f"new keys not in header: {added}")
                if removed:
                    parts.append(f"header keys missing from data: {removed}")
                raise CsvSchemaError(
                    f"CSV schema mismatch in {path}: {'; '.join(parts)}. "
                    "Delete the file to reset, or migrate it to the new schema."
                )
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(metrics.keys()),
            extrasaction="ignore",
            restval="",
        )
        if is_new:
            writer.writeheader()
        writer.writerow(metrics)
