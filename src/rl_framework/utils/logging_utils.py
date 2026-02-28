from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentPaths:
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    videos_dir: Path


def create_experiment_paths(base_dir: str, experiment_name: str, seed: int) -> ExperimentPaths:
    run_dir = Path(base_dir) / experiment_name / f"seed_{seed}"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    videos_dir = run_dir / "videos"
    for p in [run_dir, checkpoints_dir, logs_dir, videos_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(run_dir=run_dir, checkpoints_dir=checkpoints_dir, logs_dir=logs_dir, videos_dir=videos_dir)


def append_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)
