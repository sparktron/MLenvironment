from __future__ import annotations

import csv
import json
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

from rl_framework.training.sb3_runner import train
from rl_framework.utils.config_merge import set_nested


def run_sweep(cfg: dict[str, Any], dry_run: bool = False) -> list[dict[str, Any]]:
    sweep_cfg = cfg.get("sweep", {})
    params = sweep_cfg.get("parameters", {})
    keys = list(params.keys())
    values = [params[k] for k in keys]
    planned_runs: list[dict[str, Any]] = []

    for combo in product(*values):
        run_cfg = deepcopy(cfg)
        name_suffix = []
        overrides: dict[str, Any] = {}
        for k, v in zip(keys, combo):
            set_nested(run_cfg, k, v)
            name_suffix.append(f"{k.split('.')[-1]}_{v}")
            overrides[k] = v
        run_cfg["experiment_name"] = f"{cfg['experiment_name']}__{'__'.join(name_suffix)}"
        planned_runs.append({"experiment_name": run_cfg["experiment_name"], "overrides": overrides})
        if not dry_run:
            train(run_cfg)
    _write_sweep_manifest(cfg, planned_runs)
    return planned_runs


def _write_sweep_manifest(cfg: dict[str, Any], planned_runs: list[dict[str, Any]]) -> Path:
    base_dir = cfg.get("output", {}).get("base_dir", "outputs")
    out_dir = Path(base_dir) / cfg["experiment_name"] / "sweep_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "planned_runs.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "experiment_name", "overrides"])
        for i, run in enumerate(planned_runs):
            writer.writerow([i, run["experiment_name"], json.dumps(run["overrides"])])
    return path
