from __future__ import annotations

import csv
import json
import logging
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

from rl_framework.training.sb3_runner import train
from rl_framework.utils.config_merge import set_nested

logger = logging.getLogger(__name__)


def run_sweep(cfg: dict[str, Any], dry_run: bool = False) -> list[dict[str, Any]]:
    sweep_cfg = cfg.get("sweep", {})
    params = sweep_cfg.get("parameters", {})
    keys = list(params.keys())
    values = [params[k] for k in keys]

    # FIX #33: Validate all parameter keys against the base config before
    # any runs start, so a misspelled key fails fast with a clear message.
    probe = deepcopy(cfg)
    first_values = [v[0] for v in values]
    for k, v in zip(keys, first_values):
        try:
            set_nested(probe, k, v, strict=True)
        except KeyError as exc:
            raise KeyError(
                f"Sweep parameter key {k!r} not found in base config. "
                "Check for typos in the sweep.parameters keys."
            ) from exc

    planned_runs: list[dict[str, Any]] = []

    try:
        for combo in product(*values):
            run_cfg = deepcopy(cfg)
            name_suffix = []
            overrides: dict[str, Any] = {}
            for k, v in zip(keys, combo):
                set_nested(run_cfg, k, v)
                name_suffix.append(f"{k.split('.')[-1]}_{v}")
                overrides[k] = v
            # Each combo is a distinct run under the same experiment. Route it
            # through output.run_id rather than mutating experiment_name, so it
            # lands at outputs/<experiment_name>/runs/<run_id>/seed_<seed>/ and
            # stays co-located with the sweep_summary/ written below.
            run_id = "__".join(name_suffix)
            run_cfg.setdefault("output", {})["run_id"] = run_id
            planned_runs.append(
                {
                    "experiment_name": cfg["experiment_name"],
                    "run_id": run_id,
                    "overrides": overrides,
                    "status": "pending",
                }
            )
            if not dry_run:
                # FIX #31: Catch per-combo failures so the sweep continues and
                # the manifest records which combos succeeded/failed.
                try:
                    train(run_cfg)
                    planned_runs[-1]["status"] = "success"
                except Exception as exc:
                    logger.error(
                        "Sweep combo %s failed: %s",
                        run_cfg["output"]["run_id"],
                        exc,
                        exc_info=True,
                    )
                    planned_runs[-1]["status"] = f"failed: {exc}"
    finally:
        # FIX #31: Always write the manifest, even if an unhandled exception
        # aborts the loop (e.g. KeyboardInterrupt or out-of-memory).
        if planned_runs:
            _write_sweep_manifest(cfg, planned_runs)

    return planned_runs


def _write_sweep_manifest(
    cfg: dict[str, Any], planned_runs: list[dict[str, Any]]
) -> Path:
    base_dir = cfg.get("output", {}).get("base_dir", "outputs")
    out_dir = Path(base_dir) / cfg["experiment_name"] / "sweep_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "planned_runs.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "experiment_name", "run_id", "overrides", "status"])
        for i, run in enumerate(planned_runs):
            writer.writerow(
                [
                    i,
                    run["experiment_name"],
                    run.get("run_id", ""),
                    json.dumps(run["overrides"]),
                    run.get("status", ""),
                ]
            )
    return path
