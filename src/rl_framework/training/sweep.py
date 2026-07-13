from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

from rl_framework.training.sb3_runner import train
from rl_framework.utils.checkpoint import model_zip_path
from rl_framework.utils.config_merge import set_nested

logger = logging.getLogger(__name__)


def run_sweep(
    cfg: dict[str, Any], dry_run: bool = False, resume: bool = False
) -> list[dict[str, Any]]:
    """Run a grid sweep, optionally resuming completed variants from state."""
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

    for combo in product(*values):
        name_suffix = []
        overrides: dict[str, Any] = {}
        for k, v in zip(keys, combo):
            name_suffix.append(f"{k.split('.')[-1]}_{v}")
            overrides[k] = v
        # Each combo is a distinct run under the same experiment. Route it
        # through output.run_id rather than mutating experiment_name, so it
        # lands at outputs/<experiment_name>/runs/<run_id>/seed_<seed>/ and
        # stays co-located with the sweep_summary/ written below.
        run_id = "__".join(name_suffix)
        planned_runs.append(
            {
                "experiment_name": cfg["experiment_name"],
                "run_id": run_id,
                "overrides": overrides,
                "status": "pending",
            }
        )
    manifest_path = _sweep_manifest_path(cfg)
    if resume:
        _apply_resume_state(cfg, planned_runs, manifest_path)

    try:
        for run in planned_runs:
            if dry_run or run["status"] == "completed":
                continue
            run_cfg = deepcopy(cfg)
            for key, value in run["overrides"].items():
                set_nested(run_cfg, key, value)
            run_cfg.setdefault("output", {})["run_id"] = run["run_id"]
            # FIX #31: Catch per-combo failures so the sweep continues and
            # the manifest records which combos succeeded/failed.
            try:
                model_path = train(run_cfg)
                marker = _write_completion_marker(model_path)
                run["completion_marker"] = str(marker)
                run["status"] = "completed"
            except Exception as exc:
                logger.error(
                    "Sweep combo %s failed: %s",
                    run_cfg["output"]["run_id"],
                    exc,
                    exc_info=True,
                )
                run["status"] = f"failed: {exc}"
            finally:
                _write_sweep_manifest(cfg, planned_runs)
    finally:
        # Always write both human-readable and machine-readable state, even if
        # an interruption escapes the per-variant exception handler.
        if planned_runs:
            _write_sweep_manifest(cfg, planned_runs)

    return planned_runs


def _sweep_manifest_path(cfg: dict[str, Any]) -> Path:
    base_dir = cfg.get("output", {}).get("base_dir", "outputs")
    return Path(base_dir) / cfg["experiment_name"] / "sweep_summary" / "state.json"


def _sweep_fingerprint(cfg: dict[str, Any], planned_runs: list[dict[str, Any]]) -> str:
    payload = {
        "experiment_name": cfg["experiment_name"],
        "parameters": cfg.get("sweep", {}).get("parameters", {}),
        "runs": [
            {"run_id": run["run_id"], "overrides": run["overrides"]}
            for run in planned_runs
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _apply_resume_state(
    cfg: dict[str, Any], planned_runs: list[dict[str, Any]], manifest_path: Path
) -> None:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No sweep state found at {manifest_path}. Run the sweep once before resuming."
        )
    state = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = _sweep_fingerprint(cfg, planned_runs)
    if state.get("fingerprint") != fingerprint:
        raise ValueError(
            "Sweep resume state does not match this experiment's parameter grid. "
            "Use a new output directory or start without resume."
        )
    previous = {run["run_id"]: run for run in state.get("runs", [])}
    for run in planned_runs:
        prior = previous.get(run["run_id"])
        marker = (
            Path(prior["completion_marker"])
            if prior and prior.get("completion_marker")
            else None
        )
        if prior and prior.get("status") == "completed" and marker and marker.is_file():
            run["status"] = "completed"
            run["completion_marker"] = str(marker)


def _write_completion_marker(model_path: str | Path) -> Path:
    model = model_zip_path(model_path)
    if not model.exists():
        raise FileNotFoundError(f"Sweep training did not produce a model: {model}")
    marker = model.parent.parent / "completion.json"
    _atomic_json_write(marker, {"model": str(model), "status": "completed"})
    return marker


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def _write_sweep_manifest(
    cfg: dict[str, Any], planned_runs: list[dict[str, Any]]
) -> Path:
    manifest_path = _sweep_manifest_path(cfg)
    _atomic_json_write(
        manifest_path,
        {
            "schema_version": 1,
            "fingerprint": _sweep_fingerprint(cfg, planned_runs),
            "runs": planned_runs,
        },
    )
    _write_sweep_csv(cfg, planned_runs)
    return manifest_path


def _write_sweep_csv(cfg: dict[str, Any], planned_runs: list[dict[str, Any]]) -> Path:
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
