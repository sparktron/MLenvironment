from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from rl_framework.training.eval_runner import evaluate
from rl_framework.training.sb3_runner import train
from rl_framework.utils.logging_utils import create_experiment_paths


def run_multi_seed(cfg: dict[str, Any], seeds: list[int] | None = None) -> dict[str, float]:
    """Train and evaluate the same config across multiple seeds, then aggregate.

    Parameters
    ----------
    cfg:
        Experiment config dict (the same structure as a single-run config).
    seeds:
        List of integer seeds.  Falls back to ``cfg["multi_seed"]["seeds"]``
        or ``[0, 1, 2, 3, 4]`` when not provided.

    Returns
    -------
    dict with ``mean_return_mean``, ``mean_return_std``, and per-seed results.
    """
    if seeds is None:
        seeds = cfg.get("multi_seed", {}).get("seeds", [0, 1, 2, 3, 4])
    seeds = [int(s) for s in seeds]

    per_seed_metrics: list[dict[str, float]] = []
    model_paths: list[Path] = []

    for seed in seeds:
        run_cfg = deepcopy(cfg)
        run_cfg["seed"] = seed
        run_cfg["experiment_name"] = f"{cfg['experiment_name']}__seed_{seed}"

        # Train
        model_path = train(run_cfg)
        model_paths.append(model_path)

        # Evaluate
        metrics = evaluate(run_cfg, str(model_path) + ".zip")
        per_seed_metrics.append(metrics)
        print(f"[MultiSeed] seed={seed}  metrics={metrics}")

    # Aggregate across seeds
    all_returns = [m.get("mean_return", m.get("mean_reward", 0.0)) for m in per_seed_metrics]
    aggregate = {
        "mean_return_mean": float(np.mean(all_returns)),
        "mean_return_std": float(np.std(all_returns)),
        "seeds": seeds,
        "per_seed": per_seed_metrics,
    }

    # Write summary CSV
    base_dir = cfg.get("output", {}).get("base_dir", "outputs")
    summary_dir = Path(base_dir) / cfg["experiment_name"] / "multi_seed_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "aggregate.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "mean_return"])
        for seed, ret in zip(seeds, all_returns):
            writer.writerow([seed, ret])
        writer.writerow([])
        writer.writerow(["aggregate_mean", aggregate["mean_return_mean"]])
        writer.writerow(["aggregate_std", aggregate["mean_return_std"]])

    print(f"[MultiSeed] Aggregate: {aggregate['mean_return_mean']:.4f} +/- {aggregate['mean_return_std']:.4f}")
    print(f"[MultiSeed] Summary written to {summary_path}")
    return aggregate
