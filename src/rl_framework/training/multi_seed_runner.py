from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from rl_framework.training.eval_runner import evaluate
from rl_framework.training.sb3_runner import train
from rl_framework.utils.logging_utils import create_experiment_paths


def _run_one_seed(args: tuple[int, dict[str, Any]]) -> tuple[int, str, dict[str, float]]:
    """Train + evaluate a single seed. Runs in a subprocess when parallelised."""
    seed, cfg = args
    model_path = train(cfg)
    metrics = evaluate(cfg, str(model_path) + ".zip")
    return seed, str(model_path), metrics


def run_multi_seed(
    cfg: dict[str, Any],
    seeds: list[int] | None = None,
    max_workers: int | None = None,
) -> dict[str, float]:
    """Train and evaluate the same config across multiple seeds, then aggregate.

    Parameters
    ----------
    cfg:
        Experiment config dict (the same structure as a single-run config).
    seeds:
        List of integer seeds.  Falls back to ``cfg["multi_seed"]["seeds"]``
        or ``[0, 1, 2, 3, 4]`` when not provided.
    max_workers:
        Number of parallel worker processes.  Defaults to ``min(len(seeds), cpu_count)``.
        Pass ``1`` to force sequential execution (useful for debugging or when
        the training itself already saturates all CPUs via SubprocVecEnv).

    Returns
    -------
    dict with ``mean_return_mean``, ``mean_return_std``, and per-seed results.
    """
    if seeds is None:
        seeds = cfg.get("multi_seed", {}).get("seeds", [0, 1, 2, 3, 4])
    seeds = [int(s) for s in seeds]

    if max_workers is None:
        max_workers = min(len(seeds), os.cpu_count() or 1)

    # Build per-seed configs up front.
    seed_args: list[tuple[int, dict[str, Any]]] = []
    for seed in seeds:
        run_cfg = deepcopy(cfg)
        run_cfg["seed"] = seed
        run_cfg["experiment_name"] = f"{cfg['experiment_name']}__seed_{seed}"
        seed_args.append((seed, run_cfg))

    per_seed_metrics: dict[int, dict[str, float]] = {}
    failed_seeds: dict[int, str] = {}
    model_paths: dict[int, str] = {}

    if max_workers == 1:
        # Sequential path — avoids subprocess overhead and is easier to debug.
        for seed, run_cfg in seed_args:
            try:
                seed, model_path, metrics = _run_one_seed((seed, run_cfg))
                model_paths[seed] = model_path
                per_seed_metrics[seed] = metrics
                print(f"[MultiSeed] seed={seed}  metrics={metrics}")
            except Exception as exc:
                failed_seeds[seed] = str(exc)
                print(f"[MultiSeed] seed={seed}  FAILED: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            future_to_seed = {exe.submit(_run_one_seed, args): args[0] for args in seed_args}
            for future in as_completed(future_to_seed):
                seed = future_to_seed[future]
                try:
                    seed, model_path, metrics = future.result()
                    model_paths[seed] = model_path
                    per_seed_metrics[seed] = metrics
                    print(f"[MultiSeed] seed={seed}  metrics={metrics}")
                except Exception as exc:
                    failed_seeds[seed] = str(exc)
                    print(f"[MultiSeed] seed={seed}  FAILED: {exc}")

    if not per_seed_metrics:
        raise RuntimeError(
            f"All seeds failed. First failure: {next(iter(failed_seeds.values()))}"
        )

    # Collect only successful seeds for aggregation.
    successful_seeds = [s for s in seeds if s in per_seed_metrics]
    ordered_metrics = [per_seed_metrics[s] for s in successful_seeds]
    all_returns = [m.get("mean_return", m.get("mean_reward", 0.0)) for m in ordered_metrics]
    aggregate: dict[str, Any] = {
        "mean_return_mean": float(np.mean(all_returns)),
        "mean_return_std": float(np.std(all_returns)),
        "seeds": seeds,
        "successful_seeds": successful_seeds,
        "failed_seeds": failed_seeds,
        "per_seed": ordered_metrics,
    }

    # Write summary CSV
    base_dir = cfg.get("output", {}).get("base_dir", "outputs")
    summary_dir = Path(base_dir) / cfg["experiment_name"] / "multi_seed_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "aggregate.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "mean_return"])
        for seed, ret in zip(successful_seeds, all_returns):
            writer.writerow([seed, ret])
        for seed, err in failed_seeds.items():
            writer.writerow([seed, f"FAILED: {err}"])
        writer.writerow([])
        writer.writerow(["aggregate_mean", aggregate["mean_return_mean"]])
        writer.writerow(["aggregate_std", aggregate["mean_return_std"]])

    print(f"[MultiSeed] Aggregate: {aggregate['mean_return_mean']:.4f} +/- {aggregate['mean_return_std']:.4f}")
    if failed_seeds:
        print(f"[MultiSeed] WARNING: {len(failed_seeds)} seed(s) failed: {list(failed_seeds.keys())}")
    print(f"[MultiSeed] Summary written to {summary_path}")
    return aggregate
