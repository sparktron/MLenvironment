"""Random morphology search: mutate morphology params, train, keep best.

Thin loop around :class:`~rl_framework.evolution.simple_search.RandomMorphologySearch`.
Each trial gets a mutated copy of ``cfg["environment"]["morphology"]`` and a
distinct ``experiment_name`` suffix so outputs don't overwrite each other.
"""
from __future__ import annotations

import copy
from typing import Any

from rl_framework.evolution.simple_search import RandomMorphologySearch


def run_morphology_search(
    cfg: dict[str, Any],
    trials: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Run ``trials`` train+eval cycles with mutated morphology and return the best.

    Returns a dict with ``best_trial``, ``best_params``, ``best_score``, and
    the per-trial ``results`` list. Relies on lazy imports so this module can
    be imported without pulling in SB3/torch.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")

    # Lazy imports keep this module cheap to import (e.g. for tests).
    from rl_framework.training.eval_runner import evaluate
    from rl_framework.training.sb3_runner import train

    if cfg.get("environment", {}).get("type") != "organism_arena_parallel":
        raise ValueError(
            "Morphology search currently targets 'organism_arena_parallel' envs"
        )

    searcher = RandomMorphologySearch(seed=seed)
    base_morph = cfg["environment"].get("morphology", {})
    base_name = cfg["experiment_name"]

    results: list[dict[str, Any]] = []
    best_idx = -1
    best_score = float("-inf")

    for i in range(trials):
        trial_cfg = copy.deepcopy(cfg)
        mutated = searcher.mutate(base_morph)
        trial_cfg["environment"]["morphology"] = mutated
        trial_cfg["experiment_name"] = f"{base_name}_morph_{i:03d}"

        model_path = train(trial_cfg)
        metrics = evaluate(trial_cfg, str(model_path))

        score = float(metrics.get("mean_return", float("-inf")))
        entry = {
            "trial": i,
            "experiment_name": trial_cfg["experiment_name"],
            "morphology": mutated,
            "model_path": str(model_path),
            "score": score,
            "metrics": metrics,
        }
        results.append(entry)

        if score > best_score:
            best_score = score
            best_idx = i

    return {
        "best_trial": best_idx,
        "best_params": results[best_idx]["morphology"] if best_idx >= 0 else {},
        "best_score": best_score,
        "results": results,
    }
