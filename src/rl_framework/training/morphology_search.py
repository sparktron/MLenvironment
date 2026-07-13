"""Random morphology search: mutate morphology params, train, keep best.

Thin loop around :class:`~rl_framework.evolution.simple_search.RandomMorphologySearch`.
Each trial gets a mutated copy of ``cfg["environment"]["morphology"]`` and a
distinct ``output.run_id`` (``morph_<i>``) so trials land in separate output
subtrees (``<experiment_name>/runs/morph_<i>/``) without overwriting each other.
"""

from __future__ import annotations

import copy
from typing import Any

from rl_framework.evolution.simple_search import RandomMorphologySearch
from rl_framework.utils.checkpoint import model_zip_path


def _as_model_zip_path(model_path: str) -> str:
    """Return a Stable-Baselines3 model zip path as a string.

    Thin string-returning wrapper over :func:`model_zip_path` so the morphology
    loop shares the one canonical path-normalisation helper.
    """
    return str(model_zip_path(model_path))


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
    from rl_framework.training.sb3_runner import train
    scoring = str(cfg.get("morphology_search", {}).get("scoring", "mean_return"))
    if scoring not in {"mean_return", "tournament_elo"}:
        raise ValueError("morphology_search.scoring must be 'mean_return' or 'tournament_elo'")
    if scoring == "mean_return":
        from rl_framework.training.eval_runner import evaluate
    else:
        from rl_framework.training.arena_tournament import run_tournament

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
        # Each trial is a distinct run under the same experiment. Route it
        # through output.run_id rather than mutating experiment_name, so trials
        # land at outputs/<experiment_name>/runs/morph_<i>/seed_<seed>/.
        run_id = f"morph_{i:03d}"
        trial_cfg.setdefault("output", {})["run_id"] = run_id

        model_path = train(trial_cfg)
        model_zip = _as_model_zip_path(str(model_path))
        if scoring == "mean_return":
            metrics = evaluate(trial_cfg, model_zip)
            score = float(metrics.get("mean_return", float("-inf")))
        else:
            field = [model_zip, *[_as_model_zip_path(entry["model_path"]) for entry in results]]
            tournament = run_tournament(field, trial_cfg, n_episodes=int(cfg.get("morphology_search", {}).get("tournament_episodes", 10)), include_random=True)
            label = next(item["label"] for item in tournament["competitors"] if item["path"] == model_zip)
            score = float(tournament["ratings"][label])
            metrics = {"elo": score, "tournament": tournament}
        entry = {
            "trial": i,
            "experiment_name": base_name,
            "run_id": run_id,
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
