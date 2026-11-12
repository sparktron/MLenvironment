"""Arena learning-quality studies."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np

from rl_framework.training.arena_tournament import run_tournament
from rl_framework.training.quality_study import (
    _apply_overrides, _load_cfg, _prepare_cfg, _train_run,
)


ARENA_RESOURCE_VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "scarce_high_cost": {
        "environment.resources.movement_cost": 0.015,
        "environment.resources.attack_cost": 0.06,
        "environment.resources.food_count": 1,
        "environment.resources.food_respawn_steps": 60,
    },
    "abundant_low_cost": {
        "environment.resources.movement_cost": 0.006,
        "environment.resources.attack_cost": 0.025,
        "environment.resources.food_count": 4,
        "environment.resources.food_respawn_steps": 25,
    },
    "large_slow": {
        "environment.morphology.base_size": 1.25,
        "environment.sim.speed_size_exponent": 1.25,
    },
}


ARENA_DEPTH_VARIANTS: dict[str, dict[str, Any]] = {
    "feasible_combat": {
        "environment.battle_rules.attack_range": 0.4,
        "environment.resources.attack_cost": 0.02,
    },
    "feasible_combat_approach": {
        "environment.battle_rules.attack_range": 0.4,
        "environment.resources.attack_cost": 0.02,
        "environment.reward.approach_weight": 0.02,
    },
}


ARENA_BEHAVIOR_GATE = {
    "max_timeout_rate": 0.90,
    "min_attack_hits": 0.01,
    "min_damage_dealt": 0.001,
}


def _relabel_tournament(
    result: dict[str, Any], labels_by_path: dict[str, str]
) -> dict[str, Any]:
    label_map = {
        competitor["label"]: labels_by_path.get(competitor["path"], competitor["label"])
        for competitor in result["competitors"]
    }
    for competitor in result["competitors"]:
        competitor["label"] = label_map[competitor["label"]]
    for standing in result["standings"]:
        standing["competitor"] = label_map[standing["competitor"]]
    for match in result["matches"]:
        match["competitor"] = label_map[match["competitor"]]
        match["opponent"] = label_map[match["opponent"]]
    result["ratings"] = {
        label_map[key]: value for key, value in result["ratings"].items()
    }
    result["win_rate_matrix"] = {
        label_map[left]: {label_map[right]: value for right, value in row.items()}
        for left, row in result["win_rate_matrix"].items()
    }
    return result


def _arena_variant_cfg(
    base: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    cfg = deepcopy(base)
    # These optional strategic-depth fields must exist before strict config
    # preparation/validation and remain off in the baseline.
    cfg.setdefault("environment", {}).setdefault("battle_rules", {}).setdefault(
        "collision_damage", 0.0
    )
    cfg["environment"].setdefault("resources", {}).setdefault(
        "food_placement", "uniform"
    )
    cfg["environment"].setdefault("reward", {}).setdefault("approach_weight", 0.0)
    _apply_overrides(cfg, overrides)
    return cfg


def _arena_tournaments(
    variants: dict[str, dict[str, Any]],
    model_paths: dict[str, dict[int, str]],
    seeds: list[int],
    base_cfg: dict[str, Any],
    eval_episodes: int,
) -> list[dict[str, Any]]:
    tournaments = []
    for seed in seeds:
        paths = [
            model_paths[name][seed]
            for name in variants
            if seed in model_paths.get(name, {})
        ]
        if len(paths) < 2:
            continue
        labels = {
            model_paths[name][seed]: name
            for name in variants
            if seed in model_paths.get(name, {})
        }
        eval_cfg = deepcopy(base_cfg)
        eval_cfg["seed"] = seed
        eval_cfg["environment"]["seed"] = seed
        tournament = run_tournament(
            paths,
            eval_cfg,
            n_episodes=eval_episodes,
            swap_roles=True,
            include_random=True,
        )
        tournaments.append(_relabel_tournament(tournament, labels))
    return tournaments


def _arena_native_tournaments(
    variants: dict[str, dict[str, Any]],
    model_paths: dict[str, dict[int, str]],
    seeds: list[int],
    base_cfg: dict[str, Any],
    eval_episodes: int,
) -> list[dict[str, Any]]:
    """Evaluate each policy in the resource/mechanics regime that trained it."""
    tournaments = []
    for variant, overrides in variants.items():
        native_cfg = _arena_variant_cfg(base_cfg, overrides)
        for seed in seeds:
            path = model_paths.get(variant, {}).get(seed)
            if path is None:
                continue
            native_cfg["seed"] = seed
            native_cfg["environment"]["seed"] = seed
            tournament = run_tournament(
                [path],
                native_cfg,
                n_episodes=eval_episodes,
                swap_roles=True,
                include_random=True,
            )
            tournament = _relabel_tournament(tournament, {path: variant})
            tournament["environment_variant"] = variant
            tournaments.append(tournament)
    return tournaments


def _arena_ranking(tournaments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[float]] = {}
    for tournament in tournaments:
        for standing in tournament["standings"]:
            if standing["competitor"] != "random":
                by_variant.setdefault(standing["competitor"], []).append(
                    standing["elo"]
                )
    rows = [
        {
            "variant": variant,
            "elo_mean": float(np.mean(values)),
            "elo_std": float(np.std(values)),
            "seeds_completed": len(values),
        }
        for variant, values in by_variant.items()
    ]
    rows.sort(key=lambda row: cast(float, row["elo_mean"]), reverse=True)
    return rows


def _arena_measurements(tournaments: list[dict[str, Any]]) -> dict[str, Any]:
    timeout_rates = []
    timeouts_by_variant: dict[str, list[float]] = {}
    by_variant: dict[str, dict[str, list[float]]] = {}
    for tournament in tournaments:
        for match in tournament["matches"]:
            timeout_rate = float(match["timeout_rate"])
            timeout_rates.append(timeout_rate)
            for label_key, metrics_key in (
                ("competitor", "competitor_episode_metrics"),
                ("opponent", "opponent_episode_metrics"),
            ):
                label = match[label_key]
                if label == "random":
                    continue
                timeouts_by_variant.setdefault(label, []).append(timeout_rate)
                for metric, value in match.get(metrics_key, {}).items():
                    by_variant.setdefault(label, {}).setdefault(metric, []).append(
                        float(value)
                    )
    return {
        "timeout_rate_mean": float(np.mean(timeout_rates)) if timeout_rates else 0.0,
        "per_variant_timeout_rate": {
            variant: float(np.mean(values))
            for variant, values in timeouts_by_variant.items()
        },
        "per_variant_episode_metrics": {
            variant: {
                metric: float(np.mean(values)) for metric, values in metrics.items()
            }
            for variant, metrics in by_variant.items()
        },
    }


def _arena_behavior_results(
    measurements: dict[str, Any], variants: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    results = {}
    for variant in variants:
        metrics = measurements["per_variant_episode_metrics"].get(variant, {})
        timeout_rate = measurements["per_variant_timeout_rate"].get(variant, 1.0)
        attack_hits = float(metrics.get("attack_hits", 0.0))
        damage_dealt = float(metrics.get("damage_dealt", 0.0))
        results[variant] = {
            "timeout_rate": timeout_rate,
            "attack_hits": attack_hits,
            "damage_dealt": damage_dealt,
            "passed": bool(
                timeout_rate < ARENA_BEHAVIOR_GATE["max_timeout_rate"]
                and attack_hits >= ARENA_BEHAVIOR_GATE["min_attack_hits"]
                and damage_dealt >= ARENA_BEHAVIOR_GATE["min_damage_dealt"]
            ),
        }
    return results


def _run_arena_study(
    *,
    seeds: list[int],
    step_budget: int,
    eval_episodes: int,
    config_dir: Path,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    base = _arena_variant_cfg(_load_cfg("organisms_fight_arena", config_dir), {})
    model_paths: dict[str, dict[int, str]] = {}
    for variant, overrides in ARENA_RESOURCE_VARIANTS.items():
        variant_cfg = _arena_variant_cfg(base, overrides)
        for seed in seeds:
            cfg = _prepare_cfg(
                variant_cfg,
                experiment_name=f"quality_arena_{variant}",
                seed=seed,
                step_budget=step_budget,
                output_dir=output_dir,
            )
            run = _train_run(
                f"arena/resources/{variant}/seed_{seed}", cfg, state, state_path
            )
            if run.get("status") == "completed":
                model_paths.setdefault(variant, {})[seed] = run["model_path"]

    resource_tournaments = _arena_tournaments(
        ARENA_RESOURCE_VARIANTS, model_paths, seeds, base, eval_episodes
    )
    resource_native_tournaments = _arena_native_tournaments(
        ARENA_RESOURCE_VARIANTS, model_paths, seeds, base, eval_episodes
    )
    resource_ranking = _arena_ranking(resource_tournaments)

    # Strategic-depth candidates are deliberately trained/evaluated only after
    # the resource tournaments above have produced the baseline measurement.
    depth_paths: dict[str, dict[int, str]] = {
        "baseline": model_paths.get("baseline", {})
    }
    for variant, overrides in ARENA_DEPTH_VARIANTS.items():
        variant_cfg = _arena_variant_cfg(base, overrides)
        for seed in seeds:
            cfg = _prepare_cfg(
                variant_cfg,
                experiment_name=f"quality_arena_depth_{variant}",
                seed=seed,
                step_budget=step_budget,
                output_dir=output_dir,
            )
            run = _train_run(
                f"arena/depth/{variant}/seed_{seed}", cfg, state, state_path
            )
            if run.get("status") == "completed":
                depth_paths.setdefault(variant, {})[seed] = run["model_path"]
    depth_variants = {"baseline": {}, **ARENA_DEPTH_VARIANTS}
    depth_tournaments = _arena_tournaments(
        depth_variants, depth_paths, seeds, base, eval_episodes
    )
    depth_native_tournaments = _arena_native_tournaments(
        depth_variants, depth_paths, seeds, base, eval_episodes
    )
    depth_ranking = _arena_ranking(depth_tournaments)
    depth_native_measurements = _arena_measurements(depth_native_tournaments)
    behavior_results = _arena_behavior_results(
        depth_native_measurements, depth_variants
    )
    promoted_depth_variant = next(
        (
            row["variant"]
            for row in depth_ranking
            if behavior_results.get(str(row["variant"]), {}).get("passed", False)
        ),
        None,
    )
    evidence_ready = (
        len(seeds) >= 3
        and step_budget >= 30_000
        and len(resource_ranking) == len(ARENA_RESOURCE_VARIANTS)
        and len(depth_ranking) == len(depth_variants)
        and all(
            row["seeds_completed"] == len(seeds)
            for row in (*resource_ranking, *depth_ranking)
        )
    )
    return {
        "resource_rankings": resource_ranking,
        "resource_measurements": _arena_measurements(resource_tournaments),
        "resource_native_measurements": _arena_measurements(
            resource_native_tournaments
        ),
        "resource_tournaments": resource_tournaments,
        "resource_native_tournaments": resource_native_tournaments,
        "recommended_resource_variant": (
            resource_ranking[0]["variant"] if resource_ranking else None
        ),
        "depth_rankings": depth_ranking,
        "depth_measurements": _arena_measurements(depth_tournaments),
        "depth_native_measurements": depth_native_measurements,
        "depth_tournaments": depth_tournaments,
        "depth_native_tournaments": depth_native_tournaments,
        "recommended_depth_variant": depth_ranking[0]["variant"]
        if depth_ranking
        else None,
        "behavioral_gate": ARENA_BEHAVIOR_GATE,
        "behavioral_results": behavior_results,
        "promoted_depth_variant": promoted_depth_variant,
        "evidence_ready": evidence_ready,
        "promotion_ready": bool(evidence_ready and promoted_depth_variant is not None),
    }



def study_specs():
    from rl_framework.core.studies import StudySpec

    return (StudySpec("arena", "arena", _run_arena_study,
        len(ARENA_RESOURCE_VARIANTS) + len(ARENA_DEPTH_VARIANTS),
        30_000, 7, include_in_all=True),)
