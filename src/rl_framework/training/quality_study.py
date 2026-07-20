"""Reproducible Priority-3 learning-quality studies.

The runner deliberately keeps candidate settings out of shipped defaults until
they clear a multi-seed/budget gate. State is written after every training run
so long studies can resume without repeating completed work.
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl_framework.training.arena_tournament import run_tournament
from rl_framework.training.sb3_runner import train
from rl_framework.training.walker_diagnostics import (
    evaluate_walker_checkpoint,
    evaluate_walker_transfer_suite,
)
from rl_framework.utils.checkpoint import model_zip_path
from rl_framework.utils.config import load_config, to_container, validate_experiment_config
from rl_framework.utils.config_merge import set_nested


WALKER_VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    "legacy_reward_control": (
        "walker_ppo_baseline",
        {
            "environment.reward.alive_bonus": 5.0,
            "environment.reward.forward_velocity_weight": 1.5,
            "environment.reward.orientation_penalty_weight": 0.3,
        },
    ),
    "rebalanced_flat": ("walker_ppo_baseline", {}),
    "curriculum_flat": ("walker_curriculum_flat", {}),
    "curriculum_uneven": ("walker_curriculum_uneven", {}),
    "curriculum_obstacles": ("walker_curriculum_obstacles", {}),
}

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
    "contested_food": {"environment.resources.food_placement": "center"},
    "body_collision_damage": {"environment.battle_rules.collision_damage": 0.01},
}

ALGORITHM_CONFIGS = {
    "PPO": "walker_ppo_baseline",
    "SAC": "walker_sac_baseline",
    "TD3": "walker_td3_baseline",
}


class StopOnWallClock(BaseCallback):
    """Stop SB3 learning once a monotonic wall-clock budget is exhausted."""

    def __init__(self, seconds: float):
        super().__init__(verbose=0)
        self.seconds = float(seconds)
        self._started_at = 0.0

    def _on_training_start(self) -> None:
        self._started_at = time.perf_counter()

    def _on_step(self) -> bool:
        return time.perf_counter() - self._started_at < self.seconds


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def _load_cfg(config_name: str, config_dir: str | Path) -> dict[str, Any]:
    return to_container(load_config(config_name, config_dir))


def _apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        set_nested(cfg, key, value, strict=False)


def _prepare_cfg(
    cfg: dict[str, Any],
    *,
    experiment_name: str,
    seed: int,
    step_budget: int,
    output_dir: Path,
) -> dict[str, Any]:
    prepared = deepcopy(cfg)
    prepared["experiment_name"] = experiment_name
    prepared["seed"] = int(seed)
    prepared.setdefault("environment", {})["seed"] = int(seed)
    prepared.setdefault("training", {})["total_timesteps"] = int(step_budget)
    if str(prepared["training"].get("algorithm", "PPO")).upper() == "PPO":
        # PPO always finishes a complete rollout, so an oversized shipped
        # n_steps*num_envs would silently turn a 10k study smoke into a 49k
        # run. Shrink only when needed so the requested comparison budget is
        # respected within one small rollout of overshoot.
        configured_envs = int(prepared["training"].get("num_envs", 1))
        configured_steps = int(prepared["training"].get("n_steps", 1024))
        target_rollout = min(
            configured_envs * configured_steps,
            max(32, step_budget // 10),
        )
        num_envs = min(configured_envs, max(1, target_rollout // 2))
        n_steps = min(configured_steps, max(2, target_rollout // num_envs))
        prepared["training"]["num_envs"] = num_envs
        prepared["training"]["n_steps"] = n_steps
        rollout_size = n_steps * num_envs
        preferred_batch = min(
            int(prepared["training"].get("batch_size", 256)), rollout_size
        )
        prepared["training"]["batch_size"] = next(
            size
            for size in range(preferred_batch, 1, -1)
            if rollout_size % size == 0
        )
    prepared["training"]["checkpoint_every"] = min(
        int(prepared["training"].get("checkpoint_every", step_budget)), step_budget
    )
    prepared.setdefault("evaluation", {})["episodes"] = max(
        1, int(prepared.get("evaluation", {}).get("episodes", 5))
    )
    prepared.setdefault("output", {})["base_dir"] = str(output_dir / "models")
    prepared["output"].pop("run_id", None)
    validate_experiment_config(prepared)
    return prepared


def _study_identity(
    studies: list[str],
    seeds: list[int],
    step_budget: int | None,
    wall_clock_seconds: float,
    eval_episodes: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "studies": studies,
        "seeds": seeds,
        "step_budget": step_budget,
        "wall_clock_seconds": wall_clock_seconds,
        "eval_episodes": eval_episodes,
    }


def _new_state(identity: dict[str, Any]) -> dict[str, Any]:
    return {"identity": identity, "runs": {}, "results": {}, "completed": False}


def _load_state(path: Path, identity: dict[str, Any], resume: bool) -> dict[str, Any]:
    if not resume:
        return _new_state(identity)
    if not path.exists():
        raise FileNotFoundError(f"No quality-study state found at {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("identity") != identity:
        raise ValueError(
            "Quality-study resume state does not match this invocation; "
            "use a new output directory or the original budgets/seeds."
        )
    return state


def _train_run(
    key: str,
    cfg: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    *,
    wall_clock_seconds: float | None = None,
) -> dict[str, Any]:
    previous = state["runs"].get(key)
    if previous and previous.get("status") == "completed":
        path = Path(previous["model_path"])
        if path.is_file():
            return previous

    callbacks = (
        [StopOnWallClock(wall_clock_seconds)] if wall_clock_seconds is not None else None
    )
    record: dict[str, Any] = {"status": "running", "config": cfg}
    state["runs"][key] = record
    _atomic_json_write(state_path, state)
    started_at = time.perf_counter()
    try:
        model = train(cfg, extra_callbacks=callbacks)
        path = model_zip_path(model)
        elapsed = time.perf_counter() - started_at
        record.update(
            {
                "status": "completed",
                "model_path": str(path),
                "training_seconds": elapsed,
            }
        )
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "error": str(exc),
                "training_seconds": time.perf_counter() - started_at,
            }
        )
    _atomic_json_write(state_path, state)
    return record


def _behavior_score(terrain_results: dict[str, dict[str, Any]]) -> float:
    scores = []
    for result in terrain_results.values():
        metrics = result["deterministic"]
        scores.append(
            metrics["episode_length_mean"] / 800.0
            + min(max(metrics["forward_displacement_mean"], -2.0), 5.0) / 5.0
            + metrics["push_recovery_rate"]
            - metrics["fall_rate"]
            - (2.0 if result["verdict"] == "reward_hack_high_peak_z" else 0.0)
        )
    return float(np.mean(scores)) if scores else float("-inf")


def _aggregate_walker_results(
    results: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    by_variant: dict[str, list[float]] = {}
    for run_key, terrain_results in results.items():
        variant = run_key.split("/", 1)[0]
        by_variant.setdefault(variant, []).append(_behavior_score(terrain_results))
    rankings = [
        {
            "variant": variant,
            "behavior_score_mean": float(np.mean(values)),
            "behavior_score_std": float(np.std(values)),
            "seeds_completed": len(values),
        }
        for variant, values in by_variant.items()
    ]
    rankings.sort(key=lambda row: row["behavior_score_mean"], reverse=True)
    return {"rankings": rankings, "recommended_variant": rankings[0]["variant"] if rankings else None}


def _run_walker_study(
    *,
    seeds: list[int],
    step_budget: int,
    eval_episodes: int,
    config_dir: Path,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    diagnostics: dict[str, dict[str, dict[str, Any]]] = {}
    for variant, (config_name, overrides) in WALKER_VARIANTS.items():
        base = _load_cfg(config_name, config_dir)
        base.setdefault("environment", {})["observation"] = {
            "version": "v2",
            "coordinate_free": True,
        }
        _apply_overrides(base, overrides)
        for seed in seeds:
            key = f"{variant}/seed_{seed}"
            cfg = _prepare_cfg(
                base,
                experiment_name=f"quality_walker_{variant}",
                seed=seed,
                step_budget=step_budget,
                output_dir=output_dir,
            )
            run = _train_run(f"walker/{key}", cfg, state, state_path)
            if run.get("status") != "completed":
                continue
            if "diagnostics" not in run:
                run["diagnostics"] = evaluate_walker_transfer_suite(
                    cfg, run["model_path"], episodes=eval_episodes
                )
                _atomic_json_write(state_path, state)
            diagnostics[key] = run["diagnostics"]
    aggregate = _aggregate_walker_results(diagnostics)
    aggregate["promotion_ready"] = (
        len(seeds) >= 3
        and step_budget >= 300_000
        and all(row["seeds_completed"] == len(seeds) for row in aggregate["rankings"])
    )
    aggregate["runs"] = diagnostics
    return aggregate


def _relabel_tournament(result: dict[str, Any], labels_by_path: dict[str, str]) -> dict[str, Any]:
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
    result["ratings"] = {label_map[key]: value for key, value in result["ratings"].items()}
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
        paths = [model_paths[name][seed] for name in variants if seed in model_paths.get(name, {})]
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
                by_variant.setdefault(standing["competitor"], []).append(standing["elo"])
    rows = [
        {
            "variant": variant,
            "elo_mean": float(np.mean(values)),
            "elo_std": float(np.std(values)),
            "seeds_completed": len(values),
        }
        for variant, values in by_variant.items()
    ]
    rows.sort(key=lambda row: row["elo_mean"], reverse=True)
    return rows


def _arena_measurements(tournaments: list[dict[str, Any]]) -> dict[str, Any]:
    timeout_rates = []
    by_variant: dict[str, dict[str, list[float]]] = {}
    for tournament in tournaments:
        for match in tournament["matches"]:
            timeout_rates.append(float(match["timeout_rate"]))
            for label_key, metrics_key in (
                ("competitor", "competitor_episode_metrics"),
                ("opponent", "opponent_episode_metrics"),
            ):
                label = match[label_key]
                if label == "random":
                    continue
                for metric, value in match.get(metrics_key, {}).items():
                    by_variant.setdefault(label, {}).setdefault(metric, []).append(
                        float(value)
                    )
    return {
        "timeout_rate_mean": float(np.mean(timeout_rates)) if timeout_rates else 0.0,
        "per_variant_episode_metrics": {
            variant: {
                metric: float(np.mean(values)) for metric, values in metrics.items()
            }
            for variant, metrics in by_variant.items()
        },
    }


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
            run = _train_run(f"arena/resources/{variant}/seed_{seed}", cfg, state, state_path)
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
            run = _train_run(f"arena/depth/{variant}/seed_{seed}", cfg, state, state_path)
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
        "depth_native_measurements": _arena_measurements(depth_native_tournaments),
        "depth_tournaments": depth_tournaments,
        "depth_native_tournaments": depth_native_tournaments,
        "recommended_depth_variant": depth_ranking[0]["variant"] if depth_ranking else None,
        "promotion_ready": (
            len(seeds) >= 3
            and step_budget >= 30_000
            and all(row["seeds_completed"] == len(seeds) for row in resource_ranking)
        ),
    }


def _run_algorithm_study(
    *,
    seeds: list[int],
    step_budget: int,
    wall_clock_seconds: float,
    eval_episodes: int,
    config_dir: Path,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    canonical_env = _load_cfg("walker_ppo_baseline", config_dir)["environment"]
    for algorithm, config_name in ALGORITHM_CONFIGS.items():
        base = _load_cfg(config_name, config_dir)
        base["environment"] = deepcopy(canonical_env)
        for budget_mode in ("steps", "wall_clock"):
            for seed in seeds:
                configured_steps = (
                    step_budget
                    if budget_mode == "steps"
                    else max(step_budget * 100, 10_000_000)
                )
                cfg = _prepare_cfg(
                    base,
                    experiment_name=f"quality_algorithm_{budget_mode}_{algorithm.lower()}",
                    seed=seed,
                    step_budget=configured_steps,
                    output_dir=output_dir,
                )
                key = f"algorithms/{budget_mode}/{algorithm}/seed_{seed}"
                run = _train_run(
                    key,
                    cfg,
                    state,
                    state_path,
                    wall_clock_seconds=(
                        wall_clock_seconds if budget_mode == "wall_clock" else None
                    ),
                )
                if run.get("status") != "completed":
                    continue
                if "diagnostics" not in run:
                    run["diagnostics"] = evaluate_walker_checkpoint(
                        cfg, run["model_path"], episodes=eval_episodes
                    )
                    _atomic_json_write(state_path, state)
                results[f"{budget_mode}/{algorithm}/seed_{seed}"] = {
                    "training_seconds": run["training_seconds"],
                    **run["diagnostics"],
                }

    rankings = []
    for budget_mode in ("steps", "wall_clock"):
        for algorithm in ALGORITHM_CONFIGS:
            rows = [
                value
                for key, value in results.items()
                if key.startswith(f"{budget_mode}/{algorithm}/")
            ]
            if not rows:
                continue
            rankings.append(
                {
                    "budget_mode": budget_mode,
                    "algorithm": algorithm,
                    "deterministic_return_mean": float(
                        np.mean([row["deterministic"]["return_mean"] for row in rows])
                    ),
                    "deterministic_fall_rate": float(
                        np.mean([row["deterministic"]["fall_rate"] for row in rows])
                    ),
                    "model_timesteps_mean": float(
                        np.mean([row["model_timesteps"] for row in rows])
                    ),
                    "training_seconds_mean": float(
                        np.mean([row["training_seconds"] for row in rows])
                    ),
                    "seeds_completed": len(rows),
                }
            )
    return {
        "rankings": rankings,
        "runs": results,
        "comparison_ready": (
            len(seeds) >= 3
            and step_budget >= 300_000
            and wall_clock_seconds >= 300.0
            and all(row["seeds_completed"] == len(seeds) for row in rankings)
        ),
    }


def _format_report(result: dict[str, Any]) -> str:
    lines = ["# Learning Quality Study", ""]
    identity = result["identity"]
    lines.extend(
        [
            f"- Studies: {', '.join(identity['studies'])}",
            f"- Seeds: {identity['seeds']}",
            f"- Step budget override: {identity['step_budget']}",
            f"- Wall-clock budget: {identity['wall_clock_seconds']} seconds",
            "",
            "Candidate settings are promoted only when their result reports its readiness gate as true.",
            "",
        ]
    )
    for study, payload in result["results"].items():
        lines.extend([f"## {study.title()}", "", "```json", json.dumps(payload, indent=2), "```", ""])
    return "\n".join(lines)


def run_quality_study(
    study: str,
    *,
    seeds: list[int],
    config_dir: str | Path = "src/rl_framework/configs/experiments",
    output_dir: str | Path = "outputs/quality_studies",
    step_budget: int | None = None,
    wall_clock_seconds: float = 900.0,
    eval_episodes: int = 20,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run walker, arena, algorithm, or all Priority-3 quality studies."""
    studies = ["walker", "arena", "algorithms"] if study == "all" else [study]
    invalid = set(studies) - {"walker", "arena", "algorithms"}
    if invalid:
        raise ValueError(f"Unknown quality study: {sorted(invalid)}")
    if not seeds:
        raise ValueError("Quality studies need at least one seed")
    if step_budget is not None and step_budget <= 0:
        raise ValueError("step_budget must be positive")
    if wall_clock_seconds <= 0 or eval_episodes <= 0:
        raise ValueError("wall_clock_seconds and eval_episodes must be positive")

    output = Path(output_dir)
    identity = _study_identity(
        studies, seeds, step_budget, wall_clock_seconds, eval_episodes
    )
    plan = {
        "identity": identity,
        "planned_runs": {
            "walker": len(WALKER_VARIANTS) * len(seeds) if "walker" in studies else 0,
            "arena": (
                (len(ARENA_RESOURCE_VARIANTS) + len(ARENA_DEPTH_VARIANTS)) * len(seeds)
                if "arena" in studies
                else 0
            ),
            "algorithms": (
                len(ALGORITHM_CONFIGS) * 2 * len(seeds)
                if "algorithms" in studies
                else 0
            ),
        },
        "dry_run": dry_run,
    }
    if dry_run:
        return plan

    state_path = output / "state.json"
    state = _load_state(state_path, identity, resume)
    config_path = Path(config_dir)
    if "walker" in studies:
        state["results"]["walker"] = _run_walker_study(
            seeds=seeds,
            step_budget=step_budget or 750_000,
            eval_episodes=eval_episodes,
            config_dir=config_path,
            output_dir=output,
            state=state,
            state_path=state_path,
        )
    if "arena" in studies:
        state["results"]["arena"] = _run_arena_study(
            seeds=seeds,
            step_budget=step_budget or 30_000,
            eval_episodes=eval_episodes,
            config_dir=config_path,
            output_dir=output,
            state=state,
            state_path=state_path,
        )
    if "algorithms" in studies:
        state["results"]["algorithms"] = _run_algorithm_study(
            seeds=seeds,
            step_budget=step_budget or 500_000,
            wall_clock_seconds=wall_clock_seconds,
            eval_episodes=eval_episodes,
            config_dir=config_path,
            output_dir=output,
            state=state,
            state_path=state_path,
        )
    state["completed"] = all(
        run.get("status") == "completed" for run in state["runs"].values()
    )
    _atomic_json_write(state_path, state)
    report = {"identity": identity, "results": state["results"], "completed": state["completed"]}
    _atomic_json_write(output / "report.json", report)
    (output / "report.md").write_text(_format_report(report), encoding="utf-8")
    return report
