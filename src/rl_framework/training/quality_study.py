"""Reproducible Priority-3 learning-quality studies.

The runner deliberately keeps candidate settings out of shipped defaults until
they clear a multi-seed/budget gate. State is written after every training run
so long studies can resume without repeating completed work.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rl_framework.training.arena_tournament import run_tournament
from rl_framework.training.sb3_runner import train
from rl_framework.training.walker_diagnostics import (
    calibrate_walker_slip_thresholds,
    collect_walker_fall_recovery_telemetry,
    evaluate_walker_checkpoint,
    evaluate_walker_transfer_suite,
    summarize_walker_fall_windows,
)
from rl_framework.training.velocity_target_ramp_callback import (
    VelocityTargetRampCallback,
)
from rl_framework.utils.checkpoint import model_zip_path
from rl_framework.utils.config import (
    load_config,
    to_container,
    validate_experiment_config,
)
from rl_framework.utils.config_merge import set_nested


WALKER_VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    "balance_low_std": (
        "walker_balance_curriculum",
        {
            "training.log_std_init": -1.0,
            "curriculum.warmup_steps": 200_000,
        },
    ),
    "balance_low_std_low_entropy": (
        "walker_balance_curriculum",
        {
            "training.log_std_init": -1.0,
            "training.ent_coef": 0.0,
            "curriculum.warmup_steps": 200_000,
        },
    ),
    "balance_low_std_conservative": (
        "walker_balance_curriculum",
        {
            "training.log_std_init": -1.5,
            "training.ent_coef": 0.0,
            "environment.sim.action_scale": 0.15,
            "environment.reward.orientation_penalty_weight": 1.5,
            "environment.reward.fall_penalty": 40.0,
            "curriculum.warmup_steps": 200_000,
        },
    ),
    "balance_low_std_slow_scale": (
        "walker_balance_curriculum",
        {
            "training.log_std_init": -1.5,
            "training.ent_coef": 0.0,
            "environment.sim.action_scale": 0.10,
            "environment.reward.orientation_penalty_weight": 1.5,
            "environment.reward.fall_penalty": 40.0,
            "curriculum.warmup_steps": 200_000,
            "curriculum.level_params": {
                1: {
                    "sim.action_scale": 0.25,
                    "reward.alive_bonus": 0.75,
                    "reward.forward_velocity_weight": 1.0,
                    "reward.target_velocity": 0.15,
                    "reward.orientation_penalty_weight": 1.0,
                },
                2: {
                    "sim.action_scale": 0.50,
                    "reward.alive_bonus": 0.50,
                    "reward.forward_velocity_weight": 2.0,
                    "reward.target_velocity": 0.40,
                    "reward.orientation_penalty_weight": 0.7,
                },
            },
        },
    ),
}

WALKER_STUDY_TRAINING = {
    "num_envs": 24,
    "n_steps": 128,
    "batch_size": 256,
}
WALKER_BEHAVIOR_GATE: dict[str, Any] = {
    "min_episode_length": 760.0,
    "min_forward_displacement": 3.0,
    "max_fall_rate": 0.10,
    "max_peak_z": 1.0,
}
WALKER_BALANCE_GATE: dict[str, Any] = {
    "evaluation_mode": "stochastic",
    "terrain": "flat",
    "min_episode_length": 600.0,
    "max_fall_rate": 0.30,
    "max_abs_forward_displacement": 0.75,
    "max_peak_z": 1.0,
}
WALKER_VELOCITY_VARIANTS: dict[str, dict[str, Any]] = {
    "velocity_sigma_015": {
        "environment.reward.velocity_sigma": 0.15,
    },
    "velocity_sigma_010": {
        "environment.reward.velocity_sigma": 0.10,
    },
    "velocity_sigma_015_weight_15": {
        "environment.reward.velocity_sigma": 0.15,
        "environment.reward.forward_velocity_weight": 1.5,
    },
}
WALKER_VELOCITY_GATE: dict[str, Any] = {
    "evaluation_mode": "stochastic",
    "min_episode_length": 600.0,
    "min_forward_displacement": 1.0,
    "max_fall_rate": 0.30,
    "max_peak_z": 1.0,
}
WALKER_GAIT_VARIANTS: dict[str, dict[str, Any]] = {
    "control": {},
    "alternating_progress": {
        "environment.reward.gait_step_progress_weight": 40.0,
        "environment.reward.gait_step_progress_clip": 0.03,
    },
    "stance_slip": {
        "environment.reward.stance_slip_penalty_weight": 0.5,
    },
    "combined": {
        "environment.reward.gait_step_progress_weight": 40.0,
        "environment.reward.gait_step_progress_clip": 0.03,
        "environment.reward.stance_slip_penalty_weight": 0.5,
    },
}
# Gait thresholds are a *band*, not a set of floors. The 2026-07-26 correction
# (see docs/walker_learning_history.md) found that the original "at least as
# good as the observed baseline" calibration froze the chatter it was meant to
# diagnose: the baseline shuffled at ~20 alternating touchdowns per 100 steps
# with 10.9 mm strides and 0.7 mm clearance, so a floor of 15 demanded ~11 Hz
# stepping and *rejected* real walking.
#
# Cadence is derived from gait mechanics rather than from that baseline. One
# gait cycle produces two alternating touchdowns, so at 60 Hz control:
#     alternating touchdowns / 100 steps = 100 * 2 * f / 60 = 3.33 * f
# for a cycle frequency f in Hz. Plausible bipedal walking spans f = 0.45-2.5 Hz
# (human/Atlas-class walking sits at 0.8-1.4 Hz), giving 1.5-8.5. Anything
# above the ceiling is contact chatter; anything below it is not stepping.
#
# Stride and clearance floors make a touchdown mean something. An open-loop
# scripted gait at this study's own `action_scale: 0.25` / `position_gain: 0.1`
# reached 33.7 mm of foot clearance, so 20 mm is ~60% of demonstrated actuator
# capability rather than an invented number. Progress per alternating touchdown
# is set to the minimum consistent with the 3 m success gate at the fastest
# permitted cadence (3 m / ~67 touchdowns), and a same-foot stride is one full
# cycle, i.e. about twice the per-touchdown pelvis advance.
#
# Support occupancy is derived from the configured anti-phase reference rather
# than candidate outcomes. With stance_duty=0.6, the nominal schedule has 20%
# double support and no flight. The gate requires half of that nominal overlap
# and permits an equal 10% raw-contact tolerance for flight. This excludes
# running/bounding while leaving room for contact-estimation noise.
#
# Slip is the planar velocity of the slowest material contact point, not the
# foot-link origin. Link-origin velocity includes ordinary heel/toe pivoting and
# is therefore not a physical measure of sliding. The existing conservative
# 0.18 m/s ceiling is retained; changing the instrument does not justify
# loosening the threshold.
WALKER_GAIT_GATE: dict[str, Any] = {
    **WALKER_VELOCITY_GATE,
    "gate_version": 2,
    "min_alternating_touchdowns_per_100_steps": 1.5,
    "max_alternating_touchdowns_per_100_steps": 8.5,
    "min_progress_per_alternating_touchdown": 0.05,
    "min_stride_length": 0.10,
    "min_foot_clearance": 0.02,
    "slip_metric": "gait_contact_point_slip_speed_mean",
    "max_slip_speed": 0.18,
    "min_double_support_fraction": 0.10,
    "max_flight_fraction": 0.10,
    "max_longest_same_foot_sequence": 5.0,
}
WALKER_VELOCITY_RAMP_VARIANTS: dict[str, dict[str, Any]] = {
    "control": {},
    "ramp_to_100": {"target_velocity_end": 1.0},
}
WALKER_VELOCITY_RAMP_STEPS = 3_000_000
WALKER_VELOCITY_RAMP_GATE: dict[str, Any] = {
    **WALKER_GAIT_GATE,
    "min_displacement_improvement": 0.25,
}
WALKER_SWING_TOUCHDOWN_VARIANTS: dict[str, dict[str, Any]] = {
    "control": {},
    "swing_qualified_progress": {
        "environment.gait.min_swing_duration": 0.05,
        "environment.gait.min_foot_clearance": 0.0005,
        "environment.reward.swing_touchdown_progress_weight": 40.0,
        "environment.reward.swing_touchdown_progress_clip": 0.03,
    },
}
WALKER_SWING_TOUCHDOWN_GATE: dict[str, Any] = {
    **WALKER_GAIT_GATE,
    "min_displacement_improvement": 0.25,
}
WALKER_SLIP_RECOVERY_VARIANTS: dict[str, dict[str, Any]] = {
    "control": {},
    "slip_cost_025": {
        "environment.reward.stance_slip_penalty_weight": 0.25,
    },
    "slip_cost_050": {
        "environment.reward.stance_slip_penalty_weight": 0.5,
    },
}
WALKER_CONTACT_DAMPING_VARIANTS: dict[str, dict[str, Any]] = {
    "contact_damping_075": {
        "environment.sim.control.velocity_gain": 0.75,
    },
    "contact_damping_050": {
        "environment.sim.control.velocity_gain": 0.5,
    },
}
WALKER_RECOVERY_NOISE_MULTIPLIER = 2.0
WALKER_SLIP_RECOVERY_GATE: dict[str, Any] = {
    "evaluation_mode": "stochastic",
    "max_fall_rate": 0.30,
    "slip_metric": WALKER_GAIT_GATE["slip_metric"],
    "max_slip_speed": WALKER_GAIT_GATE["max_slip_speed"],
    "min_forward_displacement": 5.0,
    "max_peak_z": 1.0,
}
# The slip-recovery screen found that a bounded 0.5 slip cost was the strongest
# tested contact intervention, but stochastic actions still produced excessive
# slip and contact chatter. The next isolated question is whether observing the
# previously applied command gives PPO enough short-term state to recover and
# sustain a real swing. Both new lineages keep the measured slip intervention
# fixed; only the opt-in observation changes.
WALKER_ACTION_MEMORY_VARIANTS: dict[str, dict[str, Any]] = {
    "control": {
        "environment.observation.include_previous_action": False,
    },
    "previous_action": {
        "environment.observation.include_previous_action": True,
    },
}
WALKER_ACTION_MEMORY_SLIP_CLIP = 1.3398472589562194
WALKER_ACTION_MEMORY_SCREEN_STEPS = 10_000_000

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
            size for size in range(preferred_batch, 1, -1) if rollout_size % size == 0
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
    resume_from: str | Path | None = None,
    extra_callbacks: list[BaseCallback] | None = None,
) -> dict[str, Any]:
    previous = state["runs"].get(key)
    if previous and previous.get("status") == "completed":
        path = Path(previous["model_path"])
        if path.is_file():
            return previous

    callbacks: list[BaseCallback] = list(extra_callbacks or [])
    if wall_clock_seconds is not None:
        callbacks.append(StopOnWallClock(wall_clock_seconds))
    record: dict[str, Any] = {
        "status": "running",
        "config": cfg,
        "resume_from": str(resume_from) if resume_from is not None else None,
    }
    state["runs"][key] = record
    _atomic_json_write(state_path, state)
    started_at = time.perf_counter()
    try:
        model = train(
            cfg,
            extra_callbacks=callbacks or None,
            resume_from=resume_from,
        )
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


def _walker_behavior_passed(result: dict[str, Any]) -> bool:
    """Return whether either action mode clears the robust-locomotion gate."""
    for mode in ("deterministic", "stochastic"):
        metrics = result[mode]
        if (
            metrics["episode_length_mean"] >= WALKER_BEHAVIOR_GATE["min_episode_length"]
            and metrics["forward_displacement_mean"]
            >= WALKER_BEHAVIOR_GATE["min_forward_displacement"]
            and metrics["fall_rate"] <= WALKER_BEHAVIOR_GATE["max_fall_rate"]
            and metrics["peak_z_mean"] < WALKER_BEHAVIOR_GATE["max_peak_z"]
        ):
            return True
    return False


def _walker_balance_passed(result: dict[str, Any]) -> bool:
    """Return whether stochastic actions produce stable, non-exploitative balance."""
    metrics = result[WALKER_BALANCE_GATE["evaluation_mode"]]
    return bool(
        metrics["episode_length_mean"] >= WALKER_BALANCE_GATE["min_episode_length"]
        and metrics["fall_rate"] <= WALKER_BALANCE_GATE["max_fall_rate"]
        and abs(metrics["forward_displacement_mean"])
        <= WALKER_BALANCE_GATE["max_abs_forward_displacement"]
        and metrics["peak_z_mean"] < WALKER_BALANCE_GATE["max_peak_z"]
    )


def _aggregate_walker_results(
    results: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    by_variant: dict[str, list[float]] = {}
    behavior_by_variant: dict[str, list[bool]] = {}
    balance_by_variant: dict[str, list[bool]] = {}
    for run_key, terrain_results in results.items():
        variant = run_key.split("/", 1)[0]
        by_variant.setdefault(variant, []).append(_behavior_score(terrain_results))
        behavior_by_variant.setdefault(variant, []).extend(
            _walker_behavior_passed(result) for result in terrain_results.values()
        )
        flat_result = terrain_results.get(WALKER_BALANCE_GATE["terrain"])
        if flat_result is not None:
            balance_by_variant.setdefault(variant, []).append(
                _walker_balance_passed(flat_result)
            )
    rankings = [
        {
            "variant": variant,
            "behavior_score_mean": float(np.mean(values)),
            "behavior_score_std": float(np.std(values)),
            "seeds_completed": len(values),
            "behavioral_gate_passed": bool(
                behavior_by_variant.get(variant) and all(behavior_by_variant[variant])
            ),
            "behavioral_eval_pass_rate": float(
                np.mean(behavior_by_variant.get(variant, [False]))
            ),
            "balance_gate_passed": bool(
                balance_by_variant.get(variant) and all(balance_by_variant[variant])
            ),
            "balance_eval_pass_rate": float(
                np.mean(balance_by_variant.get(variant, [False]))
            ),
        }
        for variant, values in by_variant.items()
    ]
    rankings.sort(key=lambda row: cast(float, row["behavior_score_mean"]), reverse=True)
    promoted = next(
        (row["variant"] for row in rankings if row["behavioral_gate_passed"]),
        None,
    )
    balance_candidate = next(
        (row["variant"] for row in rankings if row["balance_gate_passed"]),
        None,
    )
    return {
        "rankings": rankings,
        "recommended_variant": rankings[0]["variant"] if rankings else None,
        "balance_candidate": balance_candidate,
        "promoted_variant": promoted,
        "balance_gate": WALKER_BALANCE_GATE,
        "behavioral_gate": WALKER_BEHAVIOR_GATE,
    }


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
        base.setdefault("training", {}).update(WALKER_STUDY_TRAINING)
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
    aggregate["screening_ready"] = (
        len(seeds) >= 1
        and step_budget >= 100_000
        and len(aggregate["rankings"]) == len(WALKER_VARIANTS)
        and all(row["seeds_completed"] == len(seeds) for row in aggregate["rankings"])
    )
    aggregate["balance_ready"] = bool(
        aggregate["screening_ready"] and aggregate["balance_candidate"] is not None
    )
    aggregate["evidence_ready"] = (
        len(seeds) >= 3
        and step_budget >= 300_000
        and len(aggregate["rankings"]) == len(WALKER_VARIANTS)
        and all(row["seeds_completed"] == len(seeds) for row in aggregate["rankings"])
    )
    aggregate["promotion_ready"] = bool(
        aggregate["evidence_ready"] and aggregate["promoted_variant"] is not None
    )
    aggregate["runs"] = diagnostics
    return aggregate


def _walker_velocity_passed(result: dict[str, Any]) -> bool:
    """Return whether a low-velocity continuation clears every behavior bound."""
    metrics = result[WALKER_VELOCITY_GATE["evaluation_mode"]]
    return bool(
        metrics["episode_length_mean"] >= WALKER_VELOCITY_GATE["min_episode_length"]
        and metrics["forward_displacement_mean"]
        >= WALKER_VELOCITY_GATE["min_forward_displacement"]
        and metrics["fall_rate"] <= WALKER_VELOCITY_GATE["max_fall_rate"]
        and metrics["peak_z_mean"] < WALKER_VELOCITY_GATE["max_peak_z"]
    )


def _aggregate_walker_velocity_results(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_variant: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for run_key, result in results.items():
        variant, seed_label = run_key.split("/", 1)
        by_variant.setdefault(variant, []).append((seed_label, result))

    rankings: list[dict[str, Any]] = []
    for variant, seed_results in by_variant.items():
        variant_results = [result for _, result in seed_results]
        stochastic = [result["stochastic"] for result in variant_results]
        per_seed_passed = {
            seed_label: _walker_velocity_passed(result)
            for seed_label, result in seed_results
        }
        row: dict[str, Any] = {
            "variant": variant,
            "seeds_completed": len(variant_results),
            "episode_length_mean": float(
                np.mean([metrics["episode_length_mean"] for metrics in stochastic])
            ),
            "forward_displacement_mean": float(
                np.mean(
                    [metrics["forward_displacement_mean"] for metrics in stochastic]
                )
            ),
            "fall_rate_mean": float(
                np.mean([metrics["fall_rate"] for metrics in stochastic])
            ),
            "peak_z_mean": float(
                np.mean([metrics["peak_z_mean"] for metrics in stochastic])
            ),
            "per_seed_passed": per_seed_passed,
            "gate_passed": bool(per_seed_passed and all(per_seed_passed.values())),
        }
        row["score"] = float(
            row["forward_displacement_mean"]
            + row["episode_length_mean"] / 800.0
            - row["fall_rate_mean"]
        )
        rankings.append(row)
    rankings.sort(
        key=lambda row: (
            cast(bool, row["gate_passed"]),
            cast(float, row["score"]),
        ),
        reverse=True,
    )
    promoted = next(
        (row["variant"] for row in rankings if row["gate_passed"]),
        None,
    )
    return {
        "rankings": rankings,
        "recommended_variant": rankings[0]["variant"] if rankings else None,
        "promoted_variant": promoted,
        "behavioral_gate": WALKER_VELOCITY_GATE,
        "runs": results,
    }


def _run_walker_velocity_study(
    *,
    seeds: list[int],
    step_budget: int,
    eval_episodes: int,
    config_dir: Path,
    source_dir: Path,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    diagnostics: dict[str, dict[str, Any]] = {}
    base = _load_cfg("walker_low_velocity_candidate", config_dir)
    for variant, overrides in WALKER_VELOCITY_VARIANTS.items():
        variant_cfg = deepcopy(base)
        _apply_overrides(variant_cfg, overrides)
        for seed in seeds:
            key = f"{variant}/seed_{seed}"
            source_model = (
                source_dir / f"seed_{seed}" / "checkpoints" / "final_model.zip"
            )
            if not source_model.is_file():
                raise FileNotFoundError(
                    f"Walker velocity source checkpoint not found: {source_model}"
                )
            cfg = _prepare_cfg(
                variant_cfg,
                experiment_name=f"quality_walker_{variant}",
                seed=seed,
                step_budget=step_budget,
                output_dir=output_dir,
            )
            run = _train_run(
                f"walker_velocity/{key}",
                cfg,
                state,
                state_path,
                resume_from=source_model,
            )
            if run.get("status") != "completed":
                continue
            if "diagnostics" not in run:
                run["diagnostics"] = evaluate_walker_checkpoint(
                    cfg,
                    run["model_path"],
                    episodes=eval_episodes,
                )
                _atomic_json_write(state_path, state)
            diagnostics[key] = run["diagnostics"]

    aggregate = _aggregate_walker_velocity_results(diagnostics)
    aggregate["evidence_ready"] = (
        len(seeds) >= 3
        and step_budget >= 100_000
        and len(aggregate["rankings"]) == len(WALKER_VELOCITY_VARIANTS)
        and all(row["seeds_completed"] == len(seeds) for row in aggregate["rankings"])
    )
    aggregate["promotion_ready"] = bool(
        aggregate["evidence_ready"] and aggregate["promoted_variant"] is not None
    )
    return aggregate


def _walker_gait_passed(result: dict[str, Any]) -> bool:
    """Return whether a continuation clears behavior and gait structure gates."""
    metrics = result[WALKER_GAIT_GATE["evaluation_mode"]]
    return bool(
        _walker_velocity_passed(result)
        # Cadence is a band: too slow is not stepping, too fast is chatter.
        and metrics["gait_alternating_touchdowns_per_100_steps_mean"]
        >= WALKER_GAIT_GATE["min_alternating_touchdowns_per_100_steps"]
        and metrics["gait_alternating_touchdowns_per_100_steps_mean"]
        <= WALKER_GAIT_GATE["max_alternating_touchdowns_per_100_steps"]
        # Stride and clearance are what separate a step from a contact flicker.
        and metrics["gait_stride_length_mean"] >= WALKER_GAIT_GATE["min_stride_length"]
        and metrics["gait_foot_clearance_mean"]
        >= WALKER_GAIT_GATE["min_foot_clearance"]
        and metrics["gait_progress_per_alternating_touchdown_mean"]
        >= WALKER_GAIT_GATE["min_progress_per_alternating_touchdown"]
        and metrics[WALKER_GAIT_GATE["slip_metric"]]
        <= WALKER_GAIT_GATE["max_slip_speed"]
        and metrics["gait_double_support_fraction_mean"]
        >= WALKER_GAIT_GATE["min_double_support_fraction"]
        and metrics["gait_flight_fraction_mean"]
        <= WALKER_GAIT_GATE["max_flight_fraction"]
        and metrics["gait_longest_same_foot_sequence_mean"]
        <= WALKER_GAIT_GATE["max_longest_same_foot_sequence"]
    )


def _aggregate_walker_gait_results(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_variant: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for run_key, result in results.items():
        variant, seed_label = run_key.split("/", 1)
        by_variant.setdefault(variant, []).append((seed_label, result))

    rankings: list[dict[str, Any]] = []
    for variant, seed_results in by_variant.items():
        stochastic = [result["stochastic"] for _, result in seed_results]
        per_seed_passed = {
            seed_label: _walker_gait_passed(result)
            for seed_label, result in seed_results
        }

        def mean(key: str) -> float:
            return float(np.mean([metrics[key] for metrics in stochastic]))

        row: dict[str, Any] = {
            "variant": variant,
            "seeds_completed": len(seed_results),
            "episode_length_mean": mean("episode_length_mean"),
            "forward_displacement_mean": mean("forward_displacement_mean"),
            "fall_rate_mean": mean("fall_rate"),
            "peak_z_mean": mean("peak_z_mean"),
            "alternating_touchdowns_per_100_steps_mean": mean(
                "gait_alternating_touchdowns_per_100_steps_mean"
            ),
            "progress_per_alternating_touchdown_mean": mean(
                "gait_progress_per_alternating_touchdown_mean"
            ),
            "slip_speed_mean": mean(WALKER_GAIT_GATE["slip_metric"]),
            "double_support_fraction_mean": mean(
                "gait_double_support_fraction_mean"
            ),
            "flight_fraction_mean": mean("gait_flight_fraction_mean"),
            "stride_length_mean": mean("gait_stride_length_mean"),
            "foot_clearance_mean": mean("gait_foot_clearance_mean"),
            "longest_same_foot_sequence_mean": mean(
                "gait_longest_same_foot_sequence_mean"
            ),
            "per_seed_passed": per_seed_passed,
            "gate_passed": bool(per_seed_passed and all(per_seed_passed.values())),
        }
        # Raw cadence is deliberately absent from the score: it is banded, not
        # maximised, and rewarding it is how the chattering shuffle came to
        # out-rank longer stepping. Stride length and clearance are monotonic —
        # more of either is unambiguously more like walking.
        row["score"] = float(
            row["forward_displacement_mean"]
            + row["episode_length_mean"] / 800.0
            - row["fall_rate_mean"]
            + row["stride_length_mean"]
            + row["foot_clearance_mean"]
            + 10.0 * row["progress_per_alternating_touchdown_mean"]
            - row["slip_speed_mean"]
        )
        rankings.append(row)
    rankings.sort(
        key=lambda row: (
            cast(bool, row["gate_passed"]),
            cast(float, row["score"]),
        ),
        reverse=True,
    )
    promoted = next(
        (row["variant"] for row in rankings if row["gate_passed"]),
        None,
    )
    return {
        "rankings": rankings,
        "recommended_variant": rankings[0]["variant"] if rankings else None,
        "promoted_variant": promoted,
        "behavioral_and_gait_gate": WALKER_GAIT_GATE,
        "runs": results,
    }


def _run_walker_gait_study(
    *,
    seeds: list[int],
    step_budget: int,
    eval_episodes: int,
    config_dir: Path,
    source_dir: Path,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    diagnostics: dict[str, dict[str, Any]] = {}
    base = _load_cfg("walker_low_velocity_candidate", config_dir)
    _apply_overrides(
        base,
        {
            "environment.reward.velocity_sigma": 0.10,
            "environment.reward.gait_step_progress_weight": 0.0,
            "environment.reward.stance_slip_penalty_weight": 0.0,
        },
    )
    for variant, overrides in WALKER_GAIT_VARIANTS.items():
        variant_cfg = deepcopy(base)
        _apply_overrides(variant_cfg, overrides)
        for seed in seeds:
            key = f"{variant}/seed_{seed}"
            source_model = (
                source_dir / f"seed_{seed}" / "checkpoints" / "final_model.zip"
            )
            if not source_model.is_file():
                raise FileNotFoundError(
                    f"Walker gait source checkpoint not found: {source_model}"
                )
            cfg = _prepare_cfg(
                variant_cfg,
                experiment_name=f"quality_walker_gait_{variant}",
                seed=seed,
                step_budget=step_budget,
                output_dir=output_dir,
            )
            run = _train_run(
                f"walker_gait/{key}",
                cfg,
                state,
                state_path,
                resume_from=source_model,
            )
            if run.get("status") != "completed":
                continue
            if "diagnostics" not in run:
                run["diagnostics"] = evaluate_walker_checkpoint(
                    cfg,
                    run["model_path"],
                    episodes=eval_episodes,
                )
                _atomic_json_write(state_path, state)
            diagnostics[key] = run["diagnostics"]

    aggregate = _aggregate_walker_gait_results(diagnostics)
    aggregate["evidence_ready"] = (
        len(seeds) >= 3
        and step_budget >= 150_000
        and len(aggregate["rankings"]) == len(WALKER_GAIT_VARIANTS)
        and all(row["seeds_completed"] == len(seeds) for row in aggregate["rankings"])
    )
    aggregate["promotion_ready"] = bool(
        aggregate["evidence_ready"] and aggregate["promoted_variant"] is not None
    )
    return aggregate


def _aggregate_walker_velocity_ramp_results(
    results: dict[str, dict[str, Any]],
    *,
    candidate_variant: str = "ramp_to_100",
    paired_gate: dict[str, Any] = WALKER_VELOCITY_RAMP_GATE,
    gate_label: str = "velocity_ramp_gate",
) -> dict[str, Any]:
    """Apply frozen gait gates and compare a paired candidate to control."""
    by_variant: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for run_key, result in results.items():
        variant, seed_label = run_key.split("/", 1)
        by_variant.setdefault(variant, []).append((seed_label, result))

    rows: dict[str, dict[str, Any]] = {}
    for variant, seed_results in by_variant.items():
        stochastic = [result["stochastic"] for _, result in seed_results]
        per_seed_passed = {
            seed_label: _walker_gait_passed(result)
            for seed_label, result in seed_results
        }
        rows[variant] = {
            "variant": variant,
            "seeds_completed": len(seed_results),
            "forward_displacement_mean": float(
                np.mean(
                    [metrics["forward_displacement_mean"] for metrics in stochastic]
                )
            ),
            "episode_length_mean": float(
                np.mean([metrics["episode_length_mean"] for metrics in stochastic])
            ),
            "fall_rate_mean": float(
                np.mean([metrics["fall_rate"] for metrics in stochastic])
            ),
            "peak_z_mean": float(
                np.mean([metrics["peak_z_mean"] for metrics in stochastic])
            ),
            "slip_speed_mean": float(
                np.mean(
                    [
                        metrics[WALKER_GAIT_GATE["slip_metric"]]
                        for metrics in stochastic
                    ]
                )
            ),
            "per_seed_passed": per_seed_passed,
            "frozen_gate_passed": bool(
                per_seed_passed and all(per_seed_passed.values())
            ),
        }

    control = rows.get("control")
    candidate = rows.get(candidate_variant)
    if control is not None and candidate is not None:
        improvement = (
            candidate["forward_displacement_mean"]
            - control["forward_displacement_mean"]
        )
        candidate["displacement_improvement_over_control"] = improvement
        candidate["material_displacement_improvement"] = bool(
            improvement >= paired_gate["min_displacement_improvement"]
        )
        candidate["gate_passed"] = bool(
            candidate["frozen_gate_passed"]
            and candidate["material_displacement_improvement"]
        )
    for variant, row in rows.items():
        if variant != candidate_variant:
            row["gate_passed"] = False
    rankings = sorted(
        rows.values(), key=lambda row: row["forward_displacement_mean"], reverse=True
    )
    return {
        "rankings": rankings,
        "promoted_variant": candidate_variant
        if candidate and candidate["gate_passed"]
        else None,
        "frozen_behavioral_and_gait_gate": WALKER_GAIT_GATE,
        gate_label: paired_gate,
        "runs": results,
    }


def _run_walker_velocity_ramp_study(
    *,
    seeds: list[int],
    step_budget: int,
    eval_episodes: int,
    config_dir: Path,
    source_dir: Path,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    diagnostics: dict[str, dict[str, Any]] = {}
    base = _load_cfg("walker_low_velocity_candidate", config_dir)
    _apply_overrides(
        base,
        {
            "environment.reward.velocity_sigma": 0.10,
            "environment.reward.gait_step_progress_weight": 0.0,
            "environment.reward.stance_slip_penalty_weight": 0.5,
        },
    )
    for variant, values in WALKER_VELOCITY_RAMP_VARIANTS.items():
        for seed in seeds:
            key = f"{variant}/seed_{seed}"
            source_model = (
                source_dir / f"seed_{seed}" / "checkpoints" / "final_model.zip"
            )
            if not source_model.is_file():
                raise FileNotFoundError(
                    f"Walker velocity-ramp source checkpoint not found: {source_model}"
                )
            target_end = float(values.get("target_velocity_end", 0.15))
            cfg = _prepare_cfg(
                base,
                experiment_name=f"quality_walker_velocity_ramp_{variant}",
                seed=seed,
                step_budget=step_budget,
                output_dir=output_dir,
            )
            callbacks = (
                [
                    VelocityTargetRampCallback(
                        start=0.15,
                        end=target_end,
                        ramp_steps=WALKER_VELOCITY_RAMP_STEPS,
                    )
                ]
                if target_end > 0.15
                else None
            )
            run = _train_run(
                f"walker_velocity_ramp/{key}",
                cfg,
                state,
                state_path,
                resume_from=source_model,
                extra_callbacks=callbacks,
            )
            if run.get("status") != "completed":
                continue
            if "diagnostics" not in run:
                evaluation_cfg = deepcopy(cfg)
                evaluation_cfg["environment"]["reward"]["target_velocity"] = target_end
                run["diagnostics"] = evaluate_walker_checkpoint(
                    evaluation_cfg, run["model_path"], episodes=eval_episodes
                )
                _atomic_json_write(state_path, state)
            diagnostics[key] = run["diagnostics"]

    aggregate = _aggregate_walker_velocity_ramp_results(diagnostics)
    aggregate["screening_ready"] = (
        step_budget >= WALKER_VELOCITY_RAMP_STEPS
        and len(seeds) >= 1
        and len(aggregate["rankings"]) == len(WALKER_VELOCITY_RAMP_VARIANTS)
        and all(row["seeds_completed"] == len(seeds) for row in aggregate["rankings"])
    )
    aggregate["evidence_ready"] = (
        step_budget >= WALKER_VELOCITY_RAMP_STEPS
        and len(seeds) >= 3
        and aggregate["screening_ready"]
    )
    aggregate["promotion_ready"] = bool(
        aggregate["evidence_ready"] and aggregate["promoted_variant"] is not None
    )
    return aggregate


def _run_walker_swing_touchdown_study(
    *,
    seeds: list[int],
    step_budget: int,
    eval_episodes: int,
    config_dir: Path,
    source_dir: Path,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    """Screen calibrated sustained-swing progress against a paired control."""
    diagnostics: dict[str, dict[str, Any]] = {}
    base = _load_cfg("walker_low_velocity_candidate", config_dir)
    _apply_overrides(
        base,
        {
            "environment.reward.velocity_sigma": 0.10,
            "environment.reward.gait_step_progress_weight": 0.0,
            "environment.reward.stance_slip_penalty_weight": 0.5,
        },
    )
    for variant, overrides in WALKER_SWING_TOUCHDOWN_VARIANTS.items():
        variant_cfg = deepcopy(base)
        _apply_overrides(variant_cfg, overrides)
        for seed in seeds:
            key = f"{variant}/seed_{seed}"
            source_model = (
                source_dir / f"seed_{seed}" / "checkpoints" / "final_model.zip"
            )
            if not source_model.is_file():
                raise FileNotFoundError(
                    f"Walker swing-touchdown source checkpoint not found: {source_model}"
                )
            cfg = _prepare_cfg(
                variant_cfg,
                experiment_name=f"quality_walker_swing_touchdown_{variant}",
                seed=seed,
                step_budget=step_budget,
                output_dir=output_dir,
            )
            run = _train_run(
                f"walker_swing_touchdown/{key}",
                cfg,
                state,
                state_path,
                resume_from=source_model,
            )
            if run.get("status") != "completed":
                continue
            if "diagnostics" not in run:
                run["diagnostics"] = evaluate_walker_checkpoint(
                    cfg, run["model_path"], episodes=eval_episodes
                )
                _atomic_json_write(state_path, state)
            diagnostics[key] = run["diagnostics"]

    aggregate = _aggregate_walker_velocity_ramp_results(
        diagnostics,
        candidate_variant="swing_qualified_progress",
        paired_gate=WALKER_SWING_TOUCHDOWN_GATE,
        gate_label="swing_touchdown_gate",
    )
    aggregate["screening_ready"] = (
        step_budget >= 50_000
        and len(seeds) >= 1
        and len(aggregate["rankings"]) == len(WALKER_SWING_TOUCHDOWN_VARIANTS)
        and all(row["seeds_completed"] == len(seeds) for row in aggregate["rankings"])
    )
    aggregate["evidence_ready"] = False
    aggregate["promotion_ready"] = False
    return aggregate


def _walker_slip_recovery_gate_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    metrics = result[WALKER_SLIP_RECOVERY_GATE["evaluation_mode"]]
    flags = {
        "launch": metrics["peak_z_mean"]
        >= WALKER_SLIP_RECOVERY_GATE["max_peak_z"],
        "sliding": (
            metrics["forward_displacement_mean"]
            >= WALKER_SLIP_RECOVERY_GATE["min_forward_displacement"]
            and metrics[WALKER_SLIP_RECOVERY_GATE["slip_metric"]]
            > WALKER_SLIP_RECOVERY_GATE["max_slip_speed"]
        ),
        "contact_chatter": (
            metrics["gait_alternating_touchdowns_per_100_steps_mean"]
            > WALKER_GAIT_GATE["max_alternating_touchdowns_per_100_steps"]
        ),
    }
    criteria = {
        "fall_rate": metrics["fall_rate"]
        <= WALKER_SLIP_RECOVERY_GATE["max_fall_rate"],
        "slip_speed": metrics[WALKER_SLIP_RECOVERY_GATE["slip_metric"]]
        <= WALKER_SLIP_RECOVERY_GATE["max_slip_speed"],
        "forward_displacement": metrics["forward_displacement_mean"]
        >= WALKER_SLIP_RECOVERY_GATE["min_forward_displacement"],
        "peak_z": metrics["peak_z_mean"]
        < WALKER_SLIP_RECOVERY_GATE["max_peak_z"],
        "automatic_exploit_screen": not any(flags.values()),
    }
    return {
        "criteria": criteria,
        "automatic_exploit_flags": flags,
        "metrics_gate_passed": all(criteria.values()),
        "slip_gate_passed": criteria["slip_speed"],
        "visual_review_required": True,
    }


def _aggregate_walker_slip_recovery_results(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for variant, result in results.items():
        metrics = result["stochastic"]
        gate = _walker_slip_recovery_gate_result(result)
        rows.append(
            {
                "variant": variant,
                "episode_length_mean": metrics["episode_length_mean"],
                "fall_rate": metrics["fall_rate"],
                "slip_speed": metrics[WALKER_SLIP_RECOVERY_GATE["slip_metric"]],
                "forward_displacement": metrics["forward_displacement_mean"],
                "peak_z": metrics["peak_z_mean"],
                "stride_length": metrics["gait_stride_length_mean"],
                "alternating_touchdowns_per_100_steps": metrics[
                    "gait_alternating_touchdowns_per_100_steps_mean"
                ],
                **gate,
                "score": float(
                    metrics["forward_displacement_mean"]
                    - metrics["fall_rate"]
                    - metrics[WALKER_SLIP_RECOVERY_GATE["slip_metric"]]
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            cast(bool, row["metrics_gate_passed"]),
            cast(float, row["score"]),
        ),
        reverse=True,
    )
    passing_candidates = [
        row["variant"]
        for row in rows
        if row["variant"] != "control" and row["metrics_gate_passed"]
    ]
    return {
        "rankings": rows,
        "objective_winner": (
            passing_candidates[0] if len(passing_candidates) == 1 else None
        ),
        "passing_candidates": passing_candidates,
        "advancement_gate": WALKER_SLIP_RECOVERY_GATE,
    }


def _phase0_slip_recovery_telemetry(
    *,
    seeds: list[int],
    config_dir: Path,
    source_dir: Path,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    analyses = state.setdefault("analyses", {})
    reports: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        key = f"walker_slip_recovery/phase0/seed_{seed}"
        source_model = source_dir / f"seed_{seed}" / "checkpoints" / "final_model.zip"
        if not source_model.is_file():
            raise FileNotFoundError(
                f"Walker slip-recovery source checkpoint not found: {source_model}"
            )
        report = analyses.get(key)
        if report is None:
            cfg = _load_cfg("walker_low_velocity_candidate", config_dir)
            cfg["seed"] = seed
            cfg["environment"]["seed"] = seed
            report = collect_walker_fall_recovery_telemetry(
                cfg,
                source_model,
                episodes=5,
                render_dir=output_dir / "videos" / "phase0" / f"seed_{seed}",
            )
            analyses[key] = report
            _atomic_json_write(state_path, state)
        reports[f"seed_{seed}"] = report

    contact_speeds = [
        value
        for report in reports.values()
        for value in report["calibration_samples"]["contact_slip_speeds"]
    ]
    touchdown_forces = [
        value
        for report in reports.values()
        for value in report["calibration_samples"]["touchdown_normal_forces"]
    ]
    thresholds = calibrate_walker_slip_thresholds(contact_speeds, touchdown_forces)
    cause_counts: Counter[str] = Counter()
    pooled_reports = {}
    for seed_label, report in reports.items():
        pooled = summarize_walker_fall_windows(
            [
                {
                    "episode": fall["episode"],
                    "termination_reason": fall["termination_reason"],
                    "window": fall["window"],
                }
                for fall in report["falls"]
            ],
            control_timestep=float(report["control_timestep"]),
            thresholds=thresholds,
        )
        cause_counts.update(pooled["cause_counts"])
        pooled_reports[seed_label] = {
            **report,
            "thresholds": thresholds,
            **pooled,
        }
    fall_count = sum(cause_counts.values())
    dominant = (
        cause_counts.most_common(1)[0][0] if cause_counts else "no_observed_falls"
    )
    return {
        "thresholds": thresholds,
        "fall_count": fall_count,
        "cause_counts": dict(sorted(cause_counts.items())),
        "dominant_mechanism": dominant,
        "contact_damping_enabled": (
            fall_count > 0
            and cause_counts["touchdown_overshoot"]
            == max(cause_counts.values())
        ),
        "seed_reports": pooled_reports,
    }


def _slip_recovery_base_cfg(
    config_dir: Path,
    slip_clip: float,
) -> dict[str, Any]:
    cfg = _load_cfg("walker_low_velocity_candidate", config_dir)
    cfg["environment"]["reward"]["target_velocity"] = 1.0
    cfg["environment"]["reward"]["stance_slip_penalty_weight"] = 0.0
    cfg["environment"]["reward"]["stance_slip_penalty_clip"] = slip_clip
    # The source checkpoints have already completed the 0.15→1.0 ramp.
    # Re-running that callback on resume would silently change the fixed
    # continuation back to a low-speed target.
    cfg["training"].pop("velocity_target_ramp", None)
    return cfg


def _run_slip_recovery_candidate(
    *,
    variant: str,
    overrides: dict[str, Any],
    base: dict[str, Any],
    seed: int,
    step_budget: int,
    eval_episodes: int,
    source_model: Path,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
    phase: str,
) -> dict[str, Any] | None:
    cfg = deepcopy(base)
    _apply_overrides(cfg, overrides)
    cfg = _prepare_cfg(
        cfg,
        experiment_name=f"quality_walker_slip_recovery_{phase}_{variant}",
        seed=seed,
        step_budget=step_budget,
        output_dir=output_dir,
    )
    run = _train_run(
        f"walker_slip_recovery/{phase}/{variant}/seed_{seed}",
        cfg,
        state,
        state_path,
        resume_from=source_model,
    )
    if run.get("status") != "completed":
        return None
    if "diagnostics" not in run:
        run["diagnostics"] = evaluate_walker_checkpoint(
            cfg,
            run["model_path"],
            episodes=eval_episodes,
        )
        _atomic_json_write(state_path, state)
    if "fall_telemetry" not in run:
        run["fall_telemetry"] = collect_walker_fall_recovery_telemetry(
            cfg,
            run["model_path"],
            episodes=5,
            render_dir=output_dir / "videos" / phase / variant / f"seed_{seed}",
        )
        _atomic_json_write(state_path, state)
    return run


def _run_walker_slip_recovery_study(
    *,
    seeds: list[int],
    step_budget: int,
    eval_episodes: int,
    config_dir: Path,
    source_dir: Path,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
    approved_variant: str | None,
) -> dict[str, Any]:
    phase0 = _phase0_slip_recovery_telemetry(
        seeds=seeds,
        config_dir=config_dir,
        source_dir=source_dir,
        output_dir=output_dir,
        state=state,
        state_path=state_path,
    )
    base = _slip_recovery_base_cfg(
        config_dir,
        float(phase0["thresholds"]["stance_slip_penalty_clip"]),
    )
    phase1_seed = 21
    if phase1_seed not in seeds:
        raise ValueError("walker-slip-recovery Phase 1 requires seed 21")
    source_model = (
        source_dir / f"seed_{phase1_seed}" / "checkpoints" / "final_model.zip"
    )
    variants = dict(WALKER_SLIP_RECOVERY_VARIANTS)
    if phase0["contact_damping_enabled"]:
        variants.update(WALKER_CONTACT_DAMPING_VARIANTS)

    phase1_results: dict[str, dict[str, Any]] = {}
    variant_overrides = dict(variants)
    for variant, overrides in variants.items():
        run = _run_slip_recovery_candidate(
            variant=variant,
            overrides=overrides,
            base=base,
            seed=phase1_seed,
            step_budget=step_budget,
            eval_episodes=eval_episodes,
            source_model=source_model,
            output_dir=output_dir,
            state=state,
            state_path=state_path,
            phase="phase1",
        )
        if run is not None:
            phase1_results[variant] = run["diagnostics"]

    initial_aggregate = _aggregate_walker_slip_recovery_results(phase1_results)
    slip_holding = [
        row
        for row in initial_aggregate["rankings"]
        if row["variant"] != "control" and row["slip_gate_passed"]
    ]
    if slip_holding:
        preceding = slip_holding[0]
        preceding_variant = cast(str, preceding["variant"])
        preceding_run = state["runs"][
            f"walker_slip_recovery/phase1/{preceding_variant}/seed_{phase1_seed}"
        ]
        reset_cfg = base["environment"].get("reset_randomization", {})
        recovery_variant = f"{preceding_variant}_recovery_noise_2x"
        recovery_overrides = {
            **variant_overrides[preceding_variant],
            "environment.reset_randomization.position_xy_noise": (
                float(reset_cfg.get("position_xy_noise", 0.02))
                * WALKER_RECOVERY_NOISE_MULTIPLIER
            ),
            "environment.reset_randomization.yaw_noise": (
                float(reset_cfg.get("yaw_noise", 0.08))
                * WALKER_RECOVERY_NOISE_MULTIPLIER
            ),
        }
        variant_overrides[recovery_variant] = recovery_overrides
        run = _run_slip_recovery_candidate(
            variant=recovery_variant,
            overrides=recovery_overrides,
            base=base,
            seed=phase1_seed,
            step_budget=step_budget,
            eval_episodes=eval_episodes,
            source_model=Path(preceding_run["model_path"]),
            output_dir=output_dir,
            state=state,
            state_path=state_path,
            phase="phase1_recovery",
        )
        if run is not None:
            phase1_results[recovery_variant] = run["diagnostics"]

    phase1 = _aggregate_walker_slip_recovery_results(phase1_results)
    objective_winner = phase1["objective_winner"]
    phase1["visual_approval"] = {
        "required": True,
        "approved_variant": approved_variant,
        "accepted": bool(
            objective_winner is not None and approved_variant == objective_winner
        ),
    }
    phase2: dict[str, Any] = {
        "status": "not_ready",
        "reason": (
            "Phase 1 did not produce exactly one objective winner."
            if objective_winner is None
            else "Review the five stochastic renders and approve the objective winner."
        ),
    }
    if objective_winner is not None and approved_variant == objective_winner:
        confirmation: dict[str, dict[str, Any]] = {}
        winner_overrides = variant_overrides[objective_winner]
        for seed in seeds:
            seed_source = source_dir / f"seed_{seed}" / "checkpoints" / "final_model.zip"
            run = _run_slip_recovery_candidate(
                variant=objective_winner,
                overrides=winner_overrides,
                base=base,
                seed=seed,
                step_budget=3_000_000,
                eval_episodes=eval_episodes,
                source_model=seed_source,
                output_dir=output_dir,
                state=state,
                state_path=state_path,
                phase="phase2_confirmation",
            )
            if run is not None:
                confirmation[f"seed_{seed}"] = run["diagnostics"]
        per_seed_gate = {
            seed_label: _walker_slip_recovery_gate_result(result)[
                "metrics_gate_passed"
            ]
            for seed_label, result in confirmation.items()
        }
        phase2 = {
            "status": "confirmation_failed",
            "winner": objective_winner,
            "per_seed_gate": per_seed_gate,
            "confirmation": confirmation,
        }
        if len(confirmation) == len(seeds) and all(per_seed_gate.values()):
            transfer = {}
            for seed in seeds:
                run = state["runs"][
                    "walker_slip_recovery/phase2_confirmation/"
                    f"{objective_winner}/seed_{seed}"
                ]
                transfer[f"seed_{seed}"] = evaluate_walker_transfer_suite(
                    run["config"],
                    run["model_path"],
                    episodes=eval_episodes,
                )
            phase2.update(
                {
                    "status": "transfer_completed",
                    "transfer": transfer,
                }
            )
    return {
        "phase0": phase0,
        "phase1": phase1,
        "phase2": phase2,
        "screening_ready": step_budget >= 1_000_000 and eval_episodes >= 20,
        "confirmation_budget_steps": 3_000_000,
    }


def _walker_action_memory_gate_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen natural-speed, slip, and real-gait gates."""
    metrics = result[WALKER_SLIP_RECOVERY_GATE["evaluation_mode"]]
    criteria = {
        "episode_length": metrics["episode_length_mean"]
        >= WALKER_GAIT_GATE["min_episode_length"],
        "fall_rate": metrics["fall_rate"]
        <= WALKER_SLIP_RECOVERY_GATE["max_fall_rate"],
        "slip_speed": metrics[WALKER_SLIP_RECOVERY_GATE["slip_metric"]]
        <= WALKER_SLIP_RECOVERY_GATE["max_slip_speed"],
        "forward_displacement": metrics["forward_displacement_mean"]
        >= WALKER_SLIP_RECOVERY_GATE["min_forward_displacement"],
        "peak_z": metrics["peak_z_mean"]
        < WALKER_SLIP_RECOVERY_GATE["max_peak_z"],
        "cadence_band": (
            WALKER_GAIT_GATE["min_alternating_touchdowns_per_100_steps"]
            <= metrics["gait_alternating_touchdowns_per_100_steps_mean"]
            <= WALKER_GAIT_GATE["max_alternating_touchdowns_per_100_steps"]
        ),
        "progress_per_touchdown": metrics[
            "gait_progress_per_alternating_touchdown_mean"
        ]
        >= WALKER_GAIT_GATE["min_progress_per_alternating_touchdown"],
        "stride_length": metrics["gait_stride_length_mean"]
        >= WALKER_GAIT_GATE["min_stride_length"],
        "foot_clearance": metrics["gait_foot_clearance_mean"]
        >= WALKER_GAIT_GATE["min_foot_clearance"],
        "double_support_fraction": metrics["gait_double_support_fraction_mean"]
        >= WALKER_GAIT_GATE["min_double_support_fraction"],
        "flight_fraction": metrics["gait_flight_fraction_mean"]
        <= WALKER_GAIT_GATE["max_flight_fraction"],
        "same_foot_sequence": metrics["gait_longest_same_foot_sequence_mean"]
        <= WALKER_GAIT_GATE["max_longest_same_foot_sequence"],
    }
    flags = {
        "launch": not criteria["peak_z"],
        "sliding": (
            criteria["forward_displacement"] and not criteria["slip_speed"]
        ),
        "contact_chatter": (
            metrics["gait_alternating_touchdowns_per_100_steps_mean"]
            > WALKER_GAIT_GATE["max_alternating_touchdowns_per_100_steps"]
        ),
    }
    return {
        "criteria": criteria,
        "automatic_exploit_flags": flags,
        "metrics_gate_passed": all(criteria.values()) and not any(flags.values()),
        "visual_review_required": True,
    }


def _aggregate_walker_action_memory_results(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant, result in results.items():
        metrics = result["stochastic"]
        gate = _walker_action_memory_gate_result(result)
        rows.append(
            {
                "variant": variant,
                "episode_length_mean": metrics["episode_length_mean"],
                "fall_rate": metrics["fall_rate"],
                "slip_speed": metrics[WALKER_SLIP_RECOVERY_GATE["slip_metric"]],
                "double_support_fraction": metrics[
                    "gait_double_support_fraction_mean"
                ],
                "forward_displacement": metrics["forward_displacement_mean"],
                "peak_z": metrics["peak_z_mean"],
                "alternating_touchdowns_per_100_steps": metrics[
                    "gait_alternating_touchdowns_per_100_steps_mean"
                ],
                "progress_per_alternating_touchdown": metrics[
                    "gait_progress_per_alternating_touchdown_mean"
                ],
                "stride_length": metrics["gait_stride_length_mean"],
                "foot_clearance": metrics["gait_foot_clearance_mean"],
                **gate,
                "score": float(
                    metrics["forward_displacement_mean"]
                    - metrics["fall_rate"]
                    - metrics[WALKER_SLIP_RECOVERY_GATE["slip_metric"]]
                    + metrics["gait_stride_length_mean"]
                    + metrics["gait_foot_clearance_mean"]
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            cast(bool, row["metrics_gate_passed"]),
            cast(float, row["score"]),
        ),
        reverse=True,
    )
    candidate = next(
        (
            row
            for row in rows
            if row["variant"] == "previous_action"
            and row["metrics_gate_passed"]
        ),
        None,
    )
    return {
        "rankings": rows,
        "objective_winner": "previous_action" if candidate is not None else None,
        "advancement_gate": {
            **WALKER_SLIP_RECOVERY_GATE,
            **{
                key: value
                for key, value in WALKER_GAIT_GATE.items()
                if key not in WALKER_SLIP_RECOVERY_GATE
            },
        },
    }


def _action_memory_base_cfg(config_dir: Path) -> dict[str, Any]:
    cfg = _load_cfg("walker_low_velocity_candidate", config_dir)
    cfg["environment"]["reward"].update(
        {
            "stance_slip_penalty_weight": 0.5,
            "stance_slip_penalty_clip": WALKER_ACTION_MEMORY_SLIP_CLIP,
        }
    )
    return cfg


def _run_action_memory_candidate(
    *,
    variant: str,
    overrides: dict[str, Any],
    base: dict[str, Any],
    seed: int,
    step_budget: int,
    eval_episodes: int,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
    phase: str,
) -> dict[str, Any] | None:
    cfg = deepcopy(base)
    _apply_overrides(cfg, overrides)
    cfg = _prepare_cfg(
        cfg,
        experiment_name=f"quality_walker_action_memory_{phase}_{variant}",
        seed=seed,
        step_budget=step_budget,
        output_dir=output_dir,
    )
    run = _train_run(
        f"walker_action_memory/{phase}/{variant}/seed_{seed}",
        cfg,
        state,
        state_path,
    )
    if run.get("status") != "completed":
        return None
    evaluation_cfg = deepcopy(cfg)
    evaluation_cfg["environment"]["reward"]["target_velocity"] = 1.0
    if "diagnostics" not in run:
        run["diagnostics"] = evaluate_walker_checkpoint(
            evaluation_cfg,
            run["model_path"],
            episodes=eval_episodes,
        )
        _atomic_json_write(state_path, state)
    if "fall_telemetry" not in run:
        run["fall_telemetry"] = collect_walker_fall_recovery_telemetry(
            evaluation_cfg,
            run["model_path"],
            episodes=5,
            render_dir=output_dir / "videos" / phase / variant / f"seed_{seed}",
        )
        _atomic_json_write(state_path, state)
    return run


def _run_walker_action_memory_study(
    *,
    seeds: list[int],
    step_budget: int,
    eval_episodes: int,
    config_dir: Path,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
    approved_variant: str | None,
) -> dict[str, Any]:
    """Train checkpoint-incompatible observation arms as matched new lineages."""
    phase1_seed = 21
    if phase1_seed not in seeds:
        raise ValueError("walker-action-memory Phase 1 requires seed 21")
    base = _action_memory_base_cfg(config_dir)
    phase1_results: dict[str, dict[str, Any]] = {}
    for variant, overrides in WALKER_ACTION_MEMORY_VARIANTS.items():
        run = _run_action_memory_candidate(
            variant=variant,
            overrides=overrides,
            base=base,
            seed=phase1_seed,
            step_budget=step_budget,
            eval_episodes=eval_episodes,
            output_dir=output_dir,
            state=state,
            state_path=state_path,
            phase="phase1",
        )
        if run is not None:
            phase1_results[variant] = run["diagnostics"]

    phase1 = _aggregate_walker_action_memory_results(phase1_results)
    screening_ready = (
        step_budget >= WALKER_ACTION_MEMORY_SCREEN_STEPS
        and eval_episodes >= 20
        and len(phase1_results) == len(WALKER_ACTION_MEMORY_VARIANTS)
    )
    objective_winner = phase1["objective_winner"]
    required_confirmation_seeds = {21, 22, 23}
    has_confirmation_seeds = required_confirmation_seeds.issubset(seeds)
    phase1["screening_ready"] = screening_ready
    phase1["visual_approval"] = {
        "required": True,
        "approved_variant": approved_variant,
        "accepted": bool(
            screening_ready
            and objective_winner is not None
            and approved_variant == objective_winner
        ),
    }

    phase2: dict[str, Any] = {
        "status": "not_ready",
        "reason": (
            "The matched new-lineage screen has not reached 10M steps per arm."
            if not screening_ready
            else (
                "Confirmation requires seeds 21, 22, and 23."
                if not has_confirmation_seeds
                else (
                    "Phase 1 did not pass every frozen locomotion and gait gate."
                    if objective_winner is None
                    else "Review the five stochastic renders and approve previous_action."
                )
            )
        ),
    }
    if (
        screening_ready
        and has_confirmation_seeds
        and objective_winner == "previous_action"
        and approved_variant == objective_winner
    ):
        confirmation: dict[str, dict[str, Any]] = {
            f"seed_{phase1_seed}": phase1_results[objective_winner]
        }
        for seed in seeds:
            if seed == phase1_seed:
                continue
            run = _run_action_memory_candidate(
                variant=objective_winner,
                overrides=WALKER_ACTION_MEMORY_VARIANTS[objective_winner],
                base=base,
                seed=seed,
                step_budget=WALKER_ACTION_MEMORY_SCREEN_STEPS,
                eval_episodes=eval_episodes,
                output_dir=output_dir,
                state=state,
                state_path=state_path,
                phase="phase2_confirmation",
            )
            if run is not None:
                confirmation[f"seed_{seed}"] = run["diagnostics"]
        per_seed_gate = {
            seed_label: _walker_action_memory_gate_result(result)[
                "metrics_gate_passed"
            ]
            for seed_label, result in confirmation.items()
        }
        phase2 = {
            "status": "confirmation_failed",
            "winner": objective_winner,
            "per_seed_gate": per_seed_gate,
            "confirmation": confirmation,
        }
        if len(confirmation) == len(seeds) and all(per_seed_gate.values()):
            transfer: dict[str, Any] = {}
            for seed in seeds:
                phase = (
                    "phase1" if seed == phase1_seed else "phase2_confirmation"
                )
                run = state["runs"][
                    f"walker_action_memory/{phase}/{objective_winner}/seed_{seed}"
                ]
                evaluation_cfg = deepcopy(run["config"])
                evaluation_cfg["environment"]["reward"]["target_velocity"] = 1.0
                transfer[f"seed_{seed}"] = evaluate_walker_transfer_suite(
                    evaluation_cfg,
                    run["model_path"],
                    episodes=eval_episodes,
                )
            phase2.update({"status": "transfer_completed", "transfer": transfer})
    return {
        "phase1": phase1,
        "phase2": phase2,
        "new_lineage_reason": (
            "Adding previous applied action changes the observation from 35 to "
            "45 values and is checkpoint-incompatible."
        ),
        "screen_budget_steps": WALKER_ACTION_MEMORY_SCREEN_STEPS,
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
        lines.extend(
            [
                f"## {study.title()}",
                "",
                "```json",
                json.dumps(payload, indent=2),
                "```",
                "",
            ]
        )
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
    source_dir: str | Path = "outputs/walker_stochastic_balance_candidate",
    approved_variant: str | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run walker, walker continuations, arena, algorithm, or all studies."""
    studies = ["walker", "arena", "algorithms"] if study == "all" else [study]
    invalid = set(studies) - {
        "walker",
        "walker-velocity",
        "walker-gait",
        "walker-velocity-ramp",
        "walker-swing-touchdown",
        "walker-slip-recovery",
        "walker-action-memory",
        "arena",
        "algorithms",
    }
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
    if {
        "walker-velocity",
        "walker-gait",
        "walker-velocity-ramp",
        "walker-swing-touchdown",
        "walker-slip-recovery",
    } & set(studies):
        identity["source_dir"] = str(Path(source_dir))
    plan = {
        "identity": identity,
        "planned_runs": {
            "walker": len(WALKER_VARIANTS) * len(seeds) if "walker" in studies else 0,
            "walker_velocity": (
                len(WALKER_VELOCITY_VARIANTS) * len(seeds)
                if "walker-velocity" in studies
                else 0
            ),
            "walker_gait": (
                len(WALKER_GAIT_VARIANTS) * len(seeds)
                if "walker-gait" in studies
                else 0
            ),
            "walker_velocity_ramp": (
                len(WALKER_VELOCITY_RAMP_VARIANTS) * len(seeds)
                if "walker-velocity-ramp" in studies
                else 0
            ),
            "walker_swing_touchdown": (
                len(WALKER_SWING_TOUCHDOWN_VARIANTS) * len(seeds)
                if "walker-swing-touchdown" in studies
                else 0
            ),
            "walker_slip_recovery": (
                len(WALKER_SLIP_RECOVERY_VARIANTS)
                if "walker-slip-recovery" in studies
                else 0
            ),
            "walker_action_memory": (
                len(WALKER_ACTION_MEMORY_VARIANTS)
                if "walker-action-memory" in studies
                else 0
            ),
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
    if "walker-velocity" in studies:
        state["results"]["walker_velocity"] = _run_walker_velocity_study(
            seeds=seeds,
            step_budget=step_budget or 150_000,
            eval_episodes=eval_episodes,
            config_dir=config_path,
            source_dir=Path(source_dir),
            output_dir=output,
            state=state,
            state_path=state_path,
        )
    if "walker-gait" in studies:
        state["results"]["walker_gait"] = _run_walker_gait_study(
            seeds=seeds,
            step_budget=step_budget or 150_000,
            eval_episodes=eval_episodes,
            config_dir=config_path,
            source_dir=Path(source_dir),
            output_dir=output,
            state=state,
            state_path=state_path,
        )
    if "walker-velocity-ramp" in studies:
        state["results"]["walker_velocity_ramp"] = _run_walker_velocity_ramp_study(
            seeds=seeds,
            step_budget=step_budget or 50_000,
            eval_episodes=eval_episodes,
            config_dir=config_path,
            source_dir=Path(source_dir),
            output_dir=output,
            state=state,
            state_path=state_path,
        )
    if "walker-swing-touchdown" in studies:
        state["results"]["walker_swing_touchdown"] = _run_walker_swing_touchdown_study(
            seeds=seeds,
            step_budget=step_budget or 50_000,
            eval_episodes=eval_episodes,
            config_dir=config_path,
            source_dir=Path(source_dir),
            output_dir=output,
            state=state,
            state_path=state_path,
        )
    if "walker-slip-recovery" in studies:
        state["results"]["walker_slip_recovery"] = (
            _run_walker_slip_recovery_study(
                seeds=seeds,
                step_budget=step_budget or 1_000_000,
                eval_episodes=eval_episodes,
                config_dir=config_path,
                source_dir=Path(source_dir),
                output_dir=output,
                state=state,
                state_path=state_path,
                approved_variant=approved_variant,
            )
        )
    if "walker-action-memory" in studies:
        state["results"]["walker_action_memory"] = (
            _run_walker_action_memory_study(
                seeds=seeds,
                step_budget=step_budget or WALKER_ACTION_MEMORY_SCREEN_STEPS,
                eval_episodes=eval_episodes,
                config_dir=config_path,
                output_dir=output,
                state=state,
                state_path=state_path,
                approved_variant=approved_variant,
            )
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
    report = {
        "identity": identity,
        "results": state["results"],
        "completed": state["completed"],
    }
    _atomic_json_write(output / "report.json", report)
    (output / "report.md").write_text(_format_report(report), encoding="utf-8")
    return report
