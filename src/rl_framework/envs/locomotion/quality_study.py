"""Walker learning-quality studies."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np

from rl_framework.training.quality_study import (
    _apply_overrides, _atomic_json_write, _load_cfg, _prepare_cfg, _train_run,
)
from rl_framework.training.walker_diagnostics import (
    calibrate_walker_slip_thresholds,
    collect_walker_fall_recovery_telemetry,
    evaluate_walker_checkpoint,
    evaluate_walker_transfer_suite,
    summarize_walker_fall_windows,
)
from rl_framework.training.velocity_target_ramp_callback import VelocityTargetRampCallback


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


WALKER_GAIT_GATE: dict[str, Any] = {
    **WALKER_VELOCITY_GATE,
    "gate_version": 4,
    "min_alternating_touchdowns_per_100_steps": 1.5,
    "max_alternating_touchdowns_per_100_steps": 8.5,
    "min_progress_per_alternating_touchdown": 0.05,
    "min_stride_length": 0.10,
    "min_foot_clearance": 0.02,
    "slip_metric": "gait_mid_stance_contact_point_slip_speed_mean",
    "max_slip_speed": 0.18,
    "max_flight_fraction": 0.60,
    "max_contact_duty_imbalance": 0.15,
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


ALGORITHM_CONFIGS = {
    "PPO": "walker_ppo_baseline",
    "SAC": "walker_sac_baseline",
    "TD3": "walker_td3_baseline",
}


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
        and metrics["gait_flight_fraction_mean"]
        <= WALKER_GAIT_GATE["max_flight_fraction"]
        # Replaces the v2 support criteria: excludes an asymmetric bound
        # directly rather than using the flight ceiling as a proxy for it.
        and metrics["gait_contact_duty_imbalance_mean"]
        <= WALKER_GAIT_GATE["max_contact_duty_imbalance"]
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
            "double_support_fraction_mean": mean("gait_double_support_fraction_mean"),
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
                    [metrics[WALKER_GAIT_GATE["slip_metric"]] for metrics in stochastic]
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
        "launch": metrics["peak_z_mean"] >= WALKER_SLIP_RECOVERY_GATE["max_peak_z"],
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
        "fall_rate": metrics["fall_rate"] <= WALKER_SLIP_RECOVERY_GATE["max_fall_rate"],
        "slip_speed": metrics[WALKER_SLIP_RECOVERY_GATE["slip_metric"]]
        <= WALKER_SLIP_RECOVERY_GATE["max_slip_speed"],
        "forward_displacement": metrics["forward_displacement_mean"]
        >= WALKER_SLIP_RECOVERY_GATE["min_forward_displacement"],
        "peak_z": metrics["peak_z_mean"] < WALKER_SLIP_RECOVERY_GATE["max_peak_z"],
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
            and cause_counts["touchdown_overshoot"] == max(cause_counts.values())
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
            seed_source = (
                source_dir / f"seed_{seed}" / "checkpoints" / "final_model.zip"
            )
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
            seed_label: _walker_slip_recovery_gate_result(result)["metrics_gate_passed"]
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
        "fall_rate": metrics["fall_rate"] <= WALKER_SLIP_RECOVERY_GATE["max_fall_rate"],
        "slip_speed": metrics[WALKER_SLIP_RECOVERY_GATE["slip_metric"]]
        <= WALKER_SLIP_RECOVERY_GATE["max_slip_speed"],
        "forward_displacement": metrics["forward_displacement_mean"]
        >= WALKER_SLIP_RECOVERY_GATE["min_forward_displacement"],
        "peak_z": metrics["peak_z_mean"] < WALKER_SLIP_RECOVERY_GATE["max_peak_z"],
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
        "flight_fraction": metrics["gait_flight_fraction_mean"]
        <= WALKER_GAIT_GATE["max_flight_fraction"],
        "contact_duty_imbalance": metrics["gait_contact_duty_imbalance_mean"]
        <= WALKER_GAIT_GATE["max_contact_duty_imbalance"],
        "same_foot_sequence": metrics["gait_longest_same_foot_sequence_mean"]
        <= WALKER_GAIT_GATE["max_longest_same_foot_sequence"],
    }
    flags = {
        "launch": not criteria["peak_z"],
        "sliding": (criteria["forward_displacement"] and not criteria["slip_speed"]),
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
                "double_support_fraction": metrics["gait_double_support_fraction_mean"],
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
            if row["variant"] == "previous_action" and row["metrics_gate_passed"]
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
            seed_label: _walker_action_memory_gate_result(result)["metrics_gate_passed"]
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
                phase = "phase1" if seed == phase1_seed else "phase2_confirmation"
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



def study_specs():
    from rl_framework.core.studies import StudySpec

    return (
        StudySpec("walker", "walker", _run_walker_study, len(WALKER_VARIANTS), 750_000, 0, include_in_all=True),
        StudySpec("walker-velocity", "walker_velocity", _run_walker_velocity_study, len(WALKER_VELOCITY_VARIANTS), 150_000, 1, uses_source=True),
        StudySpec("walker-gait", "walker_gait", _run_walker_gait_study, len(WALKER_GAIT_VARIANTS), 150_000, 2, uses_source=True),
        StudySpec("walker-velocity-ramp", "walker_velocity_ramp", _run_walker_velocity_ramp_study, len(WALKER_VELOCITY_RAMP_VARIANTS), 50_000, 3, uses_source=True),
        StudySpec("walker-swing-touchdown", "walker_swing_touchdown", _run_walker_swing_touchdown_study, len(WALKER_SWING_TOUCHDOWN_VARIANTS), 50_000, 4, uses_source=True),
        StudySpec("walker-slip-recovery", "walker_slip_recovery", _run_walker_slip_recovery_study, len(WALKER_SLIP_RECOVERY_VARIANTS), 1_000_000, 5, per_seed=False, uses_source=True, uses_approval=True),
        StudySpec("walker-action-memory", "walker_action_memory", _run_walker_action_memory_study, len(WALKER_ACTION_MEMORY_VARIANTS), WALKER_ACTION_MEMORY_SCREEN_STEPS, 6, per_seed=False, uses_approval=True),
        StudySpec("algorithms", "algorithms", _run_algorithm_study, len(ALGORITHM_CONFIGS) * 2, 500_000, 8, include_in_all=True, uses_wall_clock=True),
    )
