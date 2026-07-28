"""Detailed deterministic/stochastic diagnostics for walker checkpoints."""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from rl_framework.envs.registry import make_env
from rl_framework.training.evaluation_workers import (
    EpisodeSeedWrapper,
    episode_quotas,
    make_seeded_vec_env,
    resolve_evaluation_workers,
)
from rl_framework.utils.checkpoint import (
    find_vecnormalize_path_for_model,
    model_zip_path,
)

_GAIT_EPISODE_METRICS = (
    "gait_right_only_fraction",
    "gait_left_only_fraction",
    "gait_double_support_fraction",
    "gait_flight_fraction",
    "gait_right_touchdowns",
    "gait_left_touchdowns",
    "gait_alternating_touchdowns",
    "gait_alternating_touchdowns_per_100_steps",
    "gait_progress_per_alternating_touchdown",
    "gait_stance_slip_speed",
    "gait_longest_same_foot_sequence",
    "gait_mean_action_delta_l2",
    "gait_stride_length",
    "gait_swing_duration",
    "gait_foot_clearance",
    "gait_cadence_steps_per_min",
    "gait_qualified_touchdowns",
    "gait_sustained_swing_touchdowns",
)
_REWARD_EPISODE_METRICS = (
    "reward_alive_mean",
    "reward_velocity_mean",
    "reward_orientation_mean",
    "reward_action_mean",
    "reward_fall_mean",
    "reward_gait_step_progress_mean",
    "reward_swing_touchdown_progress_mean",
    "reward_stance_slip_mean",
    "reward_action_rate_mean",
    "reward_swing_clearance_mean",
    "reward_touchdown_rate_mean",
    "reward_flight_mean",
    "reward_gait_symmetry_mean",
)


def _model_class(cfg: dict[str, Any]):
    algorithm = str(cfg.get("training", {}).get("algorithm", "PPO")).upper()
    return {"PPO": PPO, "SAC": SAC, "TD3": TD3}[algorithm]


def _evaluation_env(
    cfg: dict[str, Any],
    model_path: str | Path,
    episodes: int,
) -> VecEnv:
    env_cfg = deepcopy(cfg["environment"])
    workers = resolve_evaluation_workers(cfg, episodes)
    vec_env: VecEnv = make_seeded_vec_env(
        "walker_bullet",
        env_cfg,
        workers,
        cfg.get("training", {}),
    )
    sidecar = find_vecnormalize_path_for_model(model_path)
    if sidecar is not None:
        vec_env = VecNormalize.load(str(sidecar), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    return vec_env


def _aggregate_episode_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    result = {
        "episode_length_mean": float(np.mean(values("episode_length"))),
        "episode_length_std": float(np.std(values("episode_length"))),
        "return_mean": float(np.mean(values("return"))),
        "return_std": float(np.std(values("return"))),
        "fall_rate": float(np.mean(values("fell"))),
        "peak_z_mean": float(np.mean(values("peak_z"))),
        "forward_displacement_mean": float(np.mean(values("forward_displacement"))),
        "forward_displacement_std": float(np.std(values("forward_displacement"))),
        "pushes_mean": float(np.mean(values("pushes"))),
        "push_recovery_rate": (
            float(sum(row["recovered_pushes"] for row in rows))
            / max(sum(row["pushes"] for row in rows), 1.0)
        ),
    }
    for key in _GAIT_EPISODE_METRICS:
        result[f"{key}_mean"] = float(np.mean(values(key)))
        result[f"{key}_std"] = float(np.std(values(key)))
    for key in _REWARD_EPISODE_METRICS:
        result[key] = float(np.mean(values(key)))
        result[f"{key}_std"] = float(np.std(values(key)))
    return result


def _rollouts(
    vec_env: VecEnv,
    *,
    episodes: int,
    seed: int,
    model: Any | None,
    deterministic: bool,
    recovery_window: int = 30,
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    workers = vec_env.num_envs
    quotas = episode_quotas(episodes, workers)
    completed = [0] * workers
    episode_returns = np.zeros(workers, dtype=np.float64)
    episode_lengths = np.zeros(workers, dtype=np.int64)
    initial_x: list[float | None] = [None] * workers
    final_x = np.zeros(workers, dtype=np.float64)
    peak_z = np.full(workers, float("-inf"), dtype=np.float64)
    push_steps: list[list[int]] = [[] for _ in range(workers)]
    vec_env.env_method("set_episode_seed_base", seed)
    obs = vec_env.reset()

    while any(done < quota for done, quota in zip(completed, quotas)):
        if model is None:
            action = np.zeros((workers, 10), dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = vec_env.step(action)
        reward_values = np.asarray(rewards).reshape(-1)
        for worker in range(workers):
            if completed[worker] >= quotas[worker]:
                continue
            episode_returns[worker] += float(reward_values[worker])
            episode_lengths[worker] += 1
            info = infos[worker]
            x_position = float(info.get("x_position", final_x[worker]))
            if initial_x[worker] is None:
                initial_x[worker] = x_position
            final_x[worker] = x_position
            peak_z[worker] = max(peak_z[worker], float(info.get("z_position", 0.0)))
            if info.get("push_applied", False):
                push_steps[worker].append(int(episode_lengths[worker]))
            if not bool(dones[worker]):
                continue

            reason = info.get("termination_reason")
            final_fell = (
                reason != "time_limit"
                if isinstance(reason, str)
                else bool(info.get("torso_contact", False))
            )
            walker_episode = info.get("walker_episode", {})
            row = {
                "episode_length": float(episode_lengths[worker]),
                "return": float(episode_returns[worker]),
                "fell": float(final_fell),
                "peak_z": (
                    float(peak_z[worker]) if np.isfinite(peak_z[worker]) else 0.0
                ),
                "forward_displacement": float(
                    final_x[worker] - (initial_x[worker] or 0.0)
                ),
                "pushes": float(len(push_steps[worker])),
                "recovered_pushes": float(
                    sum(
                        episode_lengths[worker] - step >= recovery_window
                        for step in push_steps[worker]
                    )
                ),
            }
            for key in _GAIT_EPISODE_METRICS:
                value = walker_episode.get(key, 0.0)
                row[key] = (
                    float(value)
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else 0.0
                )
            for key in _REWARD_EPISODE_METRICS:
                value = walker_episode.get(key, 0.0)
                row[key] = (
                    float(value)
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else 0.0
                )
            rows.append(row)
            completed[worker] += 1
            episode_returns[worker] = 0.0
            episode_lengths[worker] = 0
            initial_x[worker] = None
            final_x[worker] = 0.0
            peak_z[worker] = float("-inf")
            push_steps[worker] = []
    return _aggregate_episode_rows(rows)


def _verdict(
    baseline: dict[str, float],
    deterministic: dict[str, float],
    stochastic: dict[str, float],
    max_steps: int,
) -> str:
    best = max(
        (deterministic, "deterministic"),
        (stochastic, "stochastic"),
        key=lambda item: item[0]["return_mean"],
    )
    if best[0]["peak_z_mean"] > 1.4:
        return "reward_hack_high_peak_z"
    if (
        deterministic["episode_length_mean"] < baseline["episode_length_mean"]
        and stochastic["episode_length_mean"] < baseline["episode_length_mean"]
    ):
        return "untrained_equivalent"
    if stochastic["return_mean"] > 0 and stochastic["return_mean"] > 2.0 * max(
        deterministic["return_mean"], 1e-8
    ):
        return "deterministic_collapse"
    if (
        best[0]["episode_length_mean"] >= 0.95 * max_steps
        and best[0]["peak_z_mean"] < 1.0
        and best[0]["forward_displacement_mean"] >= 1.0
    ):
        return f"learned_walking_{best[1]}"
    return f"partial_learning_{best[1]}"


def evaluate_walker_checkpoint(
    cfg: dict[str, Any],
    model_path: str | Path,
    *,
    episodes: int = 20,
) -> dict[str, Any]:
    """Evaluate zero-action, deterministic, and stochastic walker behavior."""
    if cfg.get("environment", {}).get("type") != "walker_bullet":
        raise ValueError("walker diagnostics require environment.type=walker_bullet")
    path = model_zip_path(model_path)
    model = _model_class(cfg).load(str(path), device="cpu")
    seed = int(cfg.get("seed", 0))
    vec_env = _evaluation_env(cfg, path, episodes)
    try:
        baseline = _rollouts(
            vec_env,
            episodes=min(5, episodes),
            seed=seed,
            model=None,
            deterministic=True,
        )
        deterministic = _rollouts(
            vec_env,
            episodes=episodes,
            seed=seed,
            model=model,
            deterministic=True,
        )
        stochastic = _rollouts(
            vec_env,
            episodes=episodes,
            seed=seed,
            model=model,
            deterministic=False,
        )
    finally:
        vec_env.close()
    max_steps = int(
        cfg.get("environment", {}).get("termination", {}).get("max_steps", 800)
    )
    return {
        "algorithm": str(cfg.get("training", {}).get("algorithm", "PPO")).upper(),
        "model_path": str(path),
        "model_timesteps": int(getattr(model, "num_timesteps", 0)),
        "episodes": episodes,
        "zero_action": baseline,
        "deterministic": deterministic,
        "stochastic": stochastic,
        "verdict": _verdict(baseline, deterministic, stochastic, max_steps),
    }


def collect_walker_swing_event_telemetry(
    cfg: dict[str, Any],
    model_path: str | Path,
    *,
    episodes: int = 20,
) -> dict[str, Any]:
    """Summarize individual completed-swing events from stochastic rollouts.

    Quantiles are computed across events, not from episode means, so a short
    contact-chatter episode cannot receive the same influence as a sustained
    sequence of real airborne swings.
    """
    if cfg.get("environment", {}).get("type") != "walker_bullet":
        raise ValueError("walker diagnostics require environment.type=walker_bullet")
    path = model_zip_path(model_path)
    model = _model_class(cfg).load(str(path), device="cpu")
    vec_env = _evaluation_env(cfg, path, 1)
    durations: list[float] = []
    clearances: list[float] = []
    try:
        seed = int(cfg.get("seed", 0))
        for episode in range(episodes):
            vec_env.seed(seed + episode)
            obs = vec_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, _, dones, infos = vec_env.step(action)
                event = infos[0].get("swing_touchdown_event")
                if isinstance(event, dict):
                    duration = event.get("duration")
                    clearance = event.get("clearance")
                    if isinstance(duration, (int, float)) and isinstance(
                        clearance, (int, float)
                    ):
                        durations.append(float(duration))
                        clearances.append(float(clearance))
                done = bool(dones[0])
    finally:
        vec_env.close()

    def quantiles(values: list[float]) -> dict[str, float]:
        data = np.asarray(values, dtype=np.float64)
        if not len(data):
            return {"p50": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0}
        return {
            "p50": float(np.quantile(data, 0.50)),
            "p75": float(np.quantile(data, 0.75)),
            "p90": float(np.quantile(data, 0.90)),
            "max": float(np.max(data)),
        }

    return {
        "episodes": episodes,
        "event_count": len(durations),
        "duration_seconds": quantiles(durations),
        "clearance_meters": quantiles(clearances),
    }


def collect_walker_touchdown_placement_telemetry(
    cfg: dict[str, Any],
    model_path: str | Path,
    *,
    episodes: int = 20,
) -> dict[str, Any]:
    """Measure whether sustained touchdown placement predicts next-step progress."""
    if cfg.get("environment", {}).get("type") != "walker_bullet":
        raise ValueError("walker diagnostics require environment.type=walker_bullet")
    path = model_zip_path(model_path)
    model = _model_class(cfg).load(str(path), device="cpu")
    vec_env = _evaluation_env(cfg, path, 1)
    leads: list[float] = []
    next_progress: list[float] = []
    try:
        seed = int(cfg.get("seed", 0))
        for episode in range(episodes):
            vec_env.seed(seed + episode)
            obs = vec_env.reset()
            pending_pelvis_x: float | None = None
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, _, dones, infos = vec_env.step(action)
                event = infos[0].get("swing_touchdown_event")
                if isinstance(event, dict) and event.get("sustained") is True:
                    pelvis_x = event.get("pelvis_x")
                    foot_lead = event.get("foot_lead")
                    if isinstance(pelvis_x, (int, float)):
                        if pending_pelvis_x is not None:
                            progress = float(pelvis_x) - pending_pelvis_x
                            if leads:
                                next_progress.append(progress)
                        pending_pelvis_x = float(pelvis_x)
                        if isinstance(foot_lead, (int, float)):
                            leads.append(float(foot_lead))
                done = bool(dones[0])
    finally:
        vec_env.close()

    paired = min(len(leads), len(next_progress))
    lead_data = np.asarray(leads[:paired], dtype=np.float64)
    progress_data = np.asarray(next_progress[:paired], dtype=np.float64)
    correlation = (
        float(np.corrcoef(lead_data, progress_data)[0, 1])
        if paired >= 2 and np.std(lead_data) > 0 and np.std(progress_data) > 0
        else 0.0
    )
    return {
        "episodes": episodes,
        "paired_event_count": paired,
        "foot_lead_meters": {
            "mean": float(np.mean(lead_data)) if paired else 0.0,
            "p75": float(np.quantile(lead_data, 0.75)) if paired else 0.0,
        },
        "next_touchdown_progress_meters": {
            "mean": float(np.mean(progress_data)) if paired else 0.0,
            "p75": float(np.quantile(progress_data, 0.75)) if paired else 0.0,
        },
        "lead_to_next_progress_correlation": correlation,
    }


def collect_walker_stance_phase_telemetry(
    cfg: dict[str, Any],
    model_path: str | Path,
    *,
    episodes: int = 20,
) -> dict[str, Any]:
    """Test whether episode-average completed stance quality predicts progress."""
    if cfg.get("environment", {}).get("type") != "walker_bullet":
        raise ValueError("walker diagnostics require environment.type=walker_bullet")
    path = model_zip_path(model_path)
    model = _model_class(cfg).load(str(path), device="cpu")
    vec_env = _evaluation_env(cfg, path, 1)
    episode_rows: list[tuple[float, float, float, float]] = []
    completed_stances = 0
    try:
        seed = int(cfg.get("seed", 0))
        for episode in range(episodes):
            vec_env.seed(seed + episode)
            obs = vec_env.reset()
            advances: list[float] = []
            durations: list[float] = []
            slips: list[float] = []
            initial_x: float | None = None
            final_x = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, _, dones, infos = vec_env.step(action)
                info = infos[0]
                x_position = float(info.get("x_position", final_x))
                if initial_x is None:
                    initial_x = x_position
                final_x = x_position
                event = info.get("stance_completion_event")
                if isinstance(event, dict):
                    advance = event.get("pelvis_advance")
                    duration = event.get("duration")
                    slip = event.get("mean_slip_speed")
                    if all(
                        isinstance(value, (int, float))
                        for value in (advance, duration, slip)
                    ):
                        advances.append(float(advance))
                        durations.append(float(duration))
                        slips.append(float(slip))
                        completed_stances += 1
                done = bool(dones[0])
            if advances:
                episode_rows.append(
                    (
                        final_x - (initial_x or 0.0),
                        float(np.mean(advances)),
                        float(np.mean(durations)),
                        float(np.mean(slips)),
                    )
                )
    finally:
        vec_env.close()

    data = np.asarray(episode_rows, dtype=np.float64)

    def correlation(column: int) -> float:
        if len(data) < 2 or np.std(data[:, 0]) == 0 or np.std(data[:, column]) == 0:
            return 0.0
        return float(np.corrcoef(data[:, column], data[:, 0])[0, 1])

    def mean(column: int) -> float:
        return float(np.mean(data[:, column])) if len(data) else 0.0

    return {
        "episodes": episodes,
        "episodes_with_completed_stances": len(episode_rows),
        "completed_stance_count": completed_stances,
        "episode_mean_stance_pelvis_advance_meters": mean(1),
        "episode_mean_stance_duration_seconds": mean(2),
        "episode_mean_stance_slip_speed": mean(3),
        "correlation_with_episode_displacement": {
            "pelvis_advance": correlation(1),
            "duration": correlation(2),
            "slip_speed": correlation(3),
        },
    }


def _numeric_summary(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(np.mean(data)),
        "p50": float(np.quantile(data, 0.50)),
        "p90": float(np.quantile(data, 0.90)),
        "max": float(np.max(data)),
    }


def calibrate_walker_slip_thresholds(
    contact_slip_speeds: list[float],
    touchdown_normal_forces: list[float],
) -> dict[str, Any]:
    """Calibrate robust caps from pooled stochastic Phase-0 telemetry."""
    slip = np.asarray(contact_slip_speeds, dtype=np.float64)
    force = np.asarray(touchdown_normal_forces, dtype=np.float64)
    return {
        "stance_slip_penalty_clip": (
            float(np.quantile(slip, 0.90)) if len(slip) else 0.0
        ),
        "touchdown_normal_force_threshold": (
            float(np.quantile(force, 0.90)) if len(force) else 0.0
        ),
        "method": {
            "stance_slip_penalty_clip": "pooled contact-speed p90",
            "touchdown_normal_force_threshold": "pooled touchdown-force p90",
        },
        "contact_sample_count": int(len(slip)),
        "touchdown_sample_count": int(len(force)),
    }


def _fall_midstance_samples(window: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for side in ("right", "left"):
        spans: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for row in window:
            foot = row[side]
            if foot["contact"]:
                current.append(row)
            elif current:
                spans.append(current)
                current = []
        if current:
            spans.append(current)
        for span in spans:
            midpoint = span[len(span) // 2]
            samples.append(
                {
                    "side": side,
                    "step": midpoint["step"],
                    "time_to_fall_seconds": midpoint["time_to_fall_seconds"],
                    "tangential_velocity": midpoint[side]["tangential_velocity"],
                    "tangential_speed": midpoint[side]["tangential_speed"],
                    "normal_force": midpoint[side]["normal_force"],
                }
            )
    return samples


def classify_walker_fall_window(
    window: list[dict[str, Any]],
    *,
    control_timestep: float,
    slip_threshold: float,
    touchdown_force_threshold: float,
) -> str:
    """Assign one mutually exclusive, telemetry-backed fall mechanism."""
    if not window:
        return "recovery_failure"
    recent_steps = max(1, int(round(0.20 / control_timestep)))
    recent = window[-recent_steps:]
    for row in recent:
        for side in ("right", "left"):
            foot = row[side]
            if foot["touchdown"] and (
                foot["tangential_speed"] >= slip_threshold
                or (
                    touchdown_force_threshold > 0.0
                    and foot["normal_force"] >= touchdown_force_threshold
                )
            ):
                return "touchdown_overshoot"
    for row in recent:
        for side in ("right", "left"):
            foot = row[side]
            stance_age = foot.get("stance_age_steps")
            if (
                foot["contact"]
                and isinstance(stance_age, int)
                and stance_age >= recent_steps
                and foot["tangential_speed"] >= slip_threshold
            ):
                return "late_stance_push_off"
    return "recovery_failure"


def summarize_walker_fall_windows(
    fall_windows: list[dict[str, Any]],
    *,
    control_timestep: float,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Classify falls and summarize the requested half-second mechanisms."""
    slip_threshold = float(thresholds["stance_slip_penalty_clip"])
    force_threshold = float(thresholds["touchdown_normal_force_threshold"])
    cause_counts: Counter[str] = Counter()
    falls = []
    for fall in fall_windows:
        window = fall["window"]
        cause = classify_walker_fall_window(
            window,
            control_timestep=control_timestep,
            slip_threshold=slip_threshold,
            touchdown_force_threshold=force_threshold,
        )
        cause_counts[cause] += 1
        touchdown_samples = [
            {
                "side": side,
                "step": row["step"],
                "time_to_fall_seconds": row["time_to_fall_seconds"],
                "tangential_velocity": row[side]["tangential_velocity"],
                "tangential_speed": row[side]["tangential_speed"],
                "normal_force": row[side]["normal_force"],
            }
            for row in window
            for side in ("right", "left")
            if row[side]["touchdown"]
        ]
        midstance_samples = _fall_midstance_samples(window)
        falls.append(
            {
                **fall,
                "cause": cause,
                "touchdown_samples": touchdown_samples,
                "midstance_samples": midstance_samples,
                "window_summary": {
                    "normal_contact_force": _numeric_summary(
                        [
                            row[side]["normal_force"]
                            for row in window
                            for side in ("right", "left")
                            if row[side]["contact"]
                        ]
                    ),
                    "tangential_slip_speed": _numeric_summary(
                        [
                            row[side]["tangential_speed"]
                            for row in window
                            for side in ("right", "left")
                            if row[side]["contact"]
                        ]
                    ),
                    "slip_direction_mean": [
                        float(
                            np.mean(
                                [
                                    row[side]["tangential_velocity"][axis]
                                    for row in window
                                    for side in ("right", "left")
                                    if row[side]["contact"]
                                ]
                                or [0.0]
                            )
                        )
                        for axis in (0, 1)
                    ],
                    "absolute_roll": _numeric_summary(
                        [abs(float(row["roll"])) for row in window]
                    ),
                    "absolute_pitch": _numeric_summary(
                        [abs(float(row["pitch"])) for row in window]
                    ),
                    "action_delta_l2": _numeric_summary(
                        [float(row["action_delta_l2"]) for row in window]
                    ),
                    "achieved_velocity": _numeric_summary(
                        [float(row["achieved_velocity"]) for row in window]
                    ),
                    "absolute_velocity_error": _numeric_summary(
                        [abs(float(row["velocity_error"])) for row in window]
                    ),
                },
            }
        )
    total = max(len(falls), 1)
    return {
        "fall_count": len(falls),
        "cause_counts": dict(sorted(cause_counts.items())),
        "cause_rates": {
            cause: count / total for cause, count in sorted(cause_counts.items())
        },
        "falls": falls,
    }


def _fall_telemetry_env(
    cfg: dict[str, Any],
    model_path: Path,
    render_dir: str | Path | None,
) -> VecEnv:
    if render_dir is None:
        return _evaluation_env(cfg, model_path, 1)

    env_cfg = deepcopy(cfg["environment"])
    env_cfg["render_mode"] = "rgb_array"
    raw_env = make_env("walker_bullet", env_cfg)
    seeded_env: gym.Env = EpisodeSeedWrapper(raw_env, 0, 1)
    vec_env: VecEnv = DummyVecEnv([lambda: seeded_env])
    sidecar = find_vecnormalize_path_for_model(model_path)
    if sidecar is not None:
        vec_env = VecNormalize.load(str(sidecar), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    return vec_env


def collect_walker_fall_recovery_telemetry(
    cfg: dict[str, Any],
    model_path: str | Path,
    *,
    episodes: int = 5,
    window_seconds: float = 0.5,
    render_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Record stochastic pre-fall windows and calibration samples."""
    if cfg.get("environment", {}).get("type") != "walker_bullet":
        raise ValueError("walker diagnostics require environment.type=walker_bullet")
    if episodes <= 0 or window_seconds <= 0.0:
        raise ValueError("episodes and window_seconds must be positive")
    path = model_zip_path(model_path)
    model = _model_class(cfg).load(str(path), device="cpu")
    sim = cfg["environment"].get("sim", {})
    control_timestep = float(sim.get("timestep", 1.0 / 240.0)) * int(
        sim.get("frame_skip", 4)
    )
    window_steps = max(1, int(round(window_seconds / control_timestep)))
    vec_env = _fall_telemetry_env(cfg, path, render_dir)
    fall_windows: list[dict[str, Any]] = []
    contact_slip_speeds: list[float] = []
    touchdown_normal_forces: list[float] = []
    completed = 0
    episode_step = 0
    stance_ages: list[int | None] = [None, None]
    history: deque[dict[str, Any]] = deque(maxlen=window_steps)
    rendered_paths: list[str] = []
    render_frames: list[Any] = []
    render_stride = max(1, int(round((1.0 / control_timestep) / 20.0)))
    try:
        vec_env.env_method("set_episode_seed_base", int(cfg.get("seed", 0)))
        obs = vec_env.reset()
        while completed < episodes:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, dones, infos = vec_env.step(action)
            episode_step += 1
            info = infos[0]
            if render_dir is not None and (episode_step - 1) % render_stride == 0:
                from PIL import Image

                frame = vec_env.get_images()[0]
                if frame is not None:
                    render_frames.append(
                        Image.fromarray(frame).resize(
                            (320, 240), Image.Resampling.BILINEAR
                        )
                    )
            row: dict[str, Any] = {
                "step": episode_step,
                "roll": float(info.get("roll", 0.0)),
                "pitch": float(info.get("pitch", 0.0)),
                "action_delta_l2": float(info.get("action_delta_l2", 0.0)),
                "target_velocity": float(info.get("target_velocity", 0.0)),
                "achieved_velocity": float(info.get("lin_vel_x", 0.0)),
                "velocity_error": float(info.get("velocity_error", 0.0)),
            }
            for side_index, side in enumerate(("right", "left")):
                contact = bool(info.get(f"{side}_foot_contact", False))
                touchdown = bool(info.get(f"{side}_foot_touchdown", False))
                if contact:
                    stance_ages[side_index] = (
                        0
                        if touchdown or stance_ages[side_index] is None
                        else stance_ages[side_index] + 1
                    )
                else:
                    stance_ages[side_index] = None
                velocity = info.get(f"{side}_foot_tangential_velocity", (0.0, 0.0))
                velocity_pair = [float(velocity[0]), float(velocity[1])]
                speed = float(np.linalg.norm(velocity_pair))
                normal_force = float(info.get(f"{side}_foot_normal_force", 0.0))
                row[side] = {
                    "contact": contact,
                    "touchdown": touchdown,
                    "stance_age_steps": stance_ages[side_index],
                    "tangential_velocity": velocity_pair,
                    "tangential_speed": speed,
                    "normal_force": normal_force,
                }
                if contact:
                    contact_slip_speeds.append(speed)
                if touchdown:
                    touchdown_normal_forces.append(normal_force)
            history.append(row)
            if not bool(dones[0]):
                continue
            reason = info.get("termination_reason")
            fell = reason != "time_limit"
            if fell:
                window = [dict(sample) for sample in history]
                final_step = window[-1]["step"] if window else episode_step
                for sample in window:
                    sample["time_to_fall_seconds"] = (
                        final_step - sample["step"]
                    ) * control_timestep
                fall_windows.append(
                    {
                        "episode": completed,
                        "termination_reason": reason,
                        "window": window,
                    }
                )
            if render_dir is not None and render_frames:
                render_path = (
                    Path(render_dir) / f"stochastic-episode-{completed:03d}.gif"
                )
                render_path.parent.mkdir(parents=True, exist_ok=True)
                render_frames[0].save(
                    render_path,
                    save_all=True,
                    append_images=render_frames[1:],
                    duration=max(int(1000 * control_timestep * render_stride), 20),
                    loop=0,
                    optimize=False,
                )
                rendered_paths.append(str(render_path))
            completed += 1
            episode_step = 0
            stance_ages = [None, None]
            history.clear()
            render_frames = []
    finally:
        vec_env.close()

    thresholds = calibrate_walker_slip_thresholds(
        contact_slip_speeds, touchdown_normal_forces
    )
    summary = summarize_walker_fall_windows(
        fall_windows,
        control_timestep=control_timestep,
        thresholds=thresholds,
    )
    return {
        "model_path": str(path),
        "episodes": episodes,
        "window_seconds": window_seconds,
        "window_steps": window_steps,
        "control_timestep": control_timestep,
        "thresholds": thresholds,
        "calibration_samples": {
            "contact_slip_speeds": contact_slip_speeds,
            "touchdown_normal_forces": touchdown_normal_forces,
        },
        "render_paths": rendered_paths,
        **summary,
    }


def evaluate_walker_transfer_suite(
    cfg: dict[str, Any],
    model_path: str | Path,
    *,
    episodes: int = 20,
) -> dict[str, dict[str, Any]]:
    """Evaluate one checkpoint on flat, uneven, obstacle, and push terrains."""
    terrains: dict[str, dict[str, Any]] = {
        "flat": {"preset": "flat"},
        "uneven": {"preset": "uneven", "height": 0.025},
        "obstacles": {"preset": "obstacles", "obstacle_height": 0.10},
        "push_recovery": {
            "preset": "push_recovery",
            "push_recovery": {
                "interval_steps": 120,
                "start_step": 60,
                "force": 180.0,
            },
        },
    }
    results = {}
    for name, terrain in terrains.items():
        transfer_cfg = deepcopy(cfg)
        transfer_cfg["environment"]["terrain"] = terrain
        results[name] = evaluate_walker_checkpoint(
            transfer_cfg, model_path, episodes=episodes
        )
    return results
