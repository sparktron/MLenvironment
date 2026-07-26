"""Detailed deterministic/stochastic diagnostics for walker checkpoints."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_framework.envs.registry import make_env
from rl_framework.utils.checkpoint import find_vecnormalize_path_for_model, model_zip_path

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
)


def _model_class(cfg: dict[str, Any]):
    algorithm = str(cfg.get("training", {}).get("algorithm", "PPO")).upper()
    return {"PPO": PPO, "SAC": SAC, "TD3": TD3}[algorithm]


def _evaluation_env(
    cfg: dict[str, Any], model_path: str | Path
) -> DummyVecEnv | VecNormalize:
    env_cfg = deepcopy(cfg["environment"])
    vec_env: DummyVecEnv | VecNormalize = DummyVecEnv(
        [lambda: make_env("walker_bullet", env_cfg)]
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
    vec_env: DummyVecEnv | VecNormalize,
    *,
    episodes: int,
    seed: int,
    model: Any | None,
    deterministic: bool,
    recovery_window: int = 30,
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for episode in range(episodes):
        vec_env.seed(seed + episode)
        obs = vec_env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        initial_x: float | None = None
        final_x = 0.0
        peak_z = float("-inf")
        final_fell = False
        push_steps: list[int] = []
        while not done:
            if model is None:
                action = np.zeros((1, 10), dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = vec_env.step(action)
            episode_return += float(np.asarray(rewards).reshape(-1)[0])
            episode_length += 1
            info = infos[0]
            x_position = float(info.get("x_position", final_x))
            if initial_x is None:
                initial_x = x_position
            final_x = x_position
            peak_z = max(peak_z, float(info.get("z_position", 0.0)))
            if info.get("push_applied", False):
                push_steps.append(episode_length)
            done = bool(dones[0])
            if done:
                reason = info.get("termination_reason")
                final_fell = (
                    reason != "time_limit"
                    if isinstance(reason, str)
                    else bool(info.get("torso_contact", False))
                )
        walker_episode = info.get("walker_episode", {})
        row = {
            "episode_length": float(episode_length),
            "return": episode_return,
            "fell": float(final_fell),
            "peak_z": peak_z if np.isfinite(peak_z) else 0.0,
            "forward_displacement": final_x - (initial_x or 0.0),
            "pushes": float(len(push_steps)),
            "recovered_pushes": float(
                sum(episode_length - step >= recovery_window for step in push_steps)
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
    if (
        stochastic["return_mean"] > 0
        and stochastic["return_mean"] > 2.0 * max(deterministic["return_mean"], 1e-8)
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
    vec_env = _evaluation_env(cfg, path)
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
    max_steps = int(cfg.get("environment", {}).get("termination", {}).get("max_steps", 800))
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
    vec_env = _evaluation_env(cfg, path)
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
                    if isinstance(duration, (int, float)) and isinstance(clearance, (int, float)):
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
