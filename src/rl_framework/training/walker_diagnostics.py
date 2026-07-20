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

    return {
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
                final_fell = bool(info.get("torso_contact", False))
        rows.append(
            {
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
        )
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
        and best[0]["forward_displacement_mean"] > 0.0
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
