from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize

from rl_framework.core.registry import get_task
from rl_framework.training.evaluation_workers import (
    episode_quotas,
    make_seeded_vec_env,
    resolve_evaluation_workers,
)
from rl_framework.utils.checkpoint import find_vecnormalize_path_for_model
from rl_framework.utils.logging_utils import append_metrics_csv, create_experiment_paths


def _was_truncated(infos: Any) -> bool:
    """Infer truncation from VecEnv infos payload for single-env evaluation."""
    if isinstance(infos, dict):
        infos = [infos]
    if not isinstance(infos, (list, tuple)):
        return False
    for info in infos:
        if not isinstance(info, dict):
            continue
        if info.get("TimeLimit.truncated", False):
            return True
        outcome = info.get("episode_outcome")
        if isinstance(outcome, dict) and outcome.get("outcome") == "timeout":
            return True
    return False


def evaluate(cfg: dict[str, Any], model_path: str) -> dict[str, float]:
    env_cfg = cfg["environment"]
    episodes = int(cfg.get("evaluation", {}).get("episodes", 5))
    paths = create_experiment_paths(
        cfg["output"]["base_dir"],
        cfg["experiment_name"],
        cfg["seed"],
        run_id=cfg["output"].get("run_id"),
    )

    spec = get_task(env_cfg["type"])
    if spec.build_eval_vec_env is not None:
        vec_env = spec.build_eval_vec_env(env_cfg)
    else:
        workers = resolve_evaluation_workers(cfg, episodes)
        vec_env = make_seeded_vec_env(
            env_cfg["type"], env_cfg, workers, cfg.get("training", {})
        )

    vn_path = find_vecnormalize_path_for_model(model_path)
    if vn_path is not None:
        vec_env = VecNormalize.load(str(vn_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    try:
        algorithm = str(cfg.get("training", {}).get("algorithm", "PPO")).upper()
        device = str(cfg.get("training", {}).get("device", "cpu"))
        model: BaseAlgorithm
        if algorithm == "PPO":
            model = PPO.load(model_path, device=device)
        elif algorithm == "SAC":
            model = SAC.load(model_path, device=device)
        elif algorithm == "TD3":
            model = TD3.load(model_path, device=device)
        else:
            raise ValueError(f"Unsupported training.algorithm: {algorithm!r}")
        if spec.evaluate_rollouts is not None:
            metrics = spec.evaluate_rollouts(vec_env, model, episodes)
        else:
            returns = []
            terminated_episodes = 0
            truncated_episodes = 0
            episode_lengths = []
            completed = [0] * vec_env.num_envs
            quotas = episode_quotas(episodes, vec_env.num_envs)
            running_returns = np.zeros(vec_env.num_envs, dtype=np.float64)
            running_lengths = np.zeros(vec_env.num_envs, dtype=np.int64)
            vec_env.env_method(
                "set_episode_seed_base",
                int(cfg.get("seed", 0)),
            )
            obs = vec_env.reset()
            while any(done < quota for done, quota in zip(completed, quotas)):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = vec_env.step(action)
                reward_arr = np.asarray(reward).flatten()
                for worker in range(vec_env.num_envs):
                    if completed[worker] >= quotas[worker]:
                        continue
                    running_returns[worker] += float(reward_arr[worker])
                    running_lengths[worker] += 1
                    if not bool(dones[worker]):
                        continue
                    if _was_truncated([infos[worker]]):
                        truncated_episodes += 1
                    else:
                        terminated_episodes += 1
                    returns.append(float(running_returns[worker]))
                    episode_lengths.append(int(running_lengths[worker]))
                    completed[worker] += 1
                    running_returns[worker] = 0.0
                    running_lengths[worker] = 0
            metrics: dict[str, float] = {
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns)),
                "terminated_rate": float(terminated_episodes / episodes),
                "truncated_rate": float(truncated_episodes / episodes),
                "mean_episode_length": float(np.mean(episode_lengths)),
            }
        append_metrics_csv(paths.logs_dir / "eval_metrics.csv", metrics)
    finally:
        vec_env.close()
    return metrics
