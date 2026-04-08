from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_framework.envs.registry import make_env
from rl_framework.utils.logging_utils import append_metrics_csv, create_experiment_paths


def _was_truncated(infos: Any) -> bool:
    """Infer truncation from VecEnv infos payload for single-env evaluation."""
    info0 = infos[0] if isinstance(infos, list) and infos else {}
    return bool(info0.get("TimeLimit.truncated", False))


def evaluate(cfg: dict[str, Any], model_path: str) -> dict[str, float]:
    env_cfg = cfg["environment"]
    paths = create_experiment_paths(cfg["output"]["base_dir"], cfg["experiment_name"], cfg["seed"])

    if env_cfg["type"] == "organism_arena_parallel":
        # Multi-agent eval: wrap env the same way as training and load the shared PPO model.
        par_env = make_env(env_cfg["type"], env_cfg)
        vec_env = ss.pettingzoo_env_to_vec_env_v1(par_env)
        vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=1, base_class="stable_baselines3")
    else:
        vec_env = DummyVecEnv([lambda: make_env(env_cfg["type"], env_cfg)])
        vn_path = Path(model_path).with_name("vecnormalize.pkl")
        if vn_path.exists():
            vec_env = VecNormalize.load(str(vn_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

    is_multiagent = env_cfg["type"] == "organism_arena_parallel"
    num_agents = len(vec_env.observation_space) if is_multiagent else 1

    try:
        model = PPO.load(model_path)
        episodes = cfg["evaluation"].get("episodes", 5)
        returns = []
        per_agent_returns: dict[int, list[float]] = {i: [] for i in range(num_agents)}
        terminated_episodes = 0
        truncated_episodes = 0
        episode_lengths = []
        for _ in range(episodes):
            obs = vec_env.reset()
            done = False
            ep_ret = 0.0
            ep_agent_ret = np.zeros(num_agents, dtype=np.float64)
            ep_len = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = vec_env.step(action)
                reward_arr = np.asarray(reward).flatten()
                ep_ret += float(np.sum(reward_arr))
                ep_agent_ret[:len(reward_arr)] += reward_arr.astype(np.float64)
                ep_len += 1
                done = bool(np.any(dones))
            was_truncated = _was_truncated(infos)
            if was_truncated:
                truncated_episodes += 1
            else:
                terminated_episodes += 1
            episode_lengths.append(ep_len)
            returns.append(ep_ret)
            for i in range(num_agents):
                per_agent_returns[i].append(float(ep_agent_ret[i]))
        metrics: dict[str, float] = {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "terminated_rate": float(terminated_episodes / episodes),
            "truncated_rate": float(truncated_episodes / episodes),
            "mean_episode_length": float(np.mean(episode_lengths)),
        }
        # For multi-agent envs, add per-agent return metrics so zero-sum
        # cancellation doesn't hide individual agent performance.
        if is_multiagent:
            for i in range(num_agents):
                metrics[f"agent_{i}_mean_return"] = float(np.mean(per_agent_returns[i]))
                metrics[f"agent_{i}_std_return"] = float(np.std(per_agent_returns[i]))
    finally:
        vec_env.close()

    append_metrics_csv(paths.logs_dir / "eval_metrics.csv", metrics)
    return metrics
