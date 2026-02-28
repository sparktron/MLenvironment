from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_framework.envs.registry import make_env
from rl_framework.utils.logging_utils import append_metrics_csv, create_experiment_paths


def evaluate(cfg: dict[str, Any], model_path: str) -> dict[str, float]:
    env_cfg = cfg["environment"]
    paths = create_experiment_paths(cfg["output"]["base_dir"], cfg["experiment_name"], cfg["seed"])

    if env_cfg["type"] == "organism_arena_parallel":
        # Basic rollout for multi-agent mode (shared policy can be loaded via train path).
        env = make_env(env_cfg["type"], env_cfg)
        obs, _ = env.reset(seed=cfg["seed"])
        rewards = []
        for _ in range(cfg["evaluation"].get("max_steps", 200)):
            actions = {agent: env.action_space(agent).sample() for agent in obs.keys()}
            obs, rew, term, trunc, _ = env.step(actions)
            rewards.append(float(np.mean(list(rew.values()))))
            if any(term.values()) or all(trunc.values()):
                break
        metrics = {"mean_reward": float(np.mean(rewards)) if rewards else 0.0}
    else:
        vec_env = DummyVecEnv([lambda: make_env(env_cfg["type"], env_cfg)])
        vn_path = Path(model_path).with_name("vecnormalize.pkl")
        if vn_path.exists():
            vec_env = VecNormalize.load(str(vn_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

        model = PPO.load(model_path)
        episodes = cfg["evaluation"].get("episodes", 5)
        returns = []
        for _ in range(episodes):
            obs = vec_env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                ep_ret += float(reward[0])
            returns.append(ep_ret)
        metrics = {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns))}

    append_metrics_csv(paths.logs_dir / "eval_metrics.csv", metrics)
    return metrics
