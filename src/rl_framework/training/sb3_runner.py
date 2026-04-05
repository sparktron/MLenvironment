from __future__ import annotations

from pathlib import Path
from typing import Any

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_framework.envs.registry import make_env
from rl_framework.training.curriculum_callback import CurriculumCallback
from rl_framework.utils.logging_utils import create_experiment_paths


def _build_single_env(env_cfg: dict[str, Any]):
    return lambda: make_env(env_cfg["type"], env_cfg)


def train(cfg: dict[str, Any]) -> Path:
    paths = create_experiment_paths(cfg["output"]["base_dir"], cfg["experiment_name"], cfg["seed"])
    env_cfg = cfg["environment"]

    if env_cfg["type"] == "organism_arena_parallel":
        par_env = make_env(env_cfg["type"], env_cfg)
        vec_env = ss.pettingzoo_env_to_vec_env_v1(par_env)
        vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=1, base_class="stable_baselines3")
    else:
        vec_env = DummyVecEnv([_build_single_env(env_cfg)])

    if cfg["training"].get("normalize_observations", True):
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    model = PPO(
        policy=cfg["training"].get("policy", "MlpPolicy"),
        env=vec_env,
        learning_rate=cfg["training"].get("learning_rate", 3e-4),
        n_steps=cfg["training"].get("n_steps", 1024),
        batch_size=cfg["training"].get("batch_size", 256),
        tensorboard_log=str(paths.logs_dir),
        seed=cfg["seed"],
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=cfg["training"].get("checkpoint_every", 10000),
        save_path=str(paths.checkpoints_dir),
        name_prefix="ppo_model",
    )
    callbacks = [checkpoint_cb]

    # Curriculum learning: bump env difficulty when performance exceeds threshold.
    curriculum_cfg = cfg.get("curriculum", {})
    if curriculum_cfg.get("enabled", False):
        callbacks.append(CurriculumCallback(curriculum_cfg, env_cfg, verbose=1))

    model.learn(total_timesteps=cfg["training"]["total_timesteps"], callback=callbacks)
    final_path = paths.checkpoints_dir / "final_model"
    model.save(str(final_path))
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(paths.checkpoints_dir / "vecnormalize.pkl"))
    return final_path
