from __future__ import annotations

from typing import Any

import gymnasium as gym
from pettingzoo.utils.env import ParallelEnv

from rl_framework.envs.locomotion.walker_bullet import WalkerBulletEnv
from rl_framework.envs.organisms.arena_parallel import OrganismArenaParallelEnv


def make_env(env_type: str, cfg: dict[str, Any]) -> gym.Env | ParallelEnv:
    if env_type == "walker_bullet":
        return WalkerBulletEnv(cfg)
    if env_type == "organism_arena_parallel":
        return OrganismArenaParallelEnv(cfg)
    raise ValueError(f"Unknown env_type={env_type}")
