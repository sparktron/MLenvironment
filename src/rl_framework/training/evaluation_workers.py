"""Process-backed helpers shared by walker evaluation paths."""

from __future__ import annotations

import os
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Any

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from rl_framework.envs.registry import make_env


class EpisodeSeedWrapper(gym.Wrapper):
    """Give each vector worker a deterministic, non-overlapping seed stream."""

    def __init__(self, env: gym.Env, worker_index: int, worker_count: int) -> None:
        super().__init__(env)
        self._worker_index = worker_index
        self._worker_count = worker_count
        self._next_seed = worker_index

    def set_episode_seed_base(self, base_seed: int) -> None:
        """Restart this worker's strided episode-seed sequence."""
        self._next_seed = int(base_seed) + self._worker_index

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        episode_seed = self._next_seed if seed is None else int(seed)
        self._next_seed = episode_seed + self._worker_count
        return self.env.reset(seed=episode_seed, options=options)


def resolve_evaluation_workers(cfg: dict[str, Any], episodes: int) -> int:
    """Return the configured evaluation process count, bounded by episodes."""
    configured = cfg.get("evaluation", {}).get("workers")
    available = os.cpu_count() or 1
    workers = available if configured is None else int(configured)
    return max(1, min(workers, int(episodes)))


def episode_quotas(episodes: int, workers: int) -> list[int]:
    """Distribute exact episode counts over strided worker seed streams."""
    return [
        max(0, (int(episodes) - worker + int(workers) - 1) // int(workers))
        for worker in range(int(workers))
    ]


def _make_seeded_env(
    env_type: str,
    env_cfg: dict[str, Any],
    worker_index: int,
    worker_count: int,
) -> gym.Env:
    env = make_env(env_type, deepcopy(env_cfg))
    return EpisodeSeedWrapper(env, worker_index, worker_count)


def make_seeded_vec_env(
    env_type: str,
    env_cfg: dict[str, Any],
    workers: int,
    training_cfg: dict[str, Any],
) -> VecEnv:
    """Build a deterministic walker VecEnv using one process per worker."""
    env_fns: list[Callable[[], gym.Env]] = [
        partial(_make_seeded_env, env_type, env_cfg, worker, workers)
        for worker in range(workers)
    ]
    if workers == 1:
        return DummyVecEnv(env_fns)

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLAS_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")
    start_method = training_cfg.get("worker_start_method")
    if start_method is None:
        return SubprocVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method=str(start_method))
