"""Registry behavior, including loading plugins in fresh spawn workers."""
from functools import partial

import numpy as np
import pytest
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl_framework.core.registry import TaskSpec, get_task, make_env, register_task, registered_tasks
from rl_framework.training.self_play_env_wrapper import LeagueSampler, SelfPlayEnvWrapper, SingleAgentArenaEnv


def _spawn_env(name, league_dir):
    env = make_env(name, {"type": name, "seed": 0})
    if name == "organism_arena_parallel":
        env = SingleAgentArenaEnv(SelfPlayEnvWrapper(env, LeagueSampler(league_dir)))
    return env


def test_builtin_registration_and_unknown_task():
    assert {"walker_bullet", "organism_arena_parallel"} <= set(registered_tasks())
    assert get_task("walker_bullet").name == "walker_bullet"
    with pytest.raises(ValueError, match="walker_bullet"):
        get_task("unknown-task")


def test_duplicate_registration_rejected():
    existing = get_task("walker_bullet")
    register_task(existing)
    with pytest.raises(ValueError, match="already registered"):
        register_task(TaskSpec(name=existing.name, make_env=lambda cfg: None))


@pytest.mark.parametrize("name", ["walker_bullet", "organism_arena_parallel"])
def test_spawn_workers_load_plugins(name, tmp_path):
    env = SubprocVecEnv([partial(_spawn_env, name, tmp_path)] * 2, start_method="spawn")
    try:
        env.reset()
        actions = np.stack([env.action_space.sample() for _ in range(2)])
        observations, _, _, _ = env.step(actions)
        assert len(observations) == 2
    finally:
        env.close()
