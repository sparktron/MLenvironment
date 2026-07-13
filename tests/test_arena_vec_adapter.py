from __future__ import annotations

import warnings

import supersuit as ss

from rl_framework.envs.registry import make_env
from rl_framework.training.sb3_runner import _ArenaVecEnvAdapter


def test_arena_adapter_supplies_render_mode_without_supersuit_warning() -> None:
    env = make_env("organism_arena_parallel", {"type": "organism_arena_parallel", "seed": 0})
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=1, base_class="stable_baselines3")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapted = _ArenaVecEnvAdapter(vec_env)
        assert adapted.render_mode is None
        assert not any("render_mode attribute is not defined" in str(item.message) for item in caught)
    finally:
        vec_env.close()
