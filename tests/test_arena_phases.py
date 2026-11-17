"""Arena rules driven phase by phase, without the PettingZoo step wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from rl_framework.games.arena.arena_parallel import (
    LIVE_PARAM_SET,
    LIVE_PARAMS,
    OrganismArenaParallelEnv,
)


def _env(**overrides):
    cfg = {"seed": 0, "battle_rules": {"attack_range": 5.0, "cooldown_steps": 3}}
    cfg.update(overrides)
    env = OrganismArenaParallelEnv(cfg)
    env.reset(seed=0)
    return env


def test_live_keys_declared_once_and_cover_rules_fields():
    assert LIVE_PARAMS == LIVE_PARAM_SET.keys
    assert {"battle_rules.damage", "resources.food_energy", "sim.arena_half_extent"} <= LIVE_PARAMS
    assert "morphology.base_size" in LIVE_PARAMS


def test_live_update_casts_and_mirrors_into_cfg():
    env = _env()
    env.update_live_params(
        {"battle_rules.cooldown_steps": 2.0, "reward.damage_scale": "0.25",
         "sim.arena_half_extent": 0.5, "morphology.base_size": 1.5}
    )
    assert env.rules.cooldown_steps == 2 and isinstance(env.rules.cooldown_steps, int)
    assert env._damage_scale == 0.25 == env.cfg["reward"]["damage_scale"]
    assert env.bounds == 0.5 == env.cfg["sim"]["arena_half_extent"]
    assert env.cfg["morphology"]["base_size"] == 1.5
    with pytest.raises(ValueError, match="Unknown live parameters"):
        env.update_live_params({"battle_rules.nope": 1})


def test_combat_phase_applies_damage_and_dense_reward_alone():
    env = _env()
    a, b = env.possible_agents[:2]
    env.state[a]["pos"] = np.zeros(2, dtype=np.float32)
    env.state[b]["pos"] = np.array([0.01, 0.0], dtype=np.float32)
    for name in env.possible_agents[2:]:
        env._dead.add(name)
    rewards = {name: 0.0 for name in env.agents}
    health_before = env.state[b]["health"]
    actions = {name: np.array([0, 0, 1.0], dtype=np.float32) for name in env.agents}
    env._phase_combat(actions, list(env.agents), rewards)
    assert env.state[b]["health"] < health_before
    assert env.state[a]["health"] < env.state[a]["max_health"]  # mutual strikes
    assert sum(rewards.values()) == pytest.approx(0.0)  # dense reward is zero-sum
    assert env.state[a]["cooldown"] == env.rules.cooldown_steps


def test_knockout_phase_penalises_once():
    env = _env()
    victim = env.agents[0]
    env.state[victim]["health"] = 0.0
    rewards = {name: 0.0 for name in env.agents}
    env._phase_knockouts(list(env.agents), rewards)
    env._phase_knockouts(list(env.agents), rewards)
    assert rewards[victim] == -1.0 and victim in env._dead


def test_rules_import_without_matplotlib_and_renderer_is_separate():
    from rl_framework.games.arena import arena_render

    env = _env()
    assert isinstance(env._renderer, arena_render.ArenaRenderer)
    env.step({name: np.zeros(3, dtype=np.float32) for name in env.agents})
    assert env._renderer._fig is None  # stepping never touches matplotlib
