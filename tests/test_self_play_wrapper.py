from unittest.mock import MagicMock

import numpy as np

from rl_framework.envs.organisms.arena_parallel import OrganismArenaParallelEnv
from rl_framework.training.self_play_env_wrapper import SelfPlayEnvWrapper


def _make_arena(max_steps=10):
    return OrganismArenaParallelEnv({"battle_rules": {"max_steps": max_steps}})


def _make_callback(has_policy=True):
    callback = MagicMock()
    if has_policy:
        mock_policy = MagicMock()
        mock_policy.predict.return_value = (np.zeros((1, 3), dtype=np.float32), None)
        callback.sample_opponent.return_value = mock_policy
    else:
        callback.sample_opponent.return_value = None
    return callback


def test_wrapper_uses_frozen_policy_for_agent_1():
    env = _make_arena()
    callback = _make_callback(has_policy=True)
    wrapped = SelfPlayEnvWrapper(env, callback)
    wrapped.reset()
    frozen_policy = wrapped._frozen_policy
    wrapped.step(
        {
            "agent_0": env.action_space("agent_0").sample(),
            "agent_1": env.action_space("agent_1").sample(),
        }
    )
    assert frozen_policy.predict.called


def test_wrapper_uses_random_action_when_league_empty():
    env = _make_arena()
    callback = _make_callback(has_policy=False)
    wrapped = SelfPlayEnvWrapper(env, callback)
    wrapped.reset()
    # Should not raise
    wrapped.step(
        {
            "agent_0": env.action_space("agent_0").sample(),
            "agent_1": env.action_space("agent_1").sample(),
        }
    )


def test_wrapper_resamples_opponent_on_each_reset():
    env = _make_arena()
    callback = _make_callback(has_policy=True)
    wrapped = SelfPlayEnvWrapper(env, callback)
    wrapped.reset()
    wrapped.reset()
    assert callback.sample_opponent.call_count == 2


def test_wrapper_forwards_agents_property():
    env = _make_arena()
    callback = _make_callback(has_policy=False)
    wrapped = SelfPlayEnvWrapper(env, callback)
    wrapped.reset()
    assert set(wrapped.agents) == {"agent_0", "agent_1"}


def test_wrapper_clears_agents_after_episode_ends():
    env = _make_arena(max_steps=1)
    callback = _make_callback(has_policy=False)
    wrapped = SelfPlayEnvWrapper(env, callback)
    wrapped.reset()
    wrapped.step(
        {
            "agent_0": env.action_space("agent_0").sample(),
            "agent_1": env.action_space("agent_1").sample(),
        }
    )
    assert wrapped.agents == []
