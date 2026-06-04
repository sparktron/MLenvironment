"""Tests for reward annealing and the win-rate-gated curriculum (Feature 5)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rl_framework.training.curriculum_callback import CurriculumCallback
from rl_framework.training.reward_annealing_callback import RewardAnnealingCallback


class _FakeVecEnv:
    """Captures env_method calls the way SB3's VecEnv would receive them."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def env_method(self, method_name, *args, **kwargs):
        self.calls.append((method_name, args, kwargs))
        return [None]


class _StubLogger:
    def __init__(self, values=None) -> None:
        self.name_to_value = dict(values or {})
        self.records: dict[str, float] = {}

    def record(self, key, value) -> None:
        self.records[key] = value


def _attach(cb, *, num_timesteps=0, env=None, logger=None):
    cb.model = SimpleNamespace(get_env=lambda: env, logger=logger)
    cb.num_timesteps = num_timesteps


# -- RewardAnnealingCallback -------------------------------------------------


def test_reward_annealing_validates_anneal_steps() -> None:
    with pytest.raises(ValueError, match="anneal_steps must be > 0"):
        RewardAnnealingCallback(anneal_steps=0)


def test_reward_annealing_pushes_decreasing_scale() -> None:
    env = _FakeVecEnv()
    cb = RewardAnnealingCallback(anneal_steps=1000)

    _attach(cb, num_timesteps=0, env=env)
    cb._on_step()
    _attach(cb, num_timesteps=250, env=env)
    cb._on_step()
    _attach(cb, num_timesteps=1000, env=env)
    cb._on_step()

    scales = [call[1][0]["reward.damage_scale"] for call in env.calls]
    assert scales[0] == pytest.approx(1.0)
    assert scales[1] == pytest.approx(0.75)
    assert scales[2] == pytest.approx(0.0)
    assert all(call[0] == "update_live_params" for call in env.calls)


def test_reward_annealing_clamps_at_zero_and_dedupes() -> None:
    env = _FakeVecEnv()
    cb = RewardAnnealingCallback(anneal_steps=100)
    # Past anneal_steps the scale stays 0.0 and should not re-push every step.
    _attach(cb, num_timesteps=200, env=env)
    cb._on_step()
    _attach(cb, num_timesteps=300, env=env)
    cb._on_step()
    scales = [call[1][0]["reward.damage_scale"] for call in env.calls]
    assert scales == [0.0]  # second identical update suppressed


# -- CurriculumCallback configurable metric ----------------------------------


def test_curriculum_gates_on_configured_metric() -> None:
    env = _FakeVecEnv()
    cur_cfg = {
        "enabled": True,
        "metric": "arena/agent_0_win_rate",
        "level_up_threshold": 0.6,
        "max_level": 2,
        "level_params": {1: {"battle_rules.cooldown_steps": 4}},
    }
    cb = CurriculumCallback(cur_cfg, env_cfg={}, verbose=0)
    logger = _StubLogger({"arena/agent_0_win_rate": 0.5})
    _attach(cb, env=env, logger=logger)

    cb._on_rollout_end()
    assert cb.current_level == 0, "below threshold: no level up"

    logger.name_to_value["arena/agent_0_win_rate"] = 0.65
    cb._on_rollout_end()
    assert cb.current_level == 1, "win rate over threshold should advance the level"
    # Overrides must be pushed to the live env via env_method.
    assert env.calls and env.calls[-1][0] == "update_live_params"
    assert env.calls[-1][1][0] == {"battle_rules.cooldown_steps": 4}


def test_curriculum_default_metric_is_ep_rew_mean() -> None:
    cb = CurriculumCallback({"enabled": True}, env_cfg={})
    assert cb._metric == "rollout/ep_rew_mean"


def test_curriculum_warmup_suppresses_early_level_ups() -> None:
    """Below warmup_steps no level-up fires even with the metric over threshold.

    Guards the arena case where the league is empty (random opponent) early on,
    so an over-threshold win rate would otherwise advance against noise.
    """
    env = _FakeVecEnv()
    cur_cfg = {
        "enabled": True,
        "metric": "arena/agent_0_win_rate",
        "level_up_threshold": 0.6,
        "warmup_steps": 5000,
        "max_level": 2,
        "level_params": {1: {"battle_rules.cooldown_steps": 3}},
    }
    cb = CurriculumCallback(cur_cfg, env_cfg={}, verbose=0)
    logger = _StubLogger({"arena/agent_0_win_rate": 0.95})

    # Before warmup: metric well over threshold, but no level-up.
    _attach(cb, num_timesteps=4999, env=env, logger=logger)
    cb._on_rollout_end()
    assert cb.current_level == 0, "warmup must suppress level-ups"
    assert not env.calls, "no env_method push before warmup"

    # After warmup: the gate opens.
    _attach(cb, num_timesteps=5000, env=env, logger=logger)
    cb._on_rollout_end()
    assert cb.current_level == 1, "level-up should fire once past warmup"
