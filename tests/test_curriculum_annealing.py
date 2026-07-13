"""Tests for reward annealing and the win-rate-gated curriculum (Feature 5)."""

from __future__ import annotations

from collections import defaultdict
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
    """Mirrors real SB3 Logger state at ``on_rollout_end``.

    ``name_to_value`` is a ``defaultdict(float)`` (a plain-index read of a
    missing key fabricates and stores 0.0), and ``rollout/*`` keys are never
    present there — SB3 records and immediately dumps them *after* the
    callbacks run. Only pre-populate keys a callback genuinely records before
    the next dump (e.g. ``arena/*`` from ArenaMetricsCallback).
    """

    def __init__(self, values=None) -> None:
        self.name_to_value = defaultdict(float, values or {})
        self.records: dict[str, float] = {}

    def record(self, key, value) -> None:
        self.records[key] = value


def _attach(cb, *, num_timesteps=0, env=None, logger=None, ep_info_buffer=None):
    cb.model = SimpleNamespace(
        get_env=lambda: env,
        logger=logger,
        ep_info_buffer=ep_info_buffer if ep_info_buffer is not None else [],
    )
    cb.num_timesteps = num_timesteps


# -- RewardAnnealingCallback -------------------------------------------------


def test_reward_annealing_validates_anneal_steps() -> None:
    with pytest.raises(ValueError, match="anneal_steps must be > 0"):
        RewardAnnealingCallback(anneal_steps=0)


def test_reward_annealing_pushes_decreasing_scale() -> None:
    env = _FakeVecEnv()
    cb = RewardAnnealingCallback(anneal_steps=1000)

    _attach(cb, num_timesteps=0, env=env)
    cb._on_rollout_end()
    _attach(cb, num_timesteps=250, env=env)
    cb._on_rollout_end()
    _attach(cb, num_timesteps=1000, env=env)
    cb._on_rollout_end()

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
    cb._on_rollout_end()
    _attach(cb, num_timesteps=300, env=env)
    cb._on_rollout_end()
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


def test_curriculum_default_metric_reads_ep_info_buffer() -> None:
    """``rollout/ep_rew_mean`` is never in the logger at ``on_rollout_end``
    (SB3 records and clears it afterwards), so the curriculum must read
    episode stats from the model's ``ep_info_buffer`` — the walker curriculum
    was silently dead without this."""
    env = _FakeVecEnv()
    cur_cfg = {
        "enabled": True,
        "level_up_threshold": 150.0,
        "max_level": 2,
        "level_params": {1: {"reward.target_velocity": 1.5}},
    }
    cb = CurriculumCallback(cur_cfg, env_cfg={}, verbose=0)
    logger = _StubLogger()  # empty defaultdict: the real state at rollout end

    _attach(cb, env=env, logger=logger, ep_info_buffer=[{"r": 120.0, "l": 400}])
    cb._on_rollout_end()
    assert cb.current_level == 0, "below threshold: no level up"

    _attach(
        cb,
        env=env,
        logger=logger,
        ep_info_buffer=[{"r": 180.0, "l": 400}, {"r": 160.0, "l": 380}],
    )
    cb._on_rollout_end()
    assert cb.current_level == 1, "ep_info_buffer mean over threshold must level up"
    assert env.calls and env.calls[-1][0] == "update_live_params"


def test_curriculum_missing_metric_neither_levels_nor_pollutes_logger() -> None:
    """An absent metric must read as 'no data yet'. The logger map is a
    defaultdict, so a plain-index read would fabricate 0.0 (instantly leveling
    a threshold-0 gate) and insert the key into the next TensorBoard dump."""
    env = _FakeVecEnv()
    cur_cfg = {
        "enabled": True,
        "metric": "arena/agent_0_win_rate",
        "level_up_threshold": 0.0,
        "max_level": 2,
        "level_params": {1: {"battle_rules.damage": 0.1}},
    }
    cb = CurriculumCallback(cur_cfg, env_cfg={}, verbose=0)
    logger = _StubLogger()
    _attach(cb, env=env, logger=logger)

    cb._on_rollout_end()

    assert cb.current_level == 0, "missing metric must not level up"
    assert "arena/agent_0_win_rate" not in logger.name_to_value, (
        "probing the metric must not insert it into the logger map"
    )
    assert not env.calls


def test_curriculum_empty_episode_buffer_does_not_level() -> None:
    """No completed episodes yet -> no metric -> no level-up, even at a
    threshold of 0.0 (regression for the defaultdict-0.0 read)."""
    env = _FakeVecEnv()
    cur_cfg = {
        "enabled": True,
        "level_up_threshold": 0.0,
        "max_level": 2,
        "level_params": {1: {"reward.target_velocity": 1.5}},
    }
    cb = CurriculumCallback(cur_cfg, env_cfg={}, verbose=0)
    _attach(cb, env=env, logger=_StubLogger(), ep_info_buffer=[])
    cb._on_rollout_end()
    assert cb.current_level == 0


def test_arena_callbacks_order_metrics_before_curriculum(tmp_path, monkeypatch):
    """CurriculumCallback reads arena/* win rates that ArenaMetricsCallback
    records in its own _on_rollout_end, so the metrics callback must come
    first in the callback list train() assembles."""
    from rl_framework.training import sb3_runner

    captured = {}

    def _fake_learn(self, total_timesteps, callback=None, **kwargs):
        captured["callbacks"] = callback
        return self

    monkeypatch.setattr(sb3_runner.PPO, "learn", _fake_learn)

    cfg = {
        "experiment_name": "order_guard",
        "seed": 0,
        "output": {"base_dir": str(tmp_path)},
        "environment": {
            "type": "organism_arena_parallel",
            "battle_rules": {"max_steps": 20},
        },
        "training": {
            "total_timesteps": 64,
            "n_steps": 32,
            "batch_size": 32,
            "num_envs": 1,
            "device": "cpu",
        },
        "curriculum": {
            "enabled": True,
            "metric": "arena/agent_0_win_rate",
            "level_up_threshold": 0.6,
            "level_params": {1: {"battle_rules.damage": 0.1}},
        },
    }
    sb3_runner.train(cfg)

    callbacks = captured["callbacks"]
    metrics_idx = next(
        i
        for i, cb in enumerate(callbacks)
        if isinstance(cb, sb3_runner.ArenaMetricsCallback)
    )
    curriculum_idx = next(
        i for i, cb in enumerate(callbacks) if isinstance(cb, CurriculumCallback)
    )
    assert metrics_idx < curriculum_idx


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
