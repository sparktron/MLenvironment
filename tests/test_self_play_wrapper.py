"""Tests for the self-play env wrapper and disk-backed league sampler."""

from __future__ import annotations

import numpy as np

from rl_framework.envs.registry import make_env
from rl_framework.training.self_play_env_wrapper import (
    LeagueSampler,
    SelfPlayEnvWrapper,
    SingleAgentArenaEnv,
)


def _arena(**rules):
    cfg = {"type": "organism_arena_parallel", "seed": 0, "battle_rules": rules}
    return make_env("organism_arena_parallel", cfg)


class _StubSampler:
    """Sampler stub returning a fixed policy (or None for an empty league)."""

    def __init__(self, policy=None):
        self._policy = policy
        self.calls = 0

    def sample(self):
        self.calls += 1
        return self._policy


class _SpyPolicy:
    def __init__(self, action):
        self._action = np.asarray(action, dtype=np.float32)
        self.calls = 0

    def predict(self, obs, deterministic=True):
        self.calls += 1
        return self._action[np.newaxis], None


# -- SelfPlayEnvWrapper ------------------------------------------------------


def test_wrapper_exposes_only_the_live_agent() -> None:
    env = _arena(max_steps=20)
    wrapped = SelfPlayEnvWrapper(env, _StubSampler())
    assert wrapped.possible_agents == ["agent_0"]
    obs, infos = wrapped.reset()
    assert set(obs.keys()) == {"agent_0"}
    assert set(infos.keys()) == {"agent_0"}
    obs, rewards, terms, truncs, infos = wrapped.step(
        {"agent_0": np.zeros(3, dtype=np.float32)}
    )
    for d in (obs, rewards, terms, truncs, infos):
        assert set(d.keys()) == {"agent_0"}


def test_wrapper_samples_opponent_each_reset() -> None:
    env = _arena(max_steps=20)
    sampler = _StubSampler()
    wrapped = SelfPlayEnvWrapper(env, sampler)
    wrapped.reset()
    wrapped.reset()
    assert sampler.calls == 2


def test_wrapper_uses_random_action_when_league_empty() -> None:
    """Empty league (sampler returns None) must not crash; opponent acts randomly."""
    env = _arena(max_steps=20)
    wrapped = SelfPlayEnvWrapper(env, _StubSampler(policy=None))
    wrapped.reset()
    # Should run several steps without raising.
    for _ in range(5):
        wrapped.step({"agent_0": np.zeros(3, dtype=np.float32)})


def test_wrapper_drives_all_opponents_in_n_agent_arena() -> None:
    """In a 3-agent arena the wrapper exposes only agent_0 and drives both
    agent_1 and agent_2 from the frozen policy."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "num_agents": 3,
        "battle_rules": {"max_steps": 20, "damage": 0.0},
    }
    env = make_env("organism_arena_parallel", cfg)
    spy = _SpyPolicy([1.0, 0.0, 0.0])
    wrapped = SelfPlayEnvWrapper(env, _StubSampler())
    obs, _ = wrapped.reset()
    assert set(obs) == {"agent_0"}
    assert wrapped._opponent_agents == ["agent_1", "agent_2"]
    wrapped._frozen_policy = spy
    a1_x = env.state["agent_1"]["pos"][0]
    a2_x = env.state["agent_2"]["pos"][0]
    wrapped.step({"agent_0": np.zeros(3, dtype=np.float32)})
    # Frozen policy queried once per opponent; both opponents moved +x.
    assert spy.calls == 2
    assert env.state["agent_1"]["pos"][0] > a1_x
    assert env.state["agent_2"]["pos"][0] > a2_x


def test_wrapper_routes_opponent_through_frozen_policy() -> None:
    """agent_1's action must come from the frozen policy's predict()."""
    env = _arena(max_steps=20, damage=0.0)
    spy = _SpyPolicy([1.0, 0.0, 0.0])
    wrapped = SelfPlayEnvWrapper(env, _StubSampler())
    wrapped.reset()
    wrapped._frozen_policy = spy  # inject post-reset
    a1_before = env.state["agent_1"]["pos"][0]
    wrapped.step({"agent_0": np.zeros(3, dtype=np.float32)})
    assert spy.calls == 1, "frozen policy should drive the opponent each step"
    # The spy commands +x movement, so agent_1 should have moved right.
    assert env.state["agent_1"]["pos"][0] > a1_before


def test_wrapper_surfaces_terminal_when_opponent_knocked_out() -> None:
    """When the frozen opponent dies, the live agent must see a terminal."""
    env = _arena(max_steps=50, damage=10.0, attack_range=5.0, cooldown_steps=0)
    wrapped = SelfPlayEnvWrapper(env, _StubSampler())
    wrapped.reset()
    # Frozen opponent stands still (no attack); set it to near-zero health.
    wrapped._frozen_policy = _SpyPolicy([0.0, 0.0, 0.0])
    env.state["agent_1"]["health"] = 0.001
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    _, rewards, terms, truncs, _ = wrapped.step(
        {"agent_0": np.array([0.0, 0.0, 1.0], dtype=np.float32)}
    )
    assert terms["agent_0"] is True, "live agent should see terminal on opponent KO"
    assert rewards["agent_0"] > 0, "live agent should be rewarded for the KO"


# -- SingleAgentArenaEnv -----------------------------------------------------


def _single_agent_env(**rules):
    arena = _arena(**rules)
    return SingleAgentArenaEnv(SelfPlayEnvWrapper(arena, _StubSampler()))


def test_single_agent_env_gym_contract() -> None:
    """Adapter presents the live agent as a flat single-agent Gymnasium env."""
    env = _single_agent_env(max_steps=20)
    assert env.observation_space.shape == (8,)
    assert env.action_space.shape == (3,)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)
    obs, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_single_agent_env_surfaces_episode_outcome() -> None:
    """The terminal info must carry episode_outcome so ArenaMetricsCallback and
    the Monitor can read it on the native vec-env path."""
    env = _single_agent_env(max_steps=3, damage=0.0)
    env.reset(seed=0)
    info = {}
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        if terminated or truncated:
            break
    assert truncated, "episode should truncate at max_steps"
    assert info.get("episode_outcome", {}).get("outcome") == "timeout"


def test_single_agent_env_update_live_params_forwards_to_arena() -> None:
    """SB3 env_method('update_live_params', ...) must reach the arena env."""
    env = _single_agent_env(max_steps=20)
    arena = env._env.unwrapped
    env.update_live_params({"reward.damage_scale": 0.25})
    assert arena._damage_scale == 0.25


# -- LeagueSampler -----------------------------------------------------------


def test_league_sampler_empty_returns_none(tmp_path) -> None:
    sampler = LeagueSampler(tmp_path / "league", seed=0)  # dir does not exist
    assert sampler.sample() is None


def test_league_sampler_loads_and_caches_from_disk(monkeypatch, tmp_path) -> None:
    for ts in (100, 200, 300):
        (tmp_path / f"selfplay_{ts}.zip").write_text("x", encoding="utf-8")

    loads: list[str] = []

    class _FakeModel:
        pass

    def _fake_load(path):
        loads.append(str(path))
        return _FakeModel()

    monkeypatch.setattr(
        "rl_framework.training.self_play_env_wrapper.PPO.load", _fake_load
    )
    sampler = LeagueSampler(tmp_path, seed=0)
    first = sampler.sample()
    assert first is not None
    # Sampling the same snapshot again must hit the cache, not reload.
    for _ in range(20):
        sampler.sample()
    assert len(loads) <= 3, "each distinct snapshot loaded at most once"


def test_league_sampler_skips_non_numeric_snapshots(tmp_path) -> None:
    (tmp_path / "selfplay_100.zip").write_text("x", encoding="utf-8")
    (tmp_path / "selfplay_best.zip").write_text("x", encoding="utf-8")
    sampler = LeagueSampler(tmp_path, seed=0)
    files = sampler._league_files()
    assert [p.name for p in files] == ["selfplay_100.zip"]


def test_league_sampler_recent_bias_prefers_latest(monkeypatch, tmp_path) -> None:
    for ts in (100, 200, 300):
        (tmp_path / f"selfplay_{ts}.zip").write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "rl_framework.training.self_play_env_wrapper.PPO.load", lambda p: str(p)
    )
    sampler = LeagueSampler(
        tmp_path, sampling_mode="recent_bias", recent_bias_alpha=3.0, seed=0
    )
    # sample() wraps the loaded model in a FrozenPolicy; inspect the model
    # (here a path string, via the patched PPO.load).
    samples = [sampler.sample()._model for _ in range(300)]
    latest = sum(1 for s in samples if "selfplay_300" in s)
    earliest = sum(1 for s in samples if "selfplay_100" in s)
    assert latest > earliest


def test_league_sampler_caches_file_list_until_dir_changes(monkeypatch, tmp_path):
    """R2c: the snapshot dir is re-globbed only when its mtime changes, not on
    every sample/reset."""
    import os
    import pathlib

    (tmp_path / "selfplay_100.zip").write_text("x", encoding="utf-8")
    globs = {"n": 0}
    orig_glob = pathlib.Path.glob

    def _counting_glob(self, pattern):
        globs["n"] += 1
        return orig_glob(self, pattern)

    monkeypatch.setattr(pathlib.Path, "glob", _counting_glob)
    sampler = LeagueSampler(tmp_path, seed=0)

    sampler._league_files()
    sampler._league_files()
    sampler._league_files()
    assert globs["n"] == 1, "unchanged dir must not be re-globbed every call"

    # Adding a snapshot bumps the dir mtime; force a distinct mtime so the test
    # does not depend on filesystem timestamp granularity.
    (tmp_path / "selfplay_200.zip").write_text("x", encoding="utf-8")
    st = (tmp_path).stat()
    os.utime(tmp_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    files = sampler._league_files()
    assert globs["n"] == 2, "a changed dir must trigger exactly one re-glob"
    assert [p.name for p in files] == ["selfplay_100.zip", "selfplay_200.zip"]


def test_league_sampler_prewarms_newest_with_sidecar(monkeypatch, tmp_path):
    """R2c: when the newest snapshot's obs-normaliser sidecar is present, it is
    pre-warmed into the cache off the sampling path."""
    for ts in (100, 200):
        (tmp_path / f"selfplay_{ts}.zip").write_text("x", encoding="utf-8")
    # Only the newest gets a sidecar (gate for a fully-written snapshot).
    (tmp_path / "selfplay_200_vecnorm.pkl").write_text("x", encoding="utf-8")

    loaded: list[str] = []
    monkeypatch.setattr(
        "rl_framework.training.self_play_env_wrapper.PPO.load",
        lambda p: loaded.append(str(p)) or object(),
    )
    monkeypatch.setattr(
        "rl_framework.training.self_play_env_wrapper.load_obs_normalizer",
        lambda p: None,
    )
    sampler = LeagueSampler(tmp_path, seed=0)
    sampler._league_files()  # no explicit sample
    cached = {p.name for p in sampler._cache}
    assert cached == {"selfplay_200.zip"}, "only the newest+sidecar is pre-warmed"


def test_league_sampler_no_prewarm_without_sidecar(monkeypatch, tmp_path):
    """R2c: without a sidecar the newest is not eagerly loaded (avoids caching a
    snapshot mid-write); it still loads lazily on sample()."""
    (tmp_path / "selfplay_100.zip").write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        "rl_framework.training.self_play_env_wrapper.PPO.load", lambda p: object()
    )
    sampler = LeagueSampler(tmp_path, seed=0)
    sampler._league_files()
    assert sampler._cache == {}, "no sidecar -> no pre-warm"
    assert sampler.sample() is not None
    assert len(sampler._cache) == 1, "lazy load still populates the cache"


def test_league_sampler_drops_pruned_snapshot_from_cache(monkeypatch, tmp_path):
    """R2c: a snapshot removed from disk is dropped from the model cache on the
    next directory refresh."""
    import os

    (tmp_path / "selfplay_100.zip").write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        "rl_framework.training.self_play_env_wrapper.PPO.load", lambda p: object()
    )
    sampler = LeagueSampler(tmp_path, seed=0)
    sampler.sample()
    assert any(p.name == "selfplay_100.zip" for p in sampler._cache)

    # Prune the snapshot and add a new one; force a distinct dir mtime.
    (tmp_path / "selfplay_100.zip").unlink()
    (tmp_path / "selfplay_200.zip").write_text("x", encoding="utf-8")
    st = (tmp_path).stat()
    os.utime(tmp_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    sampler._league_files()
    assert not any(p.name == "selfplay_100.zip" for p in sampler._cache), (
        "pruned snapshot should be evicted from the cache"
    )
