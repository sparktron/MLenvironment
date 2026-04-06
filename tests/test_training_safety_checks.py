import pytest

from rl_framework.training.self_play_callback import SelfPlayCallback
from rl_framework.training.sweep import _set_nested


def test_sweep_set_nested_requires_existing_leaf_key() -> None:
    cfg = {"training": {"learning_rate": 3e-4}}
    with pytest.raises(KeyError, match="leaf key 'missing_key' not found"):
        _set_nested(cfg, "training.missing_key", 1)


def test_self_play_callback_validates_positive_frequencies(tmp_path) -> None:
    with pytest.raises(ValueError, match="snapshot_freq must be > 0"):
        SelfPlayCallback(snapshot_dir=tmp_path, snapshot_freq=0, max_league_size=1)

    with pytest.raises(ValueError, match="max_league_size must be > 0"):
        SelfPlayCallback(snapshot_dir=tmp_path, snapshot_freq=1, max_league_size=0)

    with pytest.raises(ValueError, match="sampling_mode"):
        SelfPlayCallback(snapshot_dir=tmp_path, snapshot_freq=1, max_league_size=1, sampling_mode="invalid")

    with pytest.raises(ValueError, match="recent_bias_alpha must be > 0"):
        SelfPlayCallback(snapshot_dir=tmp_path, snapshot_freq=1, max_league_size=1, recent_bias_alpha=0.0)


def test_self_play_sample_opponent_uses_cache(monkeypatch, tmp_path) -> None:
    callback = SelfPlayCallback(snapshot_dir=tmp_path, snapshot_freq=1, max_league_size=3)
    callback._league = [tmp_path / "a"]  # noqa: SLF001

    calls = {"count": 0}

    class _FakeModel:
        pass

    def _fake_load(_path: str):
        calls["count"] += 1
        return _FakeModel()

    monkeypatch.setattr("rl_framework.training.self_play_callback.PPO.load", _fake_load)
    first = callback.sample_opponent()
    second = callback.sample_opponent()
    assert first is second
    assert calls["count"] == 1
