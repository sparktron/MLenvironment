from pathlib import Path

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


def test_self_play_recent_bias_sampling_prefers_latest_entry(monkeypatch, tmp_path) -> None:
    callback = SelfPlayCallback(
        snapshot_dir=tmp_path,
        snapshot_freq=1,
        max_league_size=3,
        sampling_mode="recent_bias",
        recent_bias_alpha=2.0,
    )
    callback._league = [tmp_path / "a", tmp_path / "b", tmp_path / "c"]  # noqa: SLF001

    monkeypatch.setattr("rl_framework.training.self_play_callback.PPO.load", lambda path: str(path))
    samples = [callback.sample_opponent() for _ in range(300)]
    latest_count = sum(1 for s in samples if str((tmp_path / "c")) in s)
    earliest_count = sum(1 for s in samples if str((tmp_path / "a")) in s)
    assert latest_count > earliest_count


def test_self_play_pruning_also_prunes_cache(tmp_path) -> None:
    callback = SelfPlayCallback(snapshot_dir=tmp_path, snapshot_freq=1, max_league_size=1)
    old = tmp_path / "selfplay_1"
    callback._league = [old]  # noqa: SLF001
    callback._model_cache = {old: object()}  # noqa: SLF001

    class _FakeModel:
        def save(self, path: str) -> None:
            (tmp_path / f"{Path(path).name}.zip").write_text("x", encoding="utf-8")

    callback.model = _FakeModel()  # type: ignore[assignment]
    callback.num_timesteps = 2
    callback._save_snapshot()  # noqa: SLF001

    assert old not in callback._model_cache  # noqa: SLF001
