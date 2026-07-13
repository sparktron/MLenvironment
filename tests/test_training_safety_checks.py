from collections import deque
from pathlib import Path

import pytest

from rl_framework.training.self_play_callback import SelfPlayCallback
from rl_framework.utils.config_merge import set_nested


def test_shared_policy_arena_rejects_multiple_envs(tmp_path) -> None:
    """Shared-policy arena (no self-play) runs single-process via SuperSuit;
    num_envs > 1 must fail loudly.

    Without self-play the arena needs SuperSuit's multi-agent vec conversion,
    and num_envs > 1 forks it into subprocesses, which silently disables the
    live env_method updates (reward annealing, curriculum) and is unstable in
    SuperSuit 3.10. The guard fires before any env is built. (The self-play
    path bypasses SuperSuit and parallelizes — see the integration tests.)
    """
    from rl_framework.training.sb3_runner import train

    cfg = {
        "experiment_name": "arena_guard_test",
        "seed": 0,
        "environment": {"type": "organism_arena_parallel"},
        "training": {"total_timesteps": 1, "num_envs": 2},
        "output": {"base_dir": str(tmp_path)},
    }
    with pytest.raises(ValueError, match="num_envs == 1"):
        train(cfg)


def test_callback_frequency_scales_with_vector_env_count() -> None:
    from rl_framework.training.sb3_runner import _callback_freq_from_timesteps

    assert _callback_freq_from_timesteps(50_000, 1) == 50_000
    assert _callback_freq_from_timesteps(50_000, 8) == 6_250
    assert _callback_freq_from_timesteps(50_000, 24) == 2_083
    assert _callback_freq_from_timesteps(4, 24) == 1


def test_runtime_controls_apply_torch_threads_and_worker_start_method(monkeypatch) -> None:
    from rl_framework.training import sb3_runner

    configured_threads: list[int] = []
    monkeypatch.setattr("torch.set_num_threads", configured_threads.append)
    sb3_runner._configure_torch_num_threads({"torch_num_threads": 3})
    assert configured_threads == [3]

    captured = {}

    class _FakeVecEnv:
        pass

    def _fake_subproc(env_fns, start_method=None):
        captured["env_fns"] = env_fns
        captured["start_method"] = start_method
        return _FakeVecEnv()

    monkeypatch.setattr(sb3_runner, "SubprocVecEnv", _fake_subproc)
    env_fns = [object(), object()]
    result = sb3_runner._make_subproc_vec_env(
        env_fns, {"worker_start_method": "spawn"}
    )
    assert isinstance(result, _FakeVecEnv)
    assert captured == {"env_fns": env_fns, "start_method": "spawn"}


def test_set_nested_requires_existing_leaf_key() -> None:
    cfg = {"training": {"learning_rate": 3e-4}}
    with pytest.raises(KeyError, match="leaf key 'missing_key' not found"):
        set_nested(cfg, "training.missing_key", 1)


def test_self_play_callback_validates_positive_frequencies(tmp_path) -> None:
    with pytest.raises(ValueError, match="snapshot_freq must be > 0"):
        SelfPlayCallback(snapshot_dir=tmp_path, snapshot_freq=0, max_league_size=1)

    with pytest.raises(ValueError, match="max_league_size must be > 0"):
        SelfPlayCallback(snapshot_dir=tmp_path, snapshot_freq=1, max_league_size=0)

    with pytest.raises(ValueError, match="sampling_mode"):
        SelfPlayCallback(
            snapshot_dir=tmp_path,
            snapshot_freq=1,
            max_league_size=1,
            sampling_mode="invalid",
        )

    with pytest.raises(ValueError, match="recent_bias_alpha must be > 0"):
        SelfPlayCallback(
            snapshot_dir=tmp_path,
            snapshot_freq=1,
            max_league_size=1,
            recent_bias_alpha=0.0,
        )


def test_self_play_sample_opponent_uses_cache(monkeypatch, tmp_path) -> None:
    callback = SelfPlayCallback(
        snapshot_dir=tmp_path, snapshot_freq=1, max_league_size=3
    )
    callback._league = deque([tmp_path / "a"])  # noqa: SLF001

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


def test_self_play_recent_bias_sampling_prefers_latest_entry(
    monkeypatch, tmp_path
) -> None:
    callback = SelfPlayCallback(
        snapshot_dir=tmp_path,
        snapshot_freq=1,
        max_league_size=3,
        sampling_mode="recent_bias",
        recent_bias_alpha=2.0,
    )
    callback._league = deque([tmp_path / "a", tmp_path / "b", tmp_path / "c"])  # noqa: SLF001

    monkeypatch.setattr(
        "rl_framework.training.self_play_callback.PPO.load", lambda path: str(path)
    )
    samples = [callback.sample_opponent() for _ in range(300)]
    latest_count = sum(1 for s in samples if str((tmp_path / "c")) in s)
    earliest_count = sum(1 for s in samples if str((tmp_path / "a")) in s)
    assert latest_count > earliest_count


class _FakeModel:
    """SB3-faithful save stub: ``.zip`` is appended only to suffix-less paths
    (see ``save_util.open_path_pathlib``); suffixed paths are written as-is."""

    def __init__(self, vec_normalize=None) -> None:
        self._vec_normalize = vec_normalize

    def save(self, path: str) -> None:
        p = Path(path)
        if p.suffix == "":
            p = Path(str(p) + ".zip")
        p.write_text("model", encoding="utf-8")

    def get_vec_normalize_env(self):
        return self._vec_normalize


def test_self_play_pruning_also_prunes_cache(tmp_path) -> None:
    callback = SelfPlayCallback(
        snapshot_dir=tmp_path, snapshot_freq=1, max_league_size=1
    )
    old = tmp_path / "selfplay_1"
    callback._league = deque([old])  # noqa: SLF001
    callback._model_cache = {old: object()}  # noqa: SLF001

    callback.model = _FakeModel()  # type: ignore[assignment]
    callback.num_timesteps = 2
    callback._save_snapshot()  # noqa: SLF001

    assert old not in callback._model_cache  # noqa: SLF001


def test_self_play_snapshot_cadence_with_vector_env_stride(tmp_path) -> None:
    """num_timesteps advances by num_envs per callback call, so an exact
    modulo check snapshots on the LCM cadence (freq=5000 at num_envs=24 used
    to fire every 15000 steps). One snapshot must land per freq window,
    within one stride of the boundary."""
    freq = 5000
    for stride in (1, 7, 24):
        league_dir = tmp_path / f"stride_{stride}"
        callback = SelfPlayCallback(
            snapshot_dir=league_dir, snapshot_freq=freq, max_league_size=100
        )
        callback.model = _FakeModel()  # type: ignore[assignment]
        for ts in range(stride, 30000 + stride, stride):
            callback.num_timesteps = ts
            callback._on_step()  # noqa: SLF001

        snaps = sorted(
            int(p.stem.rsplit("_", 1)[-1]) for p in league_dir.glob("selfplay_*.zip")
        )
        assert len(snaps) == 6, (
            f"stride={stride}: expected one snapshot per {freq}-step window, "
            f"got {snaps}"
        )
        for i, ts in enumerate(snaps, start=1):
            assert freq * i <= ts < freq * i + stride, (
                f"stride={stride}: snapshot {ts} outside window {i}"
            )


def test_self_play_snapshot_zip_visible_only_after_sidecar(tmp_path) -> None:
    """Parallel workers glob selfplay_*.zip as soon as the league dir changes,
    so the zip must appear atomically and only after its obs-normaliser
    sidecar is complete — else an opponent gets cached without normalisation
    (or PPO.load reads a half-written archive)."""
    zips_seen_at_sidecar_write: list[list[str]] = []

    class _FakeVecNormalize:
        def save(self, path: str) -> None:
            zips_seen_at_sidecar_write.append(
                sorted(p.name for p in tmp_path.glob("selfplay_*.zip"))
            )
            Path(path).write_text("norm", encoding="utf-8")

    callback = SelfPlayCallback(
        snapshot_dir=tmp_path, snapshot_freq=1, max_league_size=5
    )
    callback.model = _FakeModel(vec_normalize=_FakeVecNormalize())  # type: ignore[assignment]
    callback.num_timesteps = 100
    callback._save_snapshot()  # noqa: SLF001

    assert zips_seen_at_sidecar_write == [[]], (
        "the sidecar must be fully written before the snapshot zip is visible"
    )
    assert (tmp_path / "selfplay_100.zip").exists()
    assert (tmp_path / "selfplay_100_vecnorm.pkl").exists()
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp" in p.name]
    assert leftovers == [], f"temp files left behind: {leftovers}"
