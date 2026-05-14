"""Unit tests for FrameCaptureCallback.

Exercises the ring buffer, since-filtering, thread-safety, episode counting,
and render fallback — without spinning up a real SB3 training loop.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import numpy as np

from rl_framework.training.frame_capture_callback import FrameCaptureCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_frame(h: int = 4, w: int = 4) -> np.ndarray:
    """Return a tiny uint8 HxWx3 array usable as a fake rendered frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_rgba_frame(h: int = 4, w: int = 4) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


_SENTINEL = object()


def _stub_callback(
    capture_interval: int = 1,
    max_frames: int = 10,
    render_return=_SENTINEL,
) -> FrameCaptureCallback:
    """Return a callback with model/env stubs attached so _on_step can run.

    training_env is a read-only property on BaseCallback that delegates to
    self.model.get_env(), so we stub through the model rather than setting
    the property directly.

    Pass render_return=None explicitly to simulate an env that returns no frame.
    Omit it (default) to get a tiny RGB frame.
    """
    cb = FrameCaptureCallback(capture_interval=capture_interval, max_frames=max_frames)

    if render_return is _SENTINEL:
        render_return = _make_rgb_frame()

    # VecEnv stub
    env = MagicMock()
    env.render.return_value = render_return

    # Model stub: num_timesteps is read in _on_step; get_env() powers training_env
    model = MagicMock()
    model.num_timesteps = 0
    model.get_env.return_value = env
    cb.model = model

    # SB3 sets self.locals on each step; provide a default
    cb.locals = {"dones": [False]}

    return cb


# ---------------------------------------------------------------------------
# Basic frame capture
# ---------------------------------------------------------------------------

def test_first_frame_captured_at_interval() -> None:
    cb = _stub_callback(capture_interval=50)
    cb.model.num_timesteps = 49
    cb._on_step()
    assert len(cb.frames) == 0  # not yet

    cb.model.num_timesteps = 50
    cb._on_step()
    assert len(cb.frames) == 1
    assert cb.frames[0].timestep == 50


def test_frame_metadata_populated() -> None:
    cb = _stub_callback(capture_interval=1)
    cb.model.num_timesteps = 1
    cb._on_step()

    f = cb.frames[0]
    assert f.frame_index == 0
    assert f.timestep == 1
    assert isinstance(f.image_base64, str) and len(f.image_base64) > 0


def test_frame_index_monotonically_increases() -> None:
    cb = _stub_callback(capture_interval=1, max_frames=10)
    for step in range(1, 6):
        cb.model.num_timesteps = step
        cb._on_step()

    indices = [f.frame_index for f in cb.frames]
    assert indices == list(range(5))


# ---------------------------------------------------------------------------
# Ring buffer (maxlen)
# ---------------------------------------------------------------------------

def test_ring_buffer_drops_oldest_when_full() -> None:
    cb = _stub_callback(capture_interval=1, max_frames=3)
    for step in range(1, 6):
        cb.model.num_timesteps = step
        cb._on_step()

    assert len(cb.frames) == 3
    # Oldest three captured are steps 3, 4, 5
    assert cb.frames[0].timestep == 3
    assert cb.frames[-1].timestep == 5


# ---------------------------------------------------------------------------
# since filtering
# ---------------------------------------------------------------------------

def test_get_frames_since_zero_returns_all() -> None:
    cb = _stub_callback(capture_interval=1, max_frames=10)
    for step in range(1, 4):
        cb.model.num_timesteps = step
        cb._on_step()

    frames = cb.get_frames(since=0)
    assert len(frames) == 3


def test_get_frames_since_filters_correctly() -> None:
    cb = _stub_callback(capture_interval=1, max_frames=10)
    for step in range(1, 6):
        cb.model.num_timesteps = step
        cb._on_step()

    frames = cb.get_frames(since=3)
    assert all(f["frame_index"] >= 3 for f in frames)
    assert len(frames) == 2  # indices 3 and 4


def test_get_frames_since_beyond_end_returns_empty() -> None:
    cb = _stub_callback(capture_interval=1, max_frames=10)
    cb.model.num_timesteps = 1
    cb._on_step()

    assert cb.get_frames(since=99) == []


def test_get_frames_returns_dicts_with_required_keys() -> None:
    cb = _stub_callback(capture_interval=1)
    cb.model.num_timesteps = 1
    cb._on_step()

    frame = cb.get_frames()[0]
    assert {"frame_index", "timestep", "episode_num", "image_base64"} <= frame.keys()


# ---------------------------------------------------------------------------
# episode_num tracking
# ---------------------------------------------------------------------------

def test_episode_num_increments_on_done() -> None:
    cb = _stub_callback(capture_interval=1, max_frames=10)

    cb.locals = {"dones": [False]}
    cb.model.num_timesteps = 1
    cb._on_step()

    cb.locals = {"dones": [True]}
    cb.model.num_timesteps = 2
    cb._on_step()

    cb.locals = {"dones": [True]}
    cb.model.num_timesteps = 3
    cb._on_step()

    # Two episodes completed before the third step's capture
    assert cb.frames[2].episode_num == 2


def test_episode_num_handles_vectorised_envs() -> None:
    cb = _stub_callback(capture_interval=1, max_frames=10)

    # Two parallel envs both done simultaneously
    cb.locals = {"dones": [True, True]}
    cb.model.num_timesteps = 1
    cb._on_step()

    assert cb.episode_num == 2


def test_episode_num_absent_dones_does_not_raise() -> None:
    cb = _stub_callback(capture_interval=1)
    cb.locals = {}  # no "dones" key
    cb.model.num_timesteps = 1
    cb._on_step()  # should not raise


# ---------------------------------------------------------------------------
# RGBA → RGB conversion
# ---------------------------------------------------------------------------

def test_rgba_frame_stripped_to_rgb() -> None:
    cb = _stub_callback(capture_interval=1, render_return=_make_rgba_frame())
    cb.model.num_timesteps = 1
    cb._on_step()
    assert len(cb.frames) == 1  # encode succeeded; alpha channel dropped


# ---------------------------------------------------------------------------
# Render fallback: VecEnv with .envs attribute
# ---------------------------------------------------------------------------

def test_fallback_to_sub_env_render() -> None:
    # Primary render() raises AttributeError; .envs[0].render() works
    sub_env = MagicMock()
    sub_env.render.return_value = _make_rgb_frame()
    primary_env = MagicMock()
    primary_env.render.side_effect = AttributeError("no render")
    primary_env.envs = [sub_env]

    cb = _stub_callback(capture_interval=1)
    cb.model.get_env.return_value = primary_env

    cb.model.num_timesteps = 1
    cb._on_step()
    assert len(cb.frames) == 1


def test_none_render_result_skipped() -> None:
    cb = _stub_callback(capture_interval=1, render_return=None)
    cb.model.num_timesteps = 1
    cb._on_step()
    assert len(cb.frames) == 0


# ---------------------------------------------------------------------------
# Thread-safety: concurrent write/read must not corrupt snapshot
# ---------------------------------------------------------------------------

def test_concurrent_write_and_read_does_not_raise() -> None:
    cb = _stub_callback(capture_interval=1, max_frames=50)
    errors: list[Exception] = []

    def _writer():
        for step in range(1, 30):
            cb.model.num_timesteps = step
            try:
                cb._on_step()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            time.sleep(0)

    def _reader():
        for _ in range(30):
            try:
                cb.get_frames(since=0)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            time.sleep(0)

    t_w = threading.Thread(target=_writer)
    t_r = threading.Thread(target=_reader)
    t_w.start()
    t_r.start()
    t_w.join()
    t_r.join()

    assert errors == [], f"Concurrent access raised: {errors}"
