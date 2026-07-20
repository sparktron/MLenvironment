from __future__ import annotations

import threading
from typing import Any

from rl_framework.gui.training_manager import TrainingManager, _RunState


def test_tuning_events_are_merged_atomically() -> None:
    manager = TrainingManager()
    run_id = "run_x"
    state = _RunState(
        run_id=run_id,
        cfg={"experiment_name": "exp"},
        status="running",
        stop_event=threading.Event(),
    )
    manager._runs[run_id] = state

    manager.apply_tuning(run_id, {"learning_rate": 3e-4})
    manager.apply_tuning(run_id, {"learning_rate": 1e-4, "reward.target_velocity": 2.0})

    merged = manager._pop_tuning_event(run_id)
    assert merged == {"learning_rate": 1e-4, "reward.target_velocity": 2.0}
    assert manager._pop_tuning_event(run_id) is None


def test_status_returns_latest_streamed_metrics() -> None:
    manager = TrainingManager()
    run_id = "run_y"
    state = _RunState(
        run_id=run_id,
        cfg={"experiment_name": "exp"},
        status="running",
        stop_event=threading.Event(),
    )
    manager._runs[run_id] = state

    payload = {"timesteps": 1024, "rollout/ep_rew_mean": 12.5}
    manager._publish_status(run_id, payload)

    status = manager.get_status(run_id)
    assert status["metrics"] == payload


def test_gui_training_selects_only_worker_zero_for_rendering(monkeypatch) -> None:
    from rl_framework.gui import training_manager as manager_module
    from rl_framework.training import sb3_runner

    captured: dict[str, Any] = {}

    def _fake_train(cfg, **kwargs):
        captured["cfg"] = cfg
        captured["kwargs"] = kwargs
        return "/tmp/model.zip"

    class _FakeRegistry:
        def update_run(self, *_args, **_kwargs) -> None:
            pass

        def record_event(self, *_args, **_kwargs) -> None:
            pass

    monkeypatch.setattr(sb3_runner, "train", _fake_train)
    monkeypatch.setattr(
        manager_module, "registry_for_config", lambda _cfg: _FakeRegistry()
    )

    manager = TrainingManager()
    state = _RunState(
        run_id="render_worker",
        cfg={
            "experiment_name": "exp",
            "environment": {"type": "walker_bullet"},
        },
        status="running",
    )

    manager._train_worker(state)

    assert captured["cfg"]["environment"]["render_mode"] == "rgb_array"
    assert captured["kwargs"]["render_env_index"] == 0
    assert state.status == "completed"


# ---------------------------------------------------------------------------
# Concurrency / thread-safety tests (plan §5d)
# ---------------------------------------------------------------------------

def _make_running_state(run_id: str) -> tuple[TrainingManager, _RunState]:
    manager = TrainingManager()
    state = _RunState(
        run_id=run_id,
        cfg={"experiment_name": "exp"},
        status="running",
        stop_event=threading.Event(),
    )
    manager._runs[run_id] = state
    return manager, state


def test_concurrent_apply_tuning_no_corruption() -> None:
    """50 threads each calling apply_tuning must not corrupt the event list.

    After all threads have submitted, _pop_tuning_event must return a
    dict (or None) — never raise and never leave the list in an
    inconsistent state.
    """
    manager, _ = _make_running_state("run_conc")
    run_id = "run_conc"

    errors: list[Exception] = []
    barrier = threading.Barrier(50)

    def _submit(idx: int) -> None:
        try:
            barrier.wait()  # all threads hit apply_tuning at the same time
            manager.apply_tuning(run_id, {f"param_{idx}": float(idx)})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_submit, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors, f"Exceptions during concurrent apply_tuning: {errors}"

    merged: dict[str, Any] | None = manager._pop_tuning_event(run_id)
    # The merged dict must contain exactly the keys that were applied.
    assert merged is not None
    # All 50 keys must be present (last-write-wins for same key, all keys unique here).
    assert len(merged) == 50
    assert manager._pop_tuning_event(run_id) is None  # queue drained


def test_concurrent_publish_status_no_corruption() -> None:
    """50 threads calling _publish_status concurrently must not corrupt latest_metrics."""
    manager, _ = _make_running_state("run_pub")
    run_id = "run_pub"

    errors: list[Exception] = []
    barrier = threading.Barrier(50)

    def _publish(idx: int) -> None:
        try:
            barrier.wait()
            manager._publish_status(run_id, {"timesteps": idx, "reward": float(idx)})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_publish, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors, f"Exceptions during concurrent _publish_status: {errors}"

    status = manager.get_status(run_id)
    metrics = status["metrics"]
    # Metrics must be a valid dict with the expected keys — not corrupted.
    assert isinstance(metrics, dict)
    assert "timesteps" in metrics
    assert "reward" in metrics
