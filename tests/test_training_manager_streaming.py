from __future__ import annotations

import threading

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
