from __future__ import annotations

import numpy as np
import pytest

from rl_framework.envs.locomotion.gait import WalkerGaitTracker


def _update(
    tracker: WalkerGaitTracker,
    *,
    step: int,
    contacts: tuple[bool, bool],
    x: float,
    slips: tuple[float, float] = (0.0, 0.0),
):
    return tracker.update(
        step=step,
        contacts=contacts,
        pelvis_x=x,
        foot_slip_speeds=slips,
        applied_action=np.zeros(10, dtype=np.float32),
    )


def test_gait_tracker_counts_only_ordered_alternating_touchdowns() -> None:
    tracker = WalkerGaitTracker(touchdown_debounce_steps=2)
    tracker.reset(initial_contacts=(True, True), action_size=10)

    _update(tracker, step=1, contacts=(False, True), x=0.00)
    first = _update(tracker, step=2, contacts=(True, True), x=0.05)
    _update(tracker, step=3, contacts=(True, False), x=0.08)
    alternating = _update(tracker, step=4, contacts=(True, True), x=0.15)
    _update(tracker, step=5, contacts=(True, False), x=0.18)
    same_foot = _update(tracker, step=6, contacts=(True, True), x=0.25)

    assert first.valid_alternating_touchdown is False
    assert alternating.valid_alternating_touchdown is True
    assert alternating.alternating_step_progress == pytest.approx(0.10)
    assert same_foot.valid_alternating_touchdown is False
    metrics = tracker.episode_metrics()
    assert metrics["gait_right_touchdowns"] == 1.0
    assert metrics["gait_left_touchdowns"] == 2.0
    assert metrics["gait_alternating_touchdowns"] == 1.0
    assert metrics["gait_longest_same_foot_sequence"] == 2.0


def test_gait_tracker_debounces_contact_bounce_and_ignores_double_landing() -> None:
    tracker = WalkerGaitTracker(touchdown_debounce_steps=3)
    tracker.reset(initial_contacts=(False, False), action_size=10)

    simultaneous = _update(tracker, step=1, contacts=(True, True), x=0.0)
    _update(tracker, step=2, contacts=(False, True), x=0.0)
    bounced = _update(tracker, step=3, contacts=(True, True), x=0.01)

    assert simultaneous.valid_alternating_touchdown is False
    assert bounced.right_touchdown is False
    metrics = tracker.episode_metrics()
    assert metrics["gait_right_touchdowns"] == 1.0
    assert metrics["gait_left_touchdowns"] == 1.0
    assert metrics["gait_alternating_touchdowns"] == 0.0


def test_gait_tracker_support_fractions_slip_and_reset() -> None:
    tracker = WalkerGaitTracker()
    tracker.reset(initial_contacts=(False, False), action_size=2)
    tracker.update(
        step=1,
        contacts=(True, False),
        pelvis_x=0.0,
        foot_slip_speeds=(0.4, 9.0),
        applied_action=np.asarray([0.3, 0.4], dtype=np.float32),
    )
    tracker.update(
        step=2,
        contacts=(True, True),
        pelvis_x=0.0,
        foot_slip_speeds=(0.2, 0.6),
        applied_action=np.asarray([0.3, 0.4], dtype=np.float32),
    )
    _update(tracker, step=3, contacts=(False, False), x=0.0)

    metrics = tracker.episode_metrics()
    support_sum = sum(
        metrics[key]
        for key in (
            "gait_right_only_fraction",
            "gait_left_only_fraction",
            "gait_double_support_fraction",
            "gait_flight_fraction",
        )
    )
    assert support_sum == pytest.approx(1.0)
    assert metrics["gait_stance_slip_speed"] == pytest.approx(0.4)
    assert metrics["gait_mean_action_delta_l2"] == pytest.approx(0.5 / 3)

    tracker.reset(initial_contacts=(True, True), action_size=2)
    reset_metrics = tracker.episode_metrics()
    assert reset_metrics["gait_right_touchdowns"] == 0.0
    assert reset_metrics["gait_stance_slip_speed"] == 0.0
    assert reset_metrics["gait_mean_action_delta_l2"] == 0.0
