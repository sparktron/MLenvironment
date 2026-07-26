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
    foot_positions: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.0), (0.0, 0.0)),
):
    return tracker.update(
        step=step,
        contacts=contacts,
        pelvis_x=x,
        foot_slip_speeds=slips,
        foot_positions=foot_positions,
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
        foot_positions=((0.0, 0.0), (0.0, 0.0)),
        applied_action=np.asarray([0.3, 0.4], dtype=np.float32),
    )
    tracker.update(
        step=2,
        contacts=(True, True),
        pelvis_x=0.0,
        foot_slip_speeds=(0.2, 0.6),
        foot_positions=((0.0, 0.0), (0.0, 0.0)),
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


def test_gait_tracker_reports_stride_swing_clearance_and_cadence() -> None:
    tracker = WalkerGaitTracker(control_timestep=0.1)
    tracker.reset(initial_contacts=(True, True), action_size=2)

    _update(tracker, step=1, contacts=(False, True), x=0.0, foot_positions=((0.0, 0.03), (0.0, 0.0)))
    _update(tracker, step=2, contacts=(False, True), x=0.0, foot_positions=((0.1, 0.12), (0.0, 0.0)))
    _update(tracker, step=3, contacts=(True, True), x=0.0, foot_positions=((0.2, 0.02), (0.0, 0.0)))
    _update(tracker, step=4, contacts=(False, True), x=0.0, foot_positions=((0.2, 0.03), (0.0, 0.0)))
    _update(tracker, step=5, contacts=(False, True), x=0.0, foot_positions=((0.4, 0.15), (0.0, 0.0)))
    _update(tracker, step=6, contacts=(True, True), x=0.0, foot_positions=((0.5, 0.02), (0.0, 0.0)))

    metrics = tracker.episode_metrics()
    assert metrics["gait_swing_duration"] == pytest.approx(0.2)
    assert metrics["gait_foot_clearance"] == pytest.approx(0.105)
    assert metrics["gait_stride_length"] == pytest.approx(0.3)
    assert metrics["gait_qualified_touchdowns"] == 2.0
    assert metrics["gait_cadence_steps_per_min"] == pytest.approx(200.0)


def test_sustained_swing_progress_requires_duration_and_clearance() -> None:
    tracker = WalkerGaitTracker(
        control_timestep=0.1,
        min_swing_duration=0.2,
        min_foot_clearance=0.05,
    )
    tracker.reset(initial_contacts=(True, True), action_size=2)

    _update(tracker, step=1, contacts=(False, True), x=0.0, foot_positions=((0.0, 0.02), (0.0, 0.0)))
    _update(tracker, step=2, contacts=(False, True), x=0.0, foot_positions=((0.1, 0.08), (0.0, 0.0)))
    first = _update(tracker, step=3, contacts=(True, True), x=0.1, foot_positions=((0.2, 0.02), (0.0, 0.0)))
    _update(tracker, step=4, contacts=(True, False), x=0.1, foot_positions=((0.2, 0.0), (0.0, 0.02)))
    _update(tracker, step=5, contacts=(True, False), x=0.1, foot_positions=((0.2, 0.0), (0.1, 0.08)))
    second = _update(tracker, step=6, contacts=(True, True), x=0.3, foot_positions=((0.2, 0.0), (0.2, 0.02)))

    assert first.sustained_swing_touchdown is True
    assert first.touchdown_foot_lead == pytest.approx(0.1)
    assert first.sustained_swing_progress == 0.0
    assert second.valid_alternating_touchdown is True
    assert second.sustained_swing_touchdown is True
    assert second.touchdown_foot_lead == pytest.approx(-0.1)
    assert second.sustained_swing_progress == pytest.approx(0.2)
    assert tracker.episode_metrics()["gait_sustained_swing_touchdowns"] == 2.0


def test_gait_tracker_reports_completed_stance_advance_duration_and_slip() -> None:
    tracker = WalkerGaitTracker(control_timestep=0.1)
    tracker.reset(initial_contacts=(False, True), action_size=2)

    touchdown = _update(
        tracker, step=1, contacts=(True, True), x=0.0, slips=(0.2, 0.0)
    )
    _update(tracker, step=2, contacts=(True, True), x=0.05, slips=(0.4, 0.0))
    liftoff = _update(
        tracker, step=3, contacts=(False, True), x=0.12, slips=(9.0, 0.0)
    )

    assert touchdown.stance_completion_side is None
    assert liftoff.stance_completion_side == 0
    assert liftoff.stance_pelvis_advance == pytest.approx(0.12)
    assert liftoff.stance_duration == pytest.approx(0.2)
    assert liftoff.stance_mean_slip_speed == pytest.approx(0.3)
