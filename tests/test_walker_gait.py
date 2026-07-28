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


def _hold(tracker, start_step, n, contacts, x):
    """Hold a contact state for n steps so the debounce can confirm it."""
    last = None
    for i in range(n):
        last = _update(tracker, step=start_step + i, contacts=contacts, x=x)
    return last


def test_gait_tracker_counts_only_ordered_alternating_touchdowns() -> None:
    tracker = WalkerGaitTracker(touchdown_debounce_steps=2)
    tracker.reset(initial_contacts=(True, True), action_size=10)

    # Each state is held for the debounce window; segmentation runs on the
    # confirmed signal, so an event lands on the step it is confirmed.
    _hold(tracker, 1, 2, (False, True), 0.00)
    first = _hold(tracker, 3, 2, (True, True), 0.05)
    _hold(tracker, 5, 2, (True, False), 0.08)
    alternating = _hold(tracker, 7, 2, (True, True), 0.15)
    _hold(tracker, 9, 2, (True, False), 0.18)
    same_foot = _hold(tracker, 11, 2, (True, True), 0.25)

    assert first.valid_alternating_touchdown is False
    assert alternating.valid_alternating_touchdown is True
    assert alternating.alternating_step_progress == pytest.approx(0.10)
    assert same_foot.valid_alternating_touchdown is False
    metrics = tracker.episode_metrics()
    assert metrics["gait_right_touchdowns"] == 1.0
    assert metrics["gait_left_touchdowns"] == 2.0
    assert metrics["gait_alternating_touchdowns"] == 1.0
    assert metrics["gait_longest_same_foot_sequence"] == 2.0


def test_debounce_rejects_brief_contact_breaks_within_one_swing() -> None:
    """A momentary re-contact must not split one swing into two.

    Before the segmentation was debounced, any contact break reset the swing's
    reference height while touchdowns stayed debounced. On the reward_v7
    checkpoint that produced 150 swing fragments where cadence implied ~45 real
    swings, understating mean clearance roughly 3x.
    """
    tracker = WalkerGaitTracker(touchdown_debounce_steps=3, control_timestep=0.1)
    tracker.reset(initial_contacts=(True, True), action_size=10)

    def step(n, contacts, foot_z):
        return _update(
            tracker, step=n, contacts=contacts, x=0.0,
            foot_positions=((0.0, foot_z), (0.0, 0.0)),
        )

    # Right foot lifts (held long enough to confirm) and climbs to 0.10 m.
    for n, z in ((1, 0.02), (2, 0.04), (3, 0.06), (4, 0.08), (5, 0.10)):
        step(n, (False, True), z)
    # A single-step contact blip mid-swing: too short to confirm, so the swing
    # reference height must survive it.
    step(6, (True, True), 0.09)
    for n, z in ((7, 0.08), (8, 0.06), (9, 0.04)):
        step(n, (False, True), z)
    # Genuine landing, held past the debounce window.
    for n in (10, 11, 12):
        step(n, (True, True), 0.02)

    metrics = tracker.episode_metrics()
    assert metrics["gait_qualified_touchdowns"] == 1.0, "the blip must not count"
    # Reference height is the 0.02 m at which contact was genuinely broken, so
    # the recorded clearance is the full 0.10 m peak minus that.
    assert metrics["gait_foot_clearance"] == pytest.approx(0.08)


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
    # debounce 1 keeps the confirmation immediate; this test is about
    # stride/clearance bookkeeping, not contact filtering.
    tracker = WalkerGaitTracker(control_timestep=0.1, touchdown_debounce_steps=1)
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
        touchdown_debounce_steps=1,
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
    tracker = WalkerGaitTracker(control_timestep=0.1, touchdown_debounce_steps=1)
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


def test_swing_clearance_now_tracks_the_lifted_foot_every_step() -> None:
    """Dense per-step clearance, unlike the one-per-landing touchdown value."""
    tracker = WalkerGaitTracker(touchdown_debounce_steps=1, control_timestep=1 / 60)
    tracker.reset(initial_contacts=(True, True), action_size=10)

    # Both feet planted at z=0.02: nothing is swinging.
    step = _update(
        tracker, step=1, contacts=(True, True), x=0.0,
        foot_positions=((0.0, 0.02), (0.0, 0.02)),
    )
    assert step.swing_clearance_now == pytest.approx(0.0)

    # Left foot leaves the ground -> swing starts, clearance measured from there.
    step = _update(
        tracker, step=2, contacts=(True, False), x=0.0,
        foot_positions=((0.0, 0.02), (0.0, 0.02)),
    )
    assert step.swing_clearance_now == pytest.approx(0.0)

    for n, (height, expected) in enumerate(
        [(0.05, 0.03), (0.09, 0.07), (0.06, 0.04)], start=3
    ):
        step = _update(
            tracker, step=n, contacts=(True, False), x=0.0,
            foot_positions=((0.0, 0.02), (0.0, height)),
        )
        assert step.swing_clearance_now == pytest.approx(expected)

    # Back down: swing ends, clearance returns to zero.
    step = _update(
        tracker, step=6, contacts=(True, True), x=0.0,
        foot_positions=((0.0, 0.02), (0.0, 0.02)),
    )
    assert step.swing_clearance_now == pytest.approx(0.0)


def test_touchdown_interval_measures_time_since_the_previous_contact() -> None:
    tracker = WalkerGaitTracker(touchdown_debounce_steps=1, control_timestep=1 / 60)
    tracker.reset(initial_contacts=(False, False), action_size=10)

    # First touchdown has no predecessor to measure against.
    step = _update(tracker, step=1, contacts=(True, False), x=0.0)
    assert step.right_touchdown and step.touchdown_interval is None

    # Steps without a touchdown report nothing.
    step = _update(tracker, step=2, contacts=(True, False), x=0.0)
    assert step.touchdown_interval is None

    # Left lands 11 steps later -> 11/60 s.
    _update(tracker, step=3, contacts=(False, False), x=0.0)
    step = _update(tracker, step=12, contacts=(False, True), x=0.0)
    assert step.left_touchdown
    assert step.touchdown_interval == pytest.approx(11 / 60)


def test_simultaneous_landing_reports_one_interval_not_a_spurious_zero() -> None:
    tracker = WalkerGaitTracker(touchdown_debounce_steps=1, control_timestep=1 / 60)
    tracker.reset(initial_contacts=(False, False), action_size=10)

    _update(tracker, step=1, contacts=(True, False), x=0.0)
    _update(tracker, step=2, contacts=(False, False), x=0.0)
    step = _update(tracker, step=7, contacts=(True, True), x=0.0)

    assert step.right_touchdown and step.left_touchdown
    # Measured from step 1, not 0.0 for whichever foot is processed second.
    assert step.touchdown_interval == pytest.approx(6 / 60)


def test_in_flight_and_contact_duty_imbalance_track_support_state() -> None:
    tracker = WalkerGaitTracker(touchdown_debounce_steps=1, control_timestep=1 / 60)
    tracker.reset(initial_contacts=(True, True), action_size=10)

    # Both feet down: grounded, and double support cancels in the imbalance.
    step = _update(tracker, step=1, contacts=(True, True), x=0.0)
    assert step.in_flight is False
    assert step.contact_duty_imbalance == pytest.approx(0.0)

    # Airborne.
    step = _update(tracker, step=2, contacts=(False, False), x=0.0)
    assert step.in_flight is True

    # Three right-only steps against one left-only: 4 vs 2 stance steps
    # (the initial double-support step counts for both legs).
    for n in (3, 4, 5):
        step = _update(tracker, step=n, contacts=(True, False), x=0.0)
    assert step.in_flight is False
    assert step.contact_duty_imbalance == pytest.approx(3 / 5)

    step = _update(tracker, step=6, contacts=(False, True), x=0.0)
    assert step.contact_duty_imbalance == pytest.approx(2 / 6)


def test_contact_duty_imbalance_is_zero_for_a_symmetric_gait() -> None:
    tracker = WalkerGaitTracker(touchdown_debounce_steps=1, control_timestep=1 / 60)
    tracker.reset(initial_contacts=(False, False), action_size=10)

    step = None
    for cycle in range(4):
        for n in range(2):
            step = _update(
                tracker, step=cycle * 4 + n, contacts=(True, False), x=0.0
            )
        for n in range(2, 4):
            step = _update(
                tracker, step=cycle * 4 + n, contacts=(False, True), x=0.0
            )

    assert step is not None
    assert step.contact_duty_imbalance == pytest.approx(0.0)


def test_in_double_support_is_reported_independently_of_in_flight() -> None:
    tracker = WalkerGaitTracker(touchdown_debounce_steps=1, control_timestep=1 / 60)
    tracker.reset(initial_contacts=(False, False), action_size=10)

    both = _update(tracker, step=1, contacts=(True, True), x=0.0)
    assert both.in_double_support is True and both.in_flight is False

    right = _update(tracker, step=2, contacts=(True, False), x=0.0)
    assert right.in_double_support is False and right.in_flight is False

    air = _update(tracker, step=3, contacts=(False, False), x=0.0)
    assert air.in_double_support is False and air.in_flight is True


def test_mid_stance_slip_excludes_the_touchdown_transient() -> None:
    """gait_stance_slip_speed mixes landing impact with genuine sliding.

    On the reward_v4 checkpoint the first step after touchdown averaged
    1.034 m/s against 0.153 m/s for the rest of the stance, so the whole-phase
    mean reported "sliding" about a foot that was planted.
    """
    tracker = WalkerGaitTracker(
        touchdown_debounce_steps=1, control_timestep=1 / 60, mid_stance_skip_steps=1
    )
    tracker.reset(initial_contacts=(False, False), action_size=10)

    # Right foot lands hard (4.0 m/s) then sits planted at 0.1 m/s.
    _update(tracker, step=1, contacts=(True, False), x=0.0, slips=(4.0, 0.0))
    for n in (2, 3, 4, 5):
        _update(tracker, step=n, contacts=(True, False), x=0.0, slips=(0.1, 0.0))

    m = tracker.episode_metrics()
    # Whole-stance mean is dragged up by the single landing sample.
    assert m["gait_stance_slip_speed"] == pytest.approx((4.0 + 0.4) / 5)
    # Mid-stance sees only the planted foot.
    assert m["gait_mid_stance_slip_speed"] == pytest.approx(0.1)


def test_mid_stance_slip_is_zero_when_never_in_contact() -> None:
    tracker = WalkerGaitTracker(touchdown_debounce_steps=1, control_timestep=1 / 60)
    tracker.reset(initial_contacts=(False, False), action_size=10)
    for n in (1, 2, 3):
        _update(tracker, step=n, contacts=(False, False), x=0.0)
    assert tracker.episode_metrics()["gait_mid_stance_slip_speed"] == 0.0


def _swing_and_land(tracker, start_step, side, peak_z, land_z=0.02):
    """Lift one foot to peak_z and land it, producing a qualified touchdown."""
    contacts = [True, True]
    contacts[side] = False
    pos = [(0.0, land_z), (0.0, land_z)]
    pos[side] = (0.0, land_z)
    tracker.update(
        step=start_step, contacts=tuple(contacts), pelvis_x=0.0,
        foot_slip_speeds=(0.0, 0.0), foot_positions=tuple(pos),
        applied_action=np.zeros(10, dtype=np.float32),
    )
    pos[side] = (0.0, land_z + peak_z)
    tracker.update(
        step=start_step + 1, contacts=tuple(contacts), pelvis_x=0.0,
        foot_slip_speeds=(0.0, 0.0), foot_positions=tuple(pos),
        applied_action=np.zeros(10, dtype=np.float32),
    )
    pos[side] = (0.0, land_z)
    return tracker.update(
        step=start_step + 2, contacts=(True, True), pelvis_x=0.0,
        foot_slip_speeds=(0.0, 0.0), foot_positions=tuple(pos),
        applied_action=np.zeros(10, dtype=np.float32),
    )


def test_bilateral_swing_clearance_tracks_the_worse_leg() -> None:
    tracker = WalkerGaitTracker(touchdown_debounce_steps=1, control_timestep=1 / 60)
    tracker.reset(initial_contacts=(True, True), action_size=10)

    # Only the right foot has swung: the left has no credit yet, so the pair
    # value stays at zero rather than paying for one good leg.
    step = _swing_and_land(tracker, 1, side=0, peak_z=0.10)
    assert step.bilateral_swing_clearance == pytest.approx(0.0)

    # Left foot now swings, but only 5 mm -- the dragging leg sets the value.
    step = _swing_and_land(tracker, 10, side=1, peak_z=0.005)
    assert step.bilateral_swing_clearance == pytest.approx(0.005)

    # Left foot swings properly: both legs now clear, so the pair value rises.
    step = _swing_and_land(tracker, 20, side=1, peak_z=0.08)
    assert step.bilateral_swing_clearance == pytest.approx(0.08)


def test_bilateral_swing_clearance_expires_a_dragged_leg() -> None:
    tracker = WalkerGaitTracker(
        touchdown_debounce_steps=1, control_timestep=1 / 60,
        clearance_staleness_steps=20,
    )
    tracker.reset(initial_contacts=(True, True), action_size=10)
    _swing_and_land(tracker, 1, side=0, peak_z=0.10)
    step = _swing_and_land(tracker, 5, side=1, peak_z=0.10)
    assert step.bilateral_swing_clearance == pytest.approx(0.10)

    # The right foot stops swinging; past the staleness window it counts as 0.
    step = _swing_and_land(tracker, 60, side=1, peak_z=0.10)
    assert step.bilateral_swing_clearance == pytest.approx(0.0)
