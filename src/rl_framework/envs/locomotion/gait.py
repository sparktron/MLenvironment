from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class GaitStep:
    """Gait events and contact measurements produced by one control step."""

    right_touchdown: bool
    left_touchdown: bool
    valid_alternating_touchdown: bool
    alternating_step_progress: float
    stance_slip_speed: float
    action_delta_l2: float
    # Height of the highest currently-swinging foot above the ground it left,
    # in metres; 0.0 whenever no foot is mid-swing. Unlike
    # ``touchdown_swing_clearance`` (one value per landing) this is available
    # on every step, so a reward can pay for lifting continuously.
    swing_clearance_now: float
    # Seconds since the previous accepted touchdown on either foot. ``None``
    # on steps with no touchdown and on the episode's first touchdown.
    touchdown_interval: float | None
    # True when neither foot is on a supporting surface. A walk has no flight
    # phase at all; a run is defined by having one.
    in_flight: bool
    # The LOWER of the two feet's most recent qualified swing clearances, in
    # metres, with a stale value counted as zero. Dense on every step and
    # independent of cadence, unlike a per-touchdown payment, and it cannot be
    # farmed by one high-swinging leg the way a max-over-feet value can.
    bilateral_swing_clearance: float
    # True when BOTH feet are supported. Flight, single support and double
    # support are three states, not two: penalising flight alone lets a policy
    # escape into extended single support, which is what the v3 screen did.
    in_double_support: bool
    # Running |right - left| stance-step imbalance as a fraction of total
    # stance steps, in [0, 1]. 0.0 is a perfectly even duty cycle between the
    # legs; 1.0 means one leg has done all the standing.
    contact_duty_imbalance: float
    touchdown_swing_duration: float | None
    touchdown_swing_clearance: float | None
    touchdown_side: int | None
    touchdown_foot_lead: float | None
    sustained_swing_touchdown: bool
    sustained_swing_progress: float
    stance_completion_side: int | None
    stance_pelvis_advance: float | None
    stance_duration: float | None
    stance_mean_slip_speed: float | None


@dataclass
class WalkerGaitTracker:
    """Episode-local, reward-agnostic walker gait telemetry."""

    touchdown_debounce_steps: int = 3
    control_timestep: float = 1.0 / 60.0
    min_swing_duration: float = 0.0
    min_foot_clearance: float = 0.0
    # Control steps to skip after each touchdown when accumulating
    # `gait_mid_stance_slip_speed`. `gait_stance_slip_speed` averages the whole
    # contact phase, which mixes the landing transient (the foot still moving
    # as it arrives) with genuine sliding. On the reward_v4 checkpoint the
    # first step after touchdown averages 1.034 m/s against 0.153 m/s for the
    # rest of the stance, so the whole-phase mean says "sliding" about a foot
    # that is planted. Stances here are only 2-4 steps, so 1 is the right skip.
    mid_stance_skip_steps: int = 1
    # A foot whose last qualified landing is older than this is treated as
    # having zero clearance, so a dragged leg cannot coast on a stale value.
    # 150 steps (2.5 s) clears the slowest gated cadence (0.45 Hz cycle -> one
    # landing per foot every 2.2 s).
    clearance_staleness_steps: int = 150
    _previous_contacts: tuple[bool, bool] = (False, False)
    # Debounced contact signal used for EVENT segmentation (swings, stances,
    # touchdowns). Raw contacts still drive the occupancy fractions, which are
    # instantaneous measures. Without this, a swing's reference height was
    # reset by any momentary re-contact while touchdowns were debounced, so the
    # two disagreed: on the reward_v7 checkpoint the raw signal produced 150
    # swing fragments where cadence implied ~45 real swings, understating mean
    # clearance 3x (7.2 mm against ~21 mm).
    _confirmed_contacts: tuple[bool, bool] = (False, False)
    _contact_disagree: list[int] = field(default_factory=lambda: [0, 0])
    # Foot height and step index captured on the FIRST step the raw signal
    # disagrees with the confirmed one. Confirmation lags by the debounce
    # window, and the foot keeps rising during it, so measuring a swing from
    # the confirmation step would understate clearance and duration.
    _pending_break_z: list[float | None] = field(default_factory=lambda: [None, None])
    _pending_break_step: list[int | None] = field(
        default_factory=lambda: [None, None]
    )
    _last_touchdown_steps: list[int] = field(default_factory=lambda: [-10_000, -10_000])
    _last_touchdown_side: int | None = None
    _last_touchdown_x: float | None = None
    _same_foot_sequence: int = 0
    _support_counts: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    _touchdown_counts: list[int] = field(default_factory=lambda: [0, 0])
    _alternating_touchdowns: int = 0
    _alternating_progress: float = 0.0
    _stance_slip_sum: float = 0.0
    _stance_steps: int = 0
    _mid_stance_slip_sum: float = 0.0
    _mid_stance_steps: int = 0
    _longest_same_foot_sequence: int = 0
    _action_delta_sum: float = 0.0
    _previous_action: np.ndarray | None = None
    _swing_start_steps: list[int | None] = field(default_factory=lambda: [None, None])
    _swing_start_z: list[float] = field(default_factory=lambda: [0.0, 0.0])
    _swing_peak_z: list[float] = field(default_factory=lambda: [0.0, 0.0])
    _last_stride_touchdown_x: list[float | None] = field(
        default_factory=lambda: [None, None]
    )
    _swing_durations: list[float] = field(default_factory=list)
    _swing_clearances: list[float] = field(default_factory=list)
    _stride_lengths: list[float] = field(default_factory=list)
    _last_swing_clearance: list[float] = field(default_factory=lambda: [0.0, 0.0])
    _last_swing_touchdown_step: list[int | None] = field(
        default_factory=lambda: [None, None]
    )
    _qualified_touchdowns: int = 0
    _sustained_swing_touchdowns: int = 0
    _stance_start_steps: list[int | None] = field(default_factory=lambda: [None, None])
    _stance_start_pelvis_x: list[float] = field(default_factory=lambda: [0.0, 0.0])
    _stance_slip_sums: list[float] = field(default_factory=lambda: [0.0, 0.0])
    _stance_slip_counts: list[int] = field(default_factory=lambda: [0, 0])
    _last_any_touchdown_step: int | None = None
    _steps: int = 0

    def __post_init__(self) -> None:
        if self.touchdown_debounce_steps < 1:
            raise ValueError("touchdown_debounce_steps must be at least 1")
        if self.control_timestep <= 0:
            raise ValueError("control_timestep must be positive")
        if self.min_swing_duration < 0:
            raise ValueError("min_swing_duration must be non-negative")
        if self.min_foot_clearance < 0:
            raise ValueError("min_foot_clearance must be non-negative")

    def reset(
        self,
        *,
        initial_contacts: tuple[bool, bool],
        action_size: int,
    ) -> None:
        self._previous_contacts = initial_contacts
        self._confirmed_contacts = initial_contacts
        self._contact_disagree = [0, 0]
        self._pending_break_z = [None, None]
        self._pending_break_step = [None, None]
        self._last_touchdown_steps = [-10_000, -10_000]
        self._last_touchdown_side = None
        self._last_touchdown_x = None
        self._same_foot_sequence = 0
        self._support_counts = [0, 0, 0, 0]
        self._touchdown_counts = [0, 0]
        self._alternating_touchdowns = 0
        self._alternating_progress = 0.0
        self._stance_slip_sum = 0.0
        self._stance_steps = 0
        self._mid_stance_slip_sum = 0.0
        self._mid_stance_steps = 0
        self._longest_same_foot_sequence = 0
        self._action_delta_sum = 0.0
        self._previous_action = np.zeros(action_size, dtype=np.float32)
        self._swing_start_steps = [None, None]
        self._swing_start_z = [0.0, 0.0]
        self._swing_peak_z = [0.0, 0.0]
        self._last_stride_touchdown_x = [None, None]
        self._swing_durations = []
        self._swing_clearances = []
        self._stride_lengths = []
        self._last_swing_clearance = [0.0, 0.0]
        self._last_swing_touchdown_step = [None, None]
        self._qualified_touchdowns = 0
        self._sustained_swing_touchdowns = 0
        self._stance_start_steps = [None, None]
        self._stance_start_pelvis_x = [0.0, 0.0]
        self._stance_slip_sums = [0.0, 0.0]
        self._stance_slip_counts = [0, 0]
        self._last_any_touchdown_step = None
        self._steps = 0

    def update(
        self,
        *,
        step: int,
        contacts: tuple[bool, bool],
        pelvis_x: float,
        foot_slip_speeds: tuple[float, float],
        foot_positions: tuple[tuple[float, float], tuple[float, float]],
        applied_action: np.ndarray,
    ) -> GaitStep:
        """Record one step and return reward-ready structural measurements."""
        self._steps += 1
        right_contact, left_contact = contacts
        if right_contact and left_contact:
            self._support_counts[2] += 1
        elif right_contact:
            self._support_counts[0] += 1
        elif left_contact:
            self._support_counts[1] += 1
        else:
            self._support_counts[3] += 1

        in_flight = not right_contact and not left_contact
        in_double_support = right_contact and left_contact
        # Double-support steps count for both legs and cancel in the
        # difference, so the imbalance is driven entirely by single-support.
        right_stance_steps = self._support_counts[0] + self._support_counts[2]
        left_stance_steps = self._support_counts[1] + self._support_counts[2]
        total_stance_steps = right_stance_steps + left_stance_steps
        contact_duty_imbalance = (
            abs(right_stance_steps - left_stance_steps) / total_stance_steps
            if total_stance_steps
            else 0.0
        )

        confirmed = list(self._confirmed_contacts)
        for side in (0, 1):
            if contacts[side] != confirmed[side]:
                if self._contact_disagree[side] == 0:
                    self._pending_break_z[side] = float(foot_positions[side][1])
                    self._pending_break_step[side] = step
                self._contact_disagree[side] += 1
                if self._contact_disagree[side] >= self.touchdown_debounce_steps:
                    confirmed[side] = contacts[side]
                    self._contact_disagree[side] = 0
            else:
                self._contact_disagree[side] = 0
        self._confirmed_contacts = (confirmed[0], confirmed[1])
        segmentation_contacts = self._confirmed_contacts

        contacting_speeds = [
            max(float(speed), 0.0)
            for contact, speed in zip(contacts, foot_slip_speeds)
            if contact and np.isfinite(speed)
        ]
        stance_slip_speed = (
            float(np.mean(contacting_speeds)) if contacting_speeds else 0.0
        )
        if contacting_speeds:
            self._stance_slip_sum += stance_slip_speed
            self._stance_steps += 1

        action = np.asarray(applied_action, dtype=np.float32)
        if self._previous_action is None or self._previous_action.shape != action.shape:
            self._previous_action = np.zeros_like(action)
        action_delta_l2 = float(np.linalg.norm(action - self._previous_action))
        self._action_delta_sum += action_delta_l2
        self._previous_action = action.copy()

        accepted_touchdowns: list[int] = []
        swing_events: dict[int, tuple[float, float]] = {}
        stance_completion_side: int | None = None
        stance_pelvis_advance: float | None = None
        stance_duration: float | None = None
        stance_mean_slip_speed: float | None = None
        swing_clearance_now = 0.0
        touchdown_interval: float | None = None
        # Captured before the loop so a simultaneous two-foot landing reports
        # one interval measured from the *previous* step, not a spurious 0.0
        # for the second foot.
        previous_any_touchdown_step = self._last_any_touchdown_step
        for side, (contact, previous_contact) in enumerate(
            zip(segmentation_contacts, self._previous_contacts)
        ):
            foot_x, foot_z = foot_positions[side]
            if not contact and previous_contact:
                stance_start = self._stance_start_steps[side]
                if stance_start is not None:
                    stance_completion_side = side
                    stance_pelvis_advance = float(
                        pelvis_x - self._stance_start_pelvis_x[side]
                    )
                    stance_duration = (step - stance_start) * self.control_timestep
                    slip_count = self._stance_slip_counts[side]
                    stance_mean_slip_speed = (
                        self._stance_slip_sums[side] / slip_count if slip_count else 0.0
                    )
                self._stance_start_steps[side] = None
                self._stance_slip_sums[side] = 0.0
                self._stance_slip_counts[side] = 0
            if not contact:
                if previous_contact:
                    # Date the swing from the raw break, not its confirmation.
                    pending_step = self._pending_break_step[side]
                    pending_z = self._pending_break_z[side]
                    self._swing_start_steps[side] = (
                        pending_step if pending_step is not None else step
                    )
                    start_z = pending_z if pending_z is not None else float(foot_z)
                    self._swing_start_z[side] = start_z
                    self._swing_peak_z[side] = max(start_z, float(foot_z))
                elif self._swing_start_steps[side] is not None:
                    self._swing_peak_z[side] = max(
                        self._swing_peak_z[side], float(foot_z)
                    )
                if self._swing_start_steps[side] is not None:
                    swing_clearance_now = max(
                        swing_clearance_now,
                        float(foot_z) - self._swing_start_z[side],
                    )
            if (
                contact
                and not previous_contact
                and step - self._last_touchdown_steps[side]
                >= self.touchdown_debounce_steps
            ):
                accepted_touchdowns.append(side)
                if previous_any_touchdown_step is not None:
                    touchdown_interval = (
                        step - previous_any_touchdown_step
                    ) * self.control_timestep
                self._last_any_touchdown_step = step
                self._last_touchdown_steps[side] = step
                self._touchdown_counts[side] += 1
                self._stance_start_steps[side] = step
                self._stance_start_pelvis_x[side] = float(pelvis_x)
                self._stance_slip_sums[side] = 0.0
                self._stance_slip_counts[side] = 0
                swing_start = self._swing_start_steps[side]
                if swing_start is not None:
                    self._qualified_touchdowns += 1
                    duration = (step - swing_start) * self.control_timestep
                    clearance = max(
                        self._swing_peak_z[side] - self._swing_start_z[side], 0.0
                    )
                    self._swing_durations.append(duration)
                    self._swing_clearances.append(clearance)
                    self._last_swing_clearance[side] = clearance
                    self._last_swing_touchdown_step[side] = step
                    swing_events[side] = (duration, clearance)
                    previous_x = self._last_stride_touchdown_x[side]
                    if previous_x is not None:
                        self._stride_lengths.append(abs(float(foot_x) - previous_x))
                    self._last_stride_touchdown_x[side] = float(foot_x)
                self._swing_start_steps[side] = None
            # Slip is an instantaneous measurement, so it follows the RAW
            # contact signal. Using the debounced one would extend the stance
            # window past the real liftoff and count swing motion as slip.
            if contacts[side] and self._stance_start_steps[side] is not None:
                slip_speed = float(foot_slip_speeds[side])
                if np.isfinite(slip_speed):
                    self._stance_slip_sums[side] += max(slip_speed, 0.0)
                    self._stance_slip_counts[side] += 1
                    stance_age = step - self._stance_start_steps[side]
                    if stance_age >= self.mid_stance_skip_steps:
                        self._mid_stance_slip_sum += max(slip_speed, 0.0)
                        self._mid_stance_steps += 1
        bilateral_swing_clearance = min(
            0.0
            if last_step is None
            or step - last_step > self.clearance_staleness_steps
            else self._last_swing_clearance[side]
            for side, last_step in enumerate(self._last_swing_touchdown_step)
        )
        self._previous_contacts = segmentation_contacts

        alternating = False
        progress = 0.0
        touchdown_swing_duration: float | None = None
        touchdown_swing_clearance: float | None = None
        touchdown_side: int | None = None
        touchdown_foot_lead: float | None = None
        sustained_swing_touchdown = False
        # A simultaneous two-foot landing is real contact telemetry, but it is
        # not evidence of an ordered right/left gait event.
        if len(accepted_touchdowns) == 1:
            side = accepted_touchdowns[0]
            touchdown_side = side
            touchdown_foot_lead = float(foot_positions[side][0] - pelvis_x)
            swing_event = swing_events.get(side)
            if swing_event is not None:
                touchdown_swing_duration, touchdown_swing_clearance = swing_event
                sustained_swing_touchdown = bool(
                    touchdown_swing_duration >= self.min_swing_duration
                    and touchdown_swing_clearance >= self.min_foot_clearance
                )
                if sustained_swing_touchdown:
                    self._sustained_swing_touchdowns += 1
            if self._last_touchdown_side is None:
                self._same_foot_sequence = 1
            elif side == self._last_touchdown_side:
                self._same_foot_sequence += 1
            else:
                self._same_foot_sequence = 1
                alternating = True
                if self._last_touchdown_x is not None:
                    progress = float(pelvis_x - self._last_touchdown_x)
                self._alternating_touchdowns += 1
                self._alternating_progress += progress
            self._longest_same_foot_sequence = max(
                self._longest_same_foot_sequence,
                self._same_foot_sequence,
            )
            self._last_touchdown_side = side
            self._last_touchdown_x = float(pelvis_x)
        elif len(accepted_touchdowns) > 1:
            self._last_touchdown_side = None
            self._last_touchdown_x = None
            self._same_foot_sequence = 0

        return GaitStep(
            right_touchdown=0 in accepted_touchdowns,
            left_touchdown=1 in accepted_touchdowns,
            valid_alternating_touchdown=alternating,
            alternating_step_progress=progress if alternating else 0.0,
            stance_slip_speed=stance_slip_speed,
            action_delta_l2=action_delta_l2,
            swing_clearance_now=swing_clearance_now,
            touchdown_interval=touchdown_interval,
            in_flight=in_flight,
            in_double_support=in_double_support,
            bilateral_swing_clearance=bilateral_swing_clearance,
            contact_duty_imbalance=contact_duty_imbalance,
            touchdown_swing_duration=touchdown_swing_duration,
            touchdown_swing_clearance=touchdown_swing_clearance,
            touchdown_side=touchdown_side,
            touchdown_foot_lead=touchdown_foot_lead,
            sustained_swing_touchdown=sustained_swing_touchdown,
            sustained_swing_progress=(
                progress if alternating and sustained_swing_touchdown else 0.0
            ),
            stance_completion_side=stance_completion_side,
            stance_pelvis_advance=stance_pelvis_advance,
            stance_duration=stance_duration,
            stance_mean_slip_speed=stance_mean_slip_speed,
        )

    def episode_metrics(self) -> dict[str, float]:
        steps = max(self._steps, 1)
        alternating = max(self._alternating_touchdowns, 1)
        return {
            "gait_right_only_fraction": self._support_counts[0] / steps,
            "gait_left_only_fraction": self._support_counts[1] / steps,
            "gait_double_support_fraction": self._support_counts[2] / steps,
            "gait_flight_fraction": self._support_counts[3] / steps,
            "gait_right_touchdowns": float(self._touchdown_counts[0]),
            "gait_left_touchdowns": float(self._touchdown_counts[1]),
            "gait_alternating_touchdowns": float(self._alternating_touchdowns),
            "gait_alternating_touchdowns_per_100_steps": (
                100.0 * self._alternating_touchdowns / steps
            ),
            "gait_progress_per_alternating_touchdown": (
                self._alternating_progress / alternating
                if self._alternating_touchdowns
                else 0.0
            ),
            "gait_mid_stance_slip_speed": (
                self._mid_stance_slip_sum / self._mid_stance_steps
                if self._mid_stance_steps
                else 0.0
            ),
            "gait_stance_slip_speed": (
                self._stance_slip_sum / self._stance_steps
                if self._stance_steps
                else 0.0
            ),
            "gait_longest_same_foot_sequence": float(self._longest_same_foot_sequence),
            "gait_mean_action_delta_l2": self._action_delta_sum / steps,
            "gait_stride_length": (
                float(np.mean(self._stride_lengths)) if self._stride_lengths else 0.0
            ),
            "gait_swing_duration": (
                float(np.mean(self._swing_durations)) if self._swing_durations else 0.0
            ),
            "gait_foot_clearance": (
                float(np.mean(self._swing_clearances))
                if self._swing_clearances
                else 0.0
            ),
            "gait_cadence_steps_per_min": (
                60.0 * self._qualified_touchdowns / (steps * self.control_timestep)
            ),
            "gait_qualified_touchdowns": float(self._qualified_touchdowns),
            "gait_sustained_swing_touchdowns": float(self._sustained_swing_touchdowns),
        }
