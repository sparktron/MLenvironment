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


@dataclass
class WalkerGaitTracker:
    """Episode-local, reward-agnostic walker gait telemetry."""

    touchdown_debounce_steps: int = 3
    _previous_contacts: tuple[bool, bool] = (False, False)
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
    _longest_same_foot_sequence: int = 0
    _action_delta_sum: float = 0.0
    _previous_action: np.ndarray | None = None
    _steps: int = 0

    def __post_init__(self) -> None:
        if self.touchdown_debounce_steps < 1:
            raise ValueError("touchdown_debounce_steps must be at least 1")

    def reset(
        self,
        *,
        initial_contacts: tuple[bool, bool],
        action_size: int,
    ) -> None:
        self._previous_contacts = initial_contacts
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
        self._longest_same_foot_sequence = 0
        self._action_delta_sum = 0.0
        self._previous_action = np.zeros(action_size, dtype=np.float32)
        self._steps = 0

    def update(
        self,
        *,
        step: int,
        contacts: tuple[bool, bool],
        pelvis_x: float,
        foot_slip_speeds: tuple[float, float],
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
        self._action_delta_sum += float(np.linalg.norm(action - self._previous_action))
        self._previous_action = action.copy()

        accepted_touchdowns: list[int] = []
        for side, (contact, previous_contact) in enumerate(
            zip(contacts, self._previous_contacts)
        ):
            if (
                contact
                and not previous_contact
                and step - self._last_touchdown_steps[side]
                >= self.touchdown_debounce_steps
            ):
                accepted_touchdowns.append(side)
                self._last_touchdown_steps[side] = step
                self._touchdown_counts[side] += 1
        self._previous_contacts = contacts

        alternating = False
        progress = 0.0
        # A simultaneous two-foot landing is real contact telemetry, but it is
        # not evidence of an ordered right/left gait event.
        if len(accepted_touchdowns) == 1:
            side = accepted_touchdowns[0]
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
            "gait_stance_slip_speed": (
                self._stance_slip_sum / self._stance_steps
                if self._stance_steps
                else 0.0
            ),
            "gait_longest_same_foot_sequence": float(
                self._longest_same_foot_sequence
            ),
            "gait_mean_action_delta_l2": self._action_delta_sum / steps,
        }
