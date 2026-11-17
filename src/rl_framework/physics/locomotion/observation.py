"""Walker observation layout and ``info`` schema.

The 35/45-dim layout used to exist only as the order of ``values.extend``
calls in ``_get_obs`` and as prose in AGENTS.md. Diagnostics, gait code and
checkpoint compatibility all depended on it implicitly. ``ObservationLayout``
is that layout as data: named slices in order, built from the observation
config, so anything that needs "where is the foot contact flag" or "is this
checkpoint's input the same shape as this env's" asks the layout.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_framework.physics.locomotion.config import ObservationConfig

N_JOINTS = 10


@dataclass(frozen=True)
class ObservationLayout:
    """Ordered, named slices of the flat observation vector."""

    slices: tuple[tuple[str, slice], ...]
    size: int

    @classmethod
    def from_config(cls, cfg: ObservationConfig) -> ObservationLayout:
        # v1: pos(3)+quat(4)+lin_vel(3)+ang_vel(3)+joint_pos(10)+joint_vel(10)
        #     +mass_scale(1)+friction_scale(1) = 35.
        # v2 adds binary right/left foot contacts. Coordinate-free drops global
        # x/y, keeping height as the useful local cue. The DR scales make
        # randomisation observable; without them it looks like pure noise.
        parts: list[tuple[str, int]] = [
            (
                "z" if cfg.coordinate_free else "position",
                1 if cfg.coordinate_free else 3,
            ),
            ("quat", 4),
            ("lin_vel", 3),
            ("ang_vel", 3),
            ("joint_pos", N_JOINTS),
            ("joint_vel", N_JOINTS),
            ("mass_scale", 1),
            ("friction_scale", 1),
        ]
        if cfg.version == "v2":
            parts.append(("foot_contacts", 2))
        if cfg.include_previous_action:
            parts.append(("previous_action", N_JOINTS))
        # The gait clock enters as (sin, cos) so it is continuous across the
        # wrap at phase 1 -> 0.
        if cfg.include_gait_phase:
            parts.append(("gait_phase", 2))
        slices: list[tuple[str, slice]] = []
        start = 0
        for name, width in parts:
            slices.append((name, slice(start, start + width)))
            start += width
        return cls(tuple(slices), start)

    def __getitem__(self, name: str) -> slice:
        for key, sl in self.slices:
            if key == name:
                return sl
        raise KeyError(f"observation has no {name!r} slice; has {self.names}")

    def __contains__(self, name: object) -> bool:
        return any(key == name for key, _ in self.slices)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self.slices)

    def signature(self) -> tuple[tuple[str, int], ...]:
        """(name, width) pairs: equal signatures mean checkpoint-compatible input."""
        return tuple((key, sl.stop - sl.start) for key, sl in self.slices)

    def describe_mismatch(self, other: ObservationLayout) -> str | None:
        """Human-readable difference from ``other``, or None if compatible."""
        if self.signature() == other.signature():
            return None
        mine, theirs = dict(self.signature()), dict(other.signature())
        only_here = [n for n in mine if n not in theirs]
        only_there = [n for n in theirs if n not in mine]
        return (
            f"observation size {self.size} vs {other.size}; "
            f"only in first: {only_here}; only in second: {only_there}"
        )


# Keys ``WalkerBulletEnv.step`` always puts in ``info``. Telemetry and
# diagnostics key off this rather than rediscovering it from a rollout.
STEP_INFO_KEYS = frozenset(
    {
        "x_position",
        "z_position",
        "lin_vel_x",
        "roll",
        "pitch",
        "abs_tilt",
        "action_l2",
        "action_scale",
        "reward_components",
        "torso_contact",
        "torso_contact_body",
        "right_foot_contact",
        "left_foot_contact",
        "right_foot_touchdown",
        "left_foot_touchdown",
        "valid_alternating_touchdown",
        "gait_step_progress",
        "sustained_swing_progress",
        "stance_slip_speed",
        "action_delta_l2",
        "swing_clearance_now",
        "gait_phase",
        "phase_contact_match",
        "touchdown_interval",
        "contact_duty_imbalance",
        "target_velocity",
        "velocity_error",
        "right_foot_tangential_velocity",
        "left_foot_tangential_velocity",
        "right_foot_normal_force",
        "left_foot_normal_force",
        "swing_touchdown_event",
        "stance_completion_event",
        "termination_reason",
        "push_applied",
        "terrain_preset",
    }
)
# Added only on the final step of an episode.
EPISODE_INFO_KEY = "walker_episode"
