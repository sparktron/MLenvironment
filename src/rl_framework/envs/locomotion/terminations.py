from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WalkerTermination:
    """Termination policy for the bipedal walker.

    The episode ends when the *torso* actually hits the ground (contact-based),
    or — as a fallback in case contact is missed for a frame — when the torso
    COM drops below ``min_height``. Tilt is intentionally *not* a termination
    condition: it is reported via the reward's orientation penalty instead, so
    the robot is allowed to flail and fall rather than being reset the moment
    it leans past some arbitrary angle.
    """

    # Torso COM must fall below this height to count as "on the ground"
    # when contact detection misses a frame. Roughly mid-shin height.
    min_height: float = 0.18
    # Hard cap on episode length (truncation, not termination).
    max_steps: int = 1000
    # Deprecated: tilt no longer ends the episode. Kept so existing YAMLs and
    # the GUI schema continue to load without errors; the value is ignored.
    max_tilt_radians: float = 3.14

    def check(
        self,
        z_height: float,
        step_count: int,
        torso_contact: bool,
    ) -> tuple[bool, bool]:
        terminated = bool(torso_contact) or z_height < self.min_height
        truncated = step_count >= self.max_steps
        return terminated, truncated
