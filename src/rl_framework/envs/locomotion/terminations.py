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
    # If the torso COM rises above this height, the agent is no longer
    # walking — it is exploiting the PD as propulsion and launching itself
    # off the ground. Default ≈ 2× standing height (TORSO_STAND_Z ≈ 0.68).
    max_height: float = 1.5
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
        too_low = z_height < self.min_height
        too_high = z_height > self.max_height
        terminated = bool(torso_contact) or too_low or too_high
        truncated = step_count >= self.max_steps
        return terminated, truncated
