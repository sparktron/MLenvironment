from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class WalkerReward:
    """Shaped reward for the bipedal walker.

    Design notes (vs. the original v-shape velocity term):
    - ``alive_bonus`` is deliberately small: velocity tracking is the main
      positive signal, while survival remains a gentle bootstrap incentive.
    - The velocity reward is a **Gaussian around the target**, so standing
      still scores a small positive (≈ alive_bonus + 0.3) instead of being
      penalised harder than falling.
    - A one-time ``fall_penalty`` fires only on the terminal step of a fall
      (not on truncation), giving a clear gradient toward staying upright.
    - ``orientation_penalty_weight`` is scaled down to make the early
      bootstrap easier; raise it later via curriculum.
    """

    alive_bonus: float = 0.25
    forward_velocity_weight: float = 2.0
    target_velocity: float = 1.0
    # Width (1 sigma) of the Gaussian around target_velocity. Larger →
    # broader credit for "close to target" speeds.
    velocity_sigma: float = 0.5
    orientation_penalty_weight: float = 0.3
    torque_penalty_weight: float = 0.01
    # One-time penalty applied on the step a fall is detected.
    fall_penalty: float = 10.0
    # Structural gait terms are opt-in so existing configs and checkpoints
    # retain bit-for-bit reward semantics.
    gait_step_progress_weight: float = 0.0
    gait_step_progress_clip: float = 0.25
    stance_slip_penalty_weight: float = 0.0

    def compute(
        self,
        lin_vel_x: float,
        pitch_roll_penalty: float,
        action: np.ndarray,
        alive: bool,
        fell: bool = False,
        gait_step_progress: float = 0.0,
        stance_slip_speed: float = 0.0,
    ) -> float:
        return float(
            sum(
                self.components(
                    lin_vel_x=lin_vel_x,
                    pitch_roll_penalty=pitch_roll_penalty,
                    action=action,
                    alive=alive,
                    fell=fell,
                    gait_step_progress=gait_step_progress,
                    stance_slip_speed=stance_slip_speed,
                ).values()
            )
        )

    def components(
        self,
        lin_vel_x: float,
        pitch_roll_penalty: float,
        action: np.ndarray,
        alive: bool,
        fell: bool = False,
        gait_step_progress: float = 0.0,
        stance_slip_speed: float = 0.0,
    ) -> dict[str, float]:
        """Return individually attributable reward terms.

        Keeping the decomposition in the reward object prevents diagnostics
        from reimplementing—and eventually drifting from—the actual reward.
        The terms sum exactly to :meth:`compute`.
        """
        alive_reward = self.alive_bonus if alive else 0.0
        # Gaussian velocity reward: 1.0 at v=target, decays smoothly outside.
        diff = (lin_vel_x - self.target_velocity) / max(self.velocity_sigma, 1e-6)
        velocity_reward = self.forward_velocity_weight * math.exp(-0.5 * diff * diff)
        orientation_penalty = -self.orientation_penalty_weight * pitch_roll_penalty
        action_penalty = -self.torque_penalty_weight * float(action @ action)
        fall_penalty = -self.fall_penalty if fell else 0.0
        clipped_progress = min(
            max(float(gait_step_progress), 0.0),
            max(self.gait_step_progress_clip, 0.0),
        )
        gait_step_progress_reward = self.gait_step_progress_weight * clipped_progress
        stance_slip_penalty = -self.stance_slip_penalty_weight * max(
            float(stance_slip_speed), 0.0
        )
        return {
            "alive": float(alive_reward),
            "velocity": float(velocity_reward),
            "orientation": float(orientation_penalty),
            "action": float(action_penalty),
            "fall": float(fall_penalty),
            "gait_step_progress": float(gait_step_progress_reward),
            "stance_slip": float(stance_slip_penalty),
        }
