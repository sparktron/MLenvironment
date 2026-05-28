from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class WalkerReward:
    """Shaped reward for the bipedal walker.

    Design notes (vs. the original v-shape velocity term):
    - ``alive_bonus`` dominates: surviving a step is worth more than walking
      perfectly at the target velocity, so the agent's first lesson is "don't
      fall." We only add the bonus while ``alive=True``.
    - The velocity reward is a **Gaussian around the target**, so standing
      still scores a small positive (≈ alive_bonus + 0.3) instead of being
      penalised harder than falling.
    - A one-time ``fall_penalty`` fires only on the terminal step of a fall
      (not on truncation), giving a clear gradient toward staying upright.
    - ``orientation_penalty_weight`` is scaled down to make the early
      bootstrap easier; raise it later via curriculum.
    """

    alive_bonus: float = 5.0
    forward_velocity_weight: float = 1.5
    target_velocity: float = 1.0
    # Width (1 sigma) of the Gaussian around target_velocity. Larger →
    # broader credit for "close to target" speeds.
    velocity_sigma: float = 0.5
    orientation_penalty_weight: float = 0.3
    torque_penalty_weight: float = 0.01
    # One-time penalty applied on the step a fall is detected.
    fall_penalty: float = 10.0

    def compute(
        self,
        lin_vel_x: float,
        pitch_roll_penalty: float,
        action: np.ndarray,
        alive: bool,
        fell: bool = False,
    ) -> float:
        reward = 0.0
        if alive:
            reward += self.alive_bonus
        # Gaussian velocity reward: 1.0 at v=target, decays smoothly outside.
        diff = (lin_vel_x - self.target_velocity) / max(self.velocity_sigma, 1e-6)
        reward += self.forward_velocity_weight * math.exp(-0.5 * diff * diff)
        reward -= self.orientation_penalty_weight * pitch_roll_penalty
        reward -= self.torque_penalty_weight * float(action @ action)
        if fell:
            reward -= self.fall_penalty
        return float(reward)
