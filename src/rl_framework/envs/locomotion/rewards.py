from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WalkerReward:
    alive_bonus: float = 1.0
    forward_velocity_weight: float = 2.0
    target_velocity: float = 1.0
    orientation_penalty_weight: float = 1.0
    torque_penalty_weight: float = 0.01

    def compute(self, lin_vel_x: float, pitch_roll_penalty: float, action: np.ndarray, alive: bool) -> float:
        reward = 0.0
        if alive:
            reward += self.alive_bonus
        reward += self.forward_velocity_weight * (1.0 - abs(self.target_velocity - lin_vel_x))
        reward -= self.orientation_penalty_weight * pitch_roll_penalty
        reward -= self.torque_penalty_weight * float(np.square(action).sum())
        return float(reward)
