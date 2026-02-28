from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p


@dataclass
class WalkerDynamics:
    max_force: float = 40.0

    def apply_action(self, body_id: int, action: np.ndarray) -> None:
        fx = float(np.clip(action[0], -1.0, 1.0) * self.max_force)
        fy = float(np.clip(action[1], -1.0, 1.0) * self.max_force)
        torque_z = float(np.clip(action[2], -1.0, 1.0) * 2.0)
        p.applyExternalForce(body_id, -1, [fx, fy, 0.0], [0.0, 0.0, 0.0], p.WORLD_FRAME)
        p.applyExternalTorque(body_id, -1, [0.0, 0.0, torque_z], p.WORLD_FRAME)
