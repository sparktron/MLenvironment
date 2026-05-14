from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p

# Joint order: rHip, rKnee, rAnkle, lHip, lKnee, lAnkle
NUM_JOINTS = 6
_JOINT_INDICES = list(range(NUM_JOINTS))


@dataclass
class WalkerDynamics:
    max_torque: float = 40.0

    def apply_action(self, body_id: int, action: np.ndarray, physicsClientId: int = 0) -> None:
        torques = [float(np.clip(action[i], -1.0, 1.0) * self.max_torque) for i in range(NUM_JOINTS)]
        p.setJointMotorControlArray(
            body_id,
            _JOINT_INDICES,
            controlMode=p.TORQUE_CONTROL,
            forces=torques,
            physicsClientId=physicsClientId,
        )

    def disable_velocity_motors(self, body_id: int, physicsClientId: int = 0) -> None:
        """Zero out the default velocity-control motor on every joint so pure torque control works."""
        p.setJointMotorControlArray(
            body_id,
            _JOINT_INDICES,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0] * NUM_JOINTS,
            physicsClientId=physicsClientId,
        )
