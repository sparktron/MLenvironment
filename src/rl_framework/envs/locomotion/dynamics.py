from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p


@dataclass(frozen=True)
class JointSpec:
    """Per-joint range, rest pose, and torque cap."""

    name: str
    low: float  # joint angle lower limit (rad)
    high: float  # joint angle upper limit (rad)
    rest: float  # neutral / "crouch" target pose
    max_torque: float  # actuator force cap (N·m)


# Joint order matches WalkerBulletEnv link order:
#   0 rHip   1 rKnee  2 rAnkle  3 lHip   4 lKnee  5 lAnkle
#   6 rShoulder  7 rElbow  8 lShoulder  9 lElbow
# Link 10 (head) is JOINT_FIXED and is NOT in this list.
JOINT_SPECS: tuple[JointSpec, ...] = (
    # Atlas DRC-class peak torques (N·m). Source: atlas_v3.urdf joint <limit>
    # effort values, restricted to the sagittal-plane joints we model here.
    # Hip pitch and knee carry the body; ankle stabilises foot pitch.
    JointSpec("rHip", -1.2, 1.5, 0.00, 190.0),
    JointSpec("rKnee", 0.0, 2.3, 0.30, 220.0),
    JointSpec("rAnkle", -0.6, 0.6, -0.15, 100.0),
    JointSpec("lHip", -1.2, 1.5, 0.00, 190.0),
    JointSpec("lKnee", 0.0, 2.3, 0.30, 220.0),
    JointSpec("lAnkle", -0.6, 0.6, -0.15, 100.0),
    JointSpec("rShoulder", -2.0, 2.0, 0.00, 90.0),
    JointSpec("rElbow", 0.0, 2.3, 0.20, 100.0),
    JointSpec("lShoulder", -2.0, 2.0, 0.00, 90.0),
    JointSpec("lElbow", 0.0, 2.3, 0.20, 100.0),
)
NUM_JOINTS = len(JOINT_SPECS)
JOINT_INDICES = list(range(NUM_JOINTS))

# Pre-computed arrays for the hot path.
_LOWS = np.array([s.low for s in JOINT_SPECS], dtype=np.float32)
_HIGHS = np.array([s.high for s in JOINT_SPECS], dtype=np.float32)
_TORQUE_CAPS = np.array([s.max_torque for s in JOINT_SPECS], dtype=np.float32)
REST_POSE = np.array([s.rest for s in JOINT_SPECS], dtype=np.float32)
# How far each joint can deviate from REST per ±1 action unit.
# Action=0 → rest pose; action=±1 → rest ± ACTION_SCALE (clipped to limits).
ACTION_SCALE = np.array(
    [
        1.2,
        1.2,
        0.5,
        1.2,
        1.2,
        0.5,  # legs (hip/knee/ankle × 2)
        1.2,
        1.0,
        1.2,
        1.0,
    ],  # arms (shoulder/elbow × 2)
    dtype=np.float32,
)


@dataclass
class WalkerDynamics:
    """Actuator model. Defaults to PD position control on all 10 joints.

    Legacy ``max_torque`` arg is reinterpreted as a global torque scale
    (``max_torque / 35``) so existing YAMLs continue to work without
    over-driving the new per-joint torque caps.
    """

    max_torque: float = 35.0
    control_mode: str = "pd"  # "pd" or "torque"
    # PyBullet POSITION_CONTROL gains. Empirically tuned: stronger gains cause
    # a lever-jack effect (ankle PD pushing body off the ground); weaker gains
    # let the body sag through rest pose. These values keep z(0)≈TORSO_STAND_Z
    # and give ~100 steps of grace before passive fall under zero action —
    # enough for PPO to bootstrap a balance policy.
    position_gain: float = 0.1
    velocity_gain: float = 1.0

    def __post_init__(self) -> None:
        self._pos_gains: list[float] = [self.position_gain] * NUM_JOINTS
        self._vel_gains: list[float] = [self.velocity_gain] * NUM_JOINTS
        self._torque_caps_list: list[float] = (
            _TORQUE_CAPS * self.torque_scale
        ).tolist()

    @property
    def torque_scale(self) -> float:
        return float(self.max_torque) / 35.0

    def apply_action(
        self, body_id: int, action: np.ndarray, physicsClientId: int = 0
    ) -> None:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        if self.control_mode == "pd":
            targets = np.clip(REST_POSE + a * ACTION_SCALE, _LOWS, _HIGHS).tolist()
            p.setJointMotorControlArray(
                body_id,
                JOINT_INDICES,
                controlMode=p.POSITION_CONTROL,
                targetPositions=targets,
                positionGains=self._pos_gains,
                velocityGains=self._vel_gains,
                forces=self._torque_caps_list,
                physicsClientId=physicsClientId,
            )
        else:
            torques = (a * _TORQUE_CAPS * self.torque_scale).tolist()
            p.setJointMotorControlArray(
                body_id,
                JOINT_INDICES,
                controlMode=p.TORQUE_CONTROL,
                forces=torques,
                physicsClientId=physicsClientId,
            )

    def hold_rest_pose(self, body_id: int, physicsClientId: int = 0) -> None:
        """Command PD targets at the rest pose. Used during the settle phase."""
        p.setJointMotorControlArray(
            body_id,
            JOINT_INDICES,
            controlMode=p.POSITION_CONTROL,
            targetPositions=REST_POSE.tolist(),
            positionGains=self._pos_gains,
            velocityGains=self._vel_gains,
            forces=self._torque_caps_list,
            physicsClientId=physicsClientId,
        )

    def disable_velocity_motors(self, body_id: int, physicsClientId: int = 0) -> None:
        """Zero the default velocity-control motor on every joint. Required
        before TORQUE_CONTROL so the implicit motor doesn't fight commanded
        torques. Harmless under POSITION_CONTROL."""
        p.setJointMotorControlArray(
            body_id,
            JOINT_INDICES,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0] * NUM_JOINTS,
            physicsClientId=physicsClientId,
        )
