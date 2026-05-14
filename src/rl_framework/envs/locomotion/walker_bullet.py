from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from rl_framework.envs.locomotion.dynamics import WalkerDynamics
from rl_framework.envs.locomotion.rewards import WalkerReward
from rl_framework.envs.locomotion.terminations import WalkerTermination


class WalkerBulletEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.render_mode = cfg.get("render_mode")
        self._connection = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
        try:
            self._rng = np.random.default_rng(cfg.get("seed", 0))

            sim_cfg = cfg.get("sim", {})
            # Accept both max_torque and legacy max_force keys.
            max_torque = sim_cfg.get("max_torque", sim_cfg.get("max_force", 40.0))
            self.dynamics = WalkerDynamics(max_torque=max_torque)
            reward_cfg = cfg.get("reward", {})
            self.reward_fn = WalkerReward(**{k: v for k, v in reward_cfg.items() if k in WalkerReward.__annotations__})
            term_cfg = cfg.get("termination", {})
            self.termination = WalkerTermination(**{k: v for k, v in term_cfg.items() if k in WalkerTermination.__annotations__})

            # obs: pos(3) + quat(4) + lin_vel(3) + ang_vel(3) + joint_pos(6) + joint_vel(6) = 25
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)
            # action: torque for each of the 6 joints (rHip, rKnee, rAnkle, lHip, lKnee, lAnkle)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

            # Domain randomisation: sensor noise
            rand_cfg = cfg.get("domain_randomization", {})
            self._sensor_noise_std = float(rand_cfg.get("sensor_noise_std", 0.0))

            # Domain randomisation: action latency
            self._action_latency_steps = int(rand_cfg.get("action_latency_steps", 0))
            self._action_buffer: deque[np.ndarray] = deque()

            self.step_count = 0
            self.robot_id = -1
        except Exception:
            p.disconnect(self._connection)
            raise

    # Geometry constants (half-extents in metres)
    _TORSO_H  = [0.14,  0.09,  0.17 ]
    _THIGH_H  = [0.055, 0.055, 0.125]
    _SHIN_H   = [0.045, 0.045, 0.115]
    _FOOT_H   = [0.085, 0.05,  0.028]
    _LEG_DX   = 0.085  # lateral hip offset
    # Torso COM height when both feet rest flat on the ground:
    #   ankle_z = foot_h[2], knee = ankle + 2*shin_h[2], hip = knee + 2*thigh_h[2]
    #   torso_z = hip + torso_h[2]
    TORSO_STAND_Z = _FOOT_H[2] + 2*_SHIN_H[2] + 2*_THIGH_H[2] + _TORSO_H[2]  # ≈ 0.678 m

    def _build_world(self) -> None:
        cid = self._connection
        p.resetSimulation(physicsClientId=cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.cfg.get("sim", {}).get("gravity", -9.81), physicsClientId=cid)
        p.loadURDF("plane.urdf", physicsClientId=cid)

        sim  = self.cfg.get("sim", {})
        mass = sim.get("mass", 3.0)
        fric = sim.get("friction", 0.8)

        th, sh, fh = self._THIGH_H, self._SHIN_H, self._FOOT_H
        dx = self._LEG_DX

        def _col(half):
            return p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)

        def _vis(half, color, offset=(0, 0, 0)):
            return p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color,
                                       visualFramePosition=list(offset), physicsClientId=cid)

        GRAY  = [0.45, 0.45, 0.45, 1.0]
        RED   = [0.80, 0.15, 0.15, 1.0]
        BLUE  = [0.15, 0.15, 0.80, 1.0]
        DGRAY = [0.30, 0.30, 0.30, 1.0]

        torso_col = _col(self._TORSO_H)
        torso_vis = _vis(self._TORSO_H, GRAY)

        # Thigh/shin visuals hang down from their joint (top of segment = joint).
        # Foot visuals are centered at ankle height, shifted 4 cm forward.
        r_thigh_vis = _vis(th, RED,  (0,    0, -th[2]))
        r_shin_vis  = _vis(sh, BLUE, (0,    0, -sh[2]))
        r_foot_col  = _col(fh)
        r_foot_vis  = _vis(fh, DGRAY, (0.04, 0, 0))
        l_thigh_vis = _vis(th, RED,  (0,    0, -th[2]))
        l_shin_vis  = _vis(sh, BLUE, (0,    0, -sh[2]))
        l_foot_col  = _col(fh)
        l_foot_vis  = _vis(fh, DGRAY, (0.04, 0, 0))

        ident = [0, 0, 0, 1]
        self.robot_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=torso_col,
            baseVisualShapeIndex=torso_vis,
            # Leg links: give each a small mass so PyBullet integrates them properly.
            linkMasses=[0.15, 0.10, 0.05,  0.15, 0.10, 0.05],
            linkCollisionShapeIndices=[-1, -1, r_foot_col, -1, -1, l_foot_col],
            linkVisualShapeIndices=[r_thigh_vis, r_shin_vis, r_foot_vis,
                                    l_thigh_vis, l_shin_vis, l_foot_vis],
            # Joint positions relative to parent joint (or base COM for hip joints).
            linkPositions=[
                [ dx, 0, -self._TORSO_H[2]],   # 0: right hip
                [  0, 0, -2*th[2]          ],   # 1: right knee
                [  0, 0, -2*sh[2]          ],   # 2: right ankle
                [-dx, 0, -self._TORSO_H[2]],   # 3: left hip
                [  0, 0, -2*th[2]          ],   # 4: left knee
                [  0, 0, -2*sh[2]          ],   # 5: left ankle
            ],
            linkOrientations=[ident] * 6,
            # Inertial frame (COM) relative to joint — matches visual centre.
            linkInertialFramePositions=[
                [0,    0, -th[2]], [0,    0, -sh[2]], [0.04, 0, 0],
                [0,    0, -th[2]], [0,    0, -sh[2]], [0.04, 0, 0],
            ],
            linkInertialFrameOrientations=[ident] * 6,
            # 0 = base, 1 = link-0 (right thigh), …
            linkParentIndices=[0, 1, 2, 0, 4, 5],
            # Revolute around Y-axis → sagittal-plane swing for all joints.
            linkJointTypes=[p.JOINT_REVOLUTE] * 6,
            linkJointAxis=[[0, 1, 0]] * 6,
            physicsClientId=cid,
        )
        # Friction on base and feet.
        p.changeDynamics(self.robot_id, -1, lateralFriction=fric, physicsClientId=cid)
        p.changeDynamics(self.robot_id,  2, lateralFriction=fric, physicsClientId=cid)  # right foot
        p.changeDynamics(self.robot_id,  5, lateralFriction=fric, physicsClientId=cid)  # left foot
        # Joint damping: light resistance prevents wild oscillation without dominating the policy.
        for j in range(6):
            p.changeDynamics(self.robot_id, j, jointDamping=0.5, physicsClientId=cid)
        # Disable the default velocity-hold motor so pure torque control works.
        self.dynamics.disable_velocity_motors(self.robot_id, physicsClientId=cid)

    def _get_obs(self) -> np.ndarray:
        pos, quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._connection)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self._connection)
        joint_states = p.getJointStates(self.robot_id, list(range(6)), physicsClientId=self._connection)
        joint_pos = [s[0] for s in joint_states]
        joint_vel = [s[1] for s in joint_states]
        obs = np.array([*pos, *quat, *lin_vel, *ang_vel, *joint_pos, *joint_vel], dtype=np.float32)
        if self._sensor_noise_std > 0.0:
            obs = obs + self._rng.normal(0.0, self._sensor_noise_std, size=obs.shape).astype(np.float32)
        return obs

    def _apply_domain_randomization(self) -> None:
        rand_cfg = self.cfg.get("domain_randomization", {})
        mass_rng = rand_cfg.get("mass_scale_range", [1.0, 1.0])
        fric_rng = rand_cfg.get("friction_range", [1.0, 1.0])
        base_mass = self.cfg.get("sim", {}).get("mass", 3.0)
        mass = base_mass * float(self._rng.uniform(mass_rng[0], mass_rng[1]))
        friction = self.cfg.get("sim", {}).get("friction", 0.8) * float(self._rng.uniform(fric_rng[0], fric_rng[1]))
        p.changeDynamics(self.robot_id, -1, mass=mass, lateralFriction=friction, physicsClientId=self._connection)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._build_world()
        self._apply_domain_randomization()
        self.step_count = 0
        # Reset action latency buffer: pre-fill with zero actions so the first
        # _action_latency_steps steps apply no-op while the buffer warms up.
        self._action_buffer.clear()
        noop = np.zeros(self.action_space.shape, dtype=np.float32)
        for _ in range(self._action_latency_steps):
            self._action_buffer.append(noop)

        reset_cfg = self.cfg.get("reset_randomization", {})
        pos_noise = reset_cfg.get("position_xy_noise", 0.02)
        yaw_noise = reset_cfg.get("yaw_noise", 0.1)
        start_pos = [float(self._rng.uniform(-pos_noise, pos_noise)), float(self._rng.uniform(-pos_noise, pos_noise)), self.TORSO_STAND_Z]
        yaw = float(self._rng.uniform(-yaw_noise, yaw_noise))
        quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(self.robot_id, start_pos, quat, physicsClientId=self._connection)
        for j in range(6):
            p.resetJointState(self.robot_id, j, targetValue=0.0, targetVelocity=0.0,
                              physicsClientId=self._connection)
        # resetJointState re-engages the internal motor; kill it again.
        self.dynamics.disable_velocity_motors(self.robot_id, physicsClientId=self._connection)

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        # Action latency: buffer the current action and apply the delayed one.
        if self._action_latency_steps > 0:
            self._action_buffer.append(action)
            action = self._action_buffer.popleft()
        self.dynamics.apply_action(self.robot_id, action, physicsClientId=self._connection)
        p.stepSimulation(physicsClientId=self._connection)
        self.step_count += 1

        obs = self._get_obs()
        pos, quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._connection)
        roll, pitch, _ = p.getEulerFromQuaternion(quat)
        lin_vel, _ = p.getBaseVelocity(self.robot_id, physicsClientId=self._connection)

        terminated, truncated = self.termination.check(pos[2], roll, pitch, self.step_count)
        reward = self.reward_fn.compute(lin_vel[0], abs(roll) + abs(pitch), action, not terminated)
        info = {"x_position": pos[0], "lin_vel_x": lin_vel[0]}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self.robot_id >= 0:
                pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._connection)
                target = [pos[0], pos[1], self.TORSO_STAND_Z * 0.5]
            else:
                target = [0.0, 0.0, 0.35]
            view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target,
                distance=2.2, yaw=45, pitch=-20, roll=0, upAxisIndex=2,
            )
            proj = p.computeProjectionMatrixFOV(fov=60, aspect=640 / 480, nearVal=0.1, farVal=100)
            _, _, px, _, _ = p.getCameraImage(640, 480, viewMatrix=view,
                                              projectionMatrix=proj, physicsClientId=self._connection)
            img = np.array(px, dtype=np.uint8).reshape(480, 640, 4)
            return img[:, :, :3]
        return None

    def close(self) -> None:
        if p.isConnected(self._connection):
            p.disconnect(self._connection)
