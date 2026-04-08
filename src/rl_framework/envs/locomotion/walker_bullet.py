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
        self._rng = np.random.default_rng(cfg.get("seed", 0))

        self.dynamics = WalkerDynamics(max_force=cfg.get("sim", {}).get("max_force", 40.0))
        reward_cfg = cfg.get("reward", {})
        self.reward_fn = WalkerReward(**{k: v for k, v in reward_cfg.items() if k in WalkerReward.__annotations__})
        term_cfg = cfg.get("termination", {})
        self.termination = WalkerTermination(**{k: v for k, v in term_cfg.items() if k in WalkerTermination.__annotations__})

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Domain randomisation: sensor noise
        rand_cfg = cfg.get("domain_randomization", {})
        self._sensor_noise_std = float(rand_cfg.get("sensor_noise_std", 0.0))

        # Domain randomisation: action latency
        self._action_latency_steps = int(rand_cfg.get("action_latency_steps", 0))
        self._action_buffer: deque[np.ndarray] = deque()

        self.step_count = 0
        self.robot_id = -1

    def _build_world(self) -> None:
        cid = self._connection
        p.resetSimulation(physicsClientId=cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.cfg.get("sim", {}).get("gravity", -9.81), physicsClientId=cid)
        p.loadURDF("plane.urdf", physicsClientId=cid)
        size = self.cfg.get("sim", {}).get("body_half_extents", [0.2, 0.1, 0.08])
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=size, physicsClientId=cid)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.2, 0.6, 0.9, 1.0], physicsClientId=cid)
        mass = self.cfg.get("sim", {}).get("mass", 3.0)
        self.robot_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual, physicsClientId=cid)
        friction = self.cfg.get("sim", {}).get("friction", 0.8)
        p.changeDynamics(self.robot_id, -1, lateralFriction=friction, physicsClientId=cid)

    def _get_obs(self) -> np.ndarray:
        pos, quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._connection)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self._connection)
        obs = np.array([*pos, *quat, *lin_vel, *ang_vel], dtype=np.float32)
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
        start_pos = [float(self._rng.uniform(-pos_noise, pos_noise)), float(self._rng.uniform(-pos_noise, pos_noise)), 0.25]
        yaw = float(self._rng.uniform(-yaw_noise, yaw_noise))
        quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(self.robot_id, start_pos, quat, physicsClientId=self._connection)

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
            _, _, px, _, _ = p.getCameraImage(width=640, height=480, physicsClientId=self._connection)
            img = np.array(px, dtype=np.uint8).reshape(480, 640, 4)
            return img[:, :, :3]  # RGBA -> RGB
        return None

    def close(self) -> None:
        if p.isConnected(self._connection):
            p.disconnect(self._connection)
