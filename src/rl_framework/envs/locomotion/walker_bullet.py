from __future__ import annotations

from collections import deque
from typing import Any
import warnings

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from rl_framework.envs.locomotion.dynamics import (
    JOINT_INDICES,
    JOINT_SPECS,
    REST_POSE,
    WalkerDynamics,
)
from rl_framework.envs.locomotion.rewards import WalkerReward
from rl_framework.envs.locomotion.terminations import WalkerTermination
from rl_framework.utils.config_merge import get_section


class WalkerBulletEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.render_mode = cfg.get("render_mode")
        self._connection = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
        try:
            self._rng = np.random.default_rng(cfg.get("seed", 0))

            # get_section everywhere: the GUI wizard sometimes writes
            # `key: null` for nested groups, which would otherwise crash
            # .get(...) chains (a present key with value None is returned as
            # None, not the .get(..., {}) default).
            sim_cfg = get_section(cfg, "sim")
            # Accept both max_torque and legacy max_force keys.
            max_torque = sim_cfg.get("max_torque", sim_cfg.get("max_force", 35.0))
            ctrl_cfg = get_section(sim_cfg, "control")
            self.dynamics = WalkerDynamics(
                max_torque=max_torque,
                control_mode=str(ctrl_cfg.get("mode", "pd")),
                position_gain=float(ctrl_cfg.get("position_gain", 0.1)),
                velocity_gain=float(ctrl_cfg.get("velocity_gain", 1.0)),
            )
            # Physics timestep + how many sim steps per agent step.
            # Default: 240 Hz physics, 4× repeat → 60 Hz control.
            self._sim_timestep = float(sim_cfg.get("timestep", 1.0 / 240.0))
            self._frame_skip = int(sim_cfg.get("frame_skip", 4))
            # How many sim steps to run with the PD holding the rest pose
            # immediately after reset, so the robot is at equilibrium before
            # the first agent observation.
            self._settle_steps = int(sim_cfg.get("settle_steps", 30))
            reward_cfg = get_section(cfg, "reward")
            self._warn_unknown_section_keys(reward_cfg, WalkerReward, "reward")
            self.reward_fn = WalkerReward(
                **{
                    k: v
                    for k, v in reward_cfg.items()
                    if k in WalkerReward.__annotations__
                }
            )
            term_cfg = get_section(cfg, "termination")
            self._warn_unknown_section_keys(term_cfg, WalkerTermination, "termination")
            self.termination = WalkerTermination(
                **{
                    k: v
                    for k, v in term_cfg.items()
                    if k in WalkerTermination.__annotations__
                }
            )

            obs_cfg = get_section(cfg, "observation")
            self._observation_version = str(obs_cfg.get("version", "v1"))
            self._coordinate_free = bool(obs_cfg.get("coordinate_free", False))
            if self._observation_version == "v1" and self._coordinate_free:
                raise ValueError("observation.coordinate_free requires observation.version: v2")
            # v1: pos(3)+quat(4)+lin_vel(3)+ang_vel(3)+joints(20)+DR(2) = 35
            # v2 adds binary right/left foot contacts. Coordinate-free v2
            # drops global x/y position, keeping height as the useful local cue.
            self._obs_size = 35 + (2 if self._observation_version == "v2" else 0)
            if self._coordinate_free:
                self._obs_size -= 2
            # joints: rHip rKnee rAnkle lHip lKnee lAnkle rShoulder rElbow lShoulder lElbow
            # The DR scales make randomization observable to the policy; without
            # them, random mass/friction look like pure noise from the agent's
            # perspective and actively hurt training.
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._obs_size,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(10,), dtype=np.float32
            )

            # Domain randomisation: sensor noise
            rand_cfg = get_section(cfg, "domain_randomization")
            self._sensor_noise_std = float(rand_cfg.get("sensor_noise_std", 0.0))

            # Domain randomisation: action latency
            self._action_latency_steps = int(rand_cfg.get("action_latency_steps", 0))
            self._action_buffer: deque[np.ndarray] = deque()

            terrain_cfg = get_section(cfg, "terrain")
            self._terrain_preset = str(terrain_cfg.get("preset", "flat"))
            self._terrain_body_ids: list[int] = []
            push_cfg = get_section(terrain_cfg, "push_recovery")
            self._push_interval_steps = int(push_cfg.get("interval_steps", 0))
            self._push_force = float(push_cfg.get("force", 0.0))
            self._push_start_step = int(push_cfg.get("start_step", 0))

            self.step_count = 0
            self.robot_id = -1
            self.plane_id = -1
            # DR scales — overwritten on each reset.
            self._mass_scale = 1.0
            self._friction_scale = 1.0
            # Nominal masses captured after createMultiBody; domain
            # randomization scales each link from its own baseline.
            self._nominal_masses: dict[int, float] = {}
            # Pre-allocated obs buffer; _get_obs() writes in-place and returns
            # a copy so VecEnv cannot mutate the buffer through the returned array.
            self._obs_buf = np.zeros(self._obs_size, dtype=np.float32)
        except Exception:
            p.disconnect(self._connection)
            raise

    # Geometry constants (half-extents in metres)
    _TORSO_H = [0.14, 0.09, 0.17]
    _THIGH_H = [0.055, 0.055, 0.125]
    _SHIN_H = [0.045, 0.045, 0.115]
    _FOOT_H = [0.085, 0.05, 0.028]
    _UPPER_ARM_H = [0.04, 0.04, 0.11]
    _FORE_ARM_H = [0.035, 0.035, 0.09]
    _HEAD_H = [0.08, 0.07, 0.10]
    _LEG_DY = 0.085  # lateral (Y) hip offset — legs are side-by-side along Y
    _DYNAMIC_LINK_IDS = [-1, *range(11)]
    _FRICTION_LINK_IDS = [-1, 2, 5]
    # Torso COM height when both feet rest flat on the ground:
    #   ankle_z = foot_h[2], knee = ankle + 2*shin_h[2], hip = knee + 2*thigh_h[2]
    #   torso_z = hip + torso_h[2]
    TORSO_STAND_Z = (
        _FOOT_H[2] + 2 * _SHIN_H[2] + 2 * _THIGH_H[2] + _TORSO_H[2]
    )  # ≈ 0.678 m

    @staticmethod
    def _warn_unknown_section_keys(section: dict[str, Any], cls: type, name: str) -> None:
        unknown = sorted(set(section) - set(cls.__annotations__))
        if unknown:
            warnings.warn(
                f"Ignoring unknown {name} keys {unknown}; valid keys are "
                f"{sorted(cls.__annotations__)}",
                stacklevel=3,
            )

    def _build_world(self) -> None:
        cid = self._connection
        p.resetSimulation(physicsClientId=cid)
        p.setTimeStep(self._sim_timestep, physicsClientId=cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        sim_cfg = get_section(self.cfg, "sim")
        p.setGravity(0, 0, sim_cfg.get("gravity", -9.81), physicsClientId=cid)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=cid)
        self._terrain_body_ids = []
        self._build_terrain()

        sim = sim_cfg
        mass = sim.get("mass", 3.0)
        fric = sim.get("friction", 0.8)

        th, sh, fh = self._THIGH_H, self._SHIN_H, self._FOOT_H
        ua, fa, hd = self._UPPER_ARM_H, self._FORE_ARM_H, self._HEAD_H
        leg_dy = self._LEG_DY
        arm_dy = self._TORSO_H[1] + ua[1]  # torso half-width + upper-arm half-width

        def _col(half):
            return p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half, physicsClientId=cid
            )

        def _vis(half, color, offset=(0, 0, 0)):
            return p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half,
                rgbaColor=color,
                visualFramePosition=list(offset),
                physicsClientId=cid,
            )

        GRAY = [0.45, 0.45, 0.45, 1.0]
        LGRAY = [0.60, 0.60, 0.60, 1.0]  # lighter gray for head & forearms
        RED = [0.80, 0.15, 0.15, 1.0]
        BLUE = [0.15, 0.15, 0.80, 1.0]
        DGRAY = [0.30, 0.30, 0.30, 1.0]

        torso_col = _col(self._TORSO_H)
        torso_vis = _vis(self._TORSO_H, GRAY)

        # ── Legs ──────────────────────────────────────────────────────────────
        # Thigh/shin visuals hang down from their joint (top of segment = joint).
        # Foot visuals are centered at ankle height, shifted 4 cm forward.
        r_thigh_vis = _vis(th, RED, (0, 0, -th[2]))
        r_shin_vis = _vis(sh, BLUE, (0, 0, -sh[2]))
        r_foot_col = _col(fh)
        r_foot_vis = _vis(fh, DGRAY, (0.04, 0, 0))
        l_thigh_vis = _vis(th, RED, (0, 0, -th[2]))
        l_shin_vis = _vis(sh, BLUE, (0, 0, -sh[2]))
        l_foot_col = _col(fh)
        l_foot_vis = _vis(fh, DGRAY, (0.04, 0, 0))

        # ── Arms ──────────────────────────────────────────────────────────────
        # Upper arms hang from shoulder joint; forearms hang from elbow joint.
        r_ua_vis = _vis(ua, GRAY, (0, 0, -ua[2]))
        r_fa_vis = _vis(fa, LGRAY, (0, 0, -fa[2]))
        l_ua_vis = _vis(ua, GRAY, (0, 0, -ua[2]))
        l_fa_vis = _vis(fa, LGRAY, (0, 0, -fa[2]))

        # ── Head ──────────────────────────────────────────────────────────────
        # Neck joint sits at the top of the torso; head box rises above it.
        head_vis = _vis(hd, LGRAY, (0, 0, hd[2]))

        ident = [0, 0, 0, 1]
        shoulder_z = self._TORSO_H[2] - 0.04  # slightly below top of torso

        # Link order (11 total):
        #  0 rThigh  1 rShin  2 rFoot  3 lThigh  4 lShin  5 lFoot
        #  6 rUpperArm  7 rForearm  8 lUpperArm  9 lForearm  10 head
        #
        # linkParentIndices use 1-based link indexing (0 = base).
        self.robot_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=torso_col,
            baseVisualShapeIndex=torso_vis,
            # Atlas DRC-class mass distribution (kg). Source: atlas_v3.urdf
            # released by Boston Dynamics for the DARPA Robotics Challenge sim.
            # Total = 28 (torso, set via baseMass) + 1.5 (head) + 14 (thighs)
            #       + 7.4 (shins) + 4 (feet) + 6 (upper arms) + 5 (forearms)
            #       ≈ 65.9 kg (hardware Atlas is ~82 kg incl. battery/hands).
            linkMasses=[
                7.0,
                3.7,
                2.0,  # right leg: thigh, shin, foot
                7.0,
                3.7,
                2.0,  # left leg
                3.0,
                2.5,  # right arm: upper, forearm
                3.0,
                2.5,  # left arm
                1.5,  # head
            ],
            linkCollisionShapeIndices=[
                -1,
                -1,
                r_foot_col,  # right leg (only foot collides)
                -1,
                -1,
                l_foot_col,  # left leg
                -1,
                -1,  # right arm (no collision)
                -1,
                -1,  # left arm
                -1,  # head
            ],
            linkVisualShapeIndices=[
                r_thigh_vis,
                r_shin_vis,
                r_foot_vis,
                l_thigh_vis,
                l_shin_vis,
                l_foot_vis,
                r_ua_vis,
                r_fa_vis,
                l_ua_vis,
                l_fa_vis,
                head_vis,
            ],
            linkPositions=[
                # ── Legs: hips attach at bottom of torso, separated along ±Y ──
                [0, -leg_dy, -self._TORSO_H[2]],  # 0 right hip
                [0, 0, -2 * th[2]],  # 1 right knee
                [0, 0, -2 * sh[2]],  # 2 right ankle
                [0, leg_dy, -self._TORSO_H[2]],  # 3 left hip
                [0, 0, -2 * th[2]],  # 4 left knee
                [0, 0, -2 * sh[2]],  # 5 left ankle
                # ── Arms: shoulders attach at upper sides of torso, ±Y ──
                [0, -arm_dy, shoulder_z],  # 6 right shoulder
                [0, 0, -2 * ua[2]],  # 7 right elbow
                [0, arm_dy, shoulder_z],  # 8 left shoulder
                [0, 0, -2 * ua[2]],  # 9 left elbow
                # ── Head: neck at top of torso ──
                [0, 0, self._TORSO_H[2]],  # 10 neck/head
            ],
            linkOrientations=[ident] * 11,
            linkInertialFramePositions=[
                [0, 0, -th[2]],
                [0, 0, -sh[2]],
                [0.04, 0, 0],  # right leg
                [0, 0, -th[2]],
                [0, 0, -sh[2]],
                [0.04, 0, 0],  # left leg
                [0, 0, -ua[2]],
                [0, 0, -fa[2]],  # right arm
                [0, 0, -ua[2]],
                [0, 0, -fa[2]],  # left arm
                [0, 0, hd[2]],  # head
            ],
            linkInertialFrameOrientations=[ident] * 11,
            # 0=base, 1=link-0, 2=link-1, …
            linkParentIndices=[0, 1, 2, 0, 4, 5, 0, 7, 0, 9, 0],
            linkJointTypes=[p.JOINT_REVOLUTE] * 10 + [p.JOINT_FIXED],
            # All revolute joints rotate around Y → sagittal-plane forward/back swing.
            # Axis for fixed head joint is arbitrary.
            linkJointAxis=[[0, 1, 0]] * 10 + [[0, 0, 1]],
            physicsClientId=cid,
        )

        # Friction on base torso and both feet.
        p.changeDynamics(self.robot_id, -1, lateralFriction=fric, physicsClientId=cid)
        p.changeDynamics(
            self.robot_id, 2, lateralFriction=fric, physicsClientId=cid
        )  # right foot
        p.changeDynamics(
            self.robot_id, 5, lateralFriction=fric, physicsClientId=cid
        )  # left foot
        # Joint limits + light damping. Without limits, knees would hyperextend
        # backward, ankles would spin freely, etc. — the body becomes nonsensical.
        for j in JOINT_INDICES:
            spec = JOINT_SPECS[j]
            p.changeDynamics(
                self.robot_id,
                j,
                jointLowerLimit=spec.low,
                jointUpperLimit=spec.high,
                jointDamping=0.5,
                physicsClientId=cid,
            )
        # Under TORQUE_CONTROL the implicit velocity motor must be silenced.
        # Under POSITION_CONTROL (our default) it's harmless but cheap to do.
        self.dynamics.disable_velocity_motors(self.robot_id, physicsClientId=cid)
        self._nominal_masses = {
            link_id: float(
                p.getDynamicsInfo(self.robot_id, link_id, physicsClientId=cid)[0]
            )
            for link_id in self._DYNAMIC_LINK_IDS
        }

    def _build_terrain(self) -> None:
        """Add deterministic static terrain without changing the spawn surface."""
        if self._terrain_preset == "flat" or self._terrain_preset == "push_recovery":
            return

        cid = self._connection
        terrain_cfg = get_section(self.cfg, "terrain")
        friction = float(get_section(self.cfg, "sim").get("friction", 0.8))

        def add_box(position: tuple[float, float, float], half_extents: tuple[float, float, float]) -> None:
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=cid)
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.25, 0.45, 0.25, 1.0], physicsClientId=cid
            )
            body_id = p.createMultiBody(
                baseMass=0.0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
                basePosition=position, physicsClientId=cid,
            )
            p.changeDynamics(body_id, -1, lateralFriction=friction, physicsClientId=cid)
            self._terrain_body_ids.append(body_id)

        if self._terrain_preset == "uneven":
            height = float(terrain_cfg.get("height", 0.025))
            # Start after x=0.6 so reset randomization never begins in contact.
            for index, x in enumerate((0.8, 1.4, 2.0, 2.6, 3.2)):
                offset = height if index % 2 == 0 else -height
                add_box((x, 0.0, offset - 0.03), (0.32, 0.6, 0.03))
        elif self._terrain_preset == "obstacles":
            obstacle_height = float(terrain_cfg.get("obstacle_height", 0.10))
            for x in (1.2, 2.4, 3.6):
                add_box((x, 0.0, obstacle_height / 2), (0.10, 0.55, obstacle_height / 2))

    def _get_obs(
        self,
        pos=None,
        quat=None,
        lin_vel=None,
        ang_vel=None,
    ) -> np.ndarray:
        # Callers that have already queried PyBullet (e.g. step()) pass the
        # values in to avoid a second round-trip; reset() and render() let them
        # default to None and we query here for backward compatibility.
        if pos is None or quat is None:
            pos, quat = p.getBasePositionAndOrientation(
                self.robot_id, physicsClientId=self._connection
            )
        if lin_vel is None or ang_vel is None:
            lin_vel, ang_vel = p.getBaseVelocity(
                self.robot_id, physicsClientId=self._connection
            )
        joint_states = p.getJointStates(
            self.robot_id, JOINT_INDICES, physicsClientId=self._connection
        )
        values: list[float] = []
        values.extend((pos[2],) if self._coordinate_free else pos)
        values.extend(quat)
        values.extend(lin_vel)
        values.extend(ang_vel)
        values.extend(s[0] for s in joint_states)
        values.extend(s[1] for s in joint_states)
        values.extend((self._mass_scale, self._friction_scale))
        if self._observation_version == "v2":
            values.extend(
                float(bool(p.getContactPoints(bodyA=self.robot_id, linkIndexA=link, physicsClientId=self._connection)))
                for link in (2, 5)
            )
        self._obs_buf[:] = values
        if self._sensor_noise_std > 0.0:
            obs = self._obs_buf + self._rng.normal(
                0.0, self._sensor_noise_std, size=self._obs_size
            ).astype(np.float32)
        else:
            obs = self._obs_buf.copy()
        # Guard against PyBullet solver divergence poisoning VecNormalize stats.
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return obs

    def _apply_domain_randomization(self) -> None:
        rand_cfg = get_section(self.cfg, "domain_randomization")
        sim_cfg = get_section(self.cfg, "sim")
        mass_rng = rand_cfg.get("mass_scale_range", [1.0, 1.0])
        fric_rng = rand_cfg.get("friction_range", [1.0, 1.0])
        # Store the *scales* (not absolute values) so they're surfaced in the
        # observation as dimensionless multipliers centered on 1.0.
        self._mass_scale = float(self._rng.uniform(mass_rng[0], mass_rng[1]))
        self._friction_scale = float(self._rng.uniform(fric_rng[0], fric_rng[1]))
        friction = sim_cfg.get("friction", 0.8) * self._friction_scale
        for link_id, nominal_mass in self._nominal_masses.items():
            p.changeDynamics(
                self.robot_id,
                link_id,
                mass=nominal_mass * self._mass_scale,
                physicsClientId=self._connection,
            )
        for link_id in self._FRICTION_LINK_IDS:
            p.changeDynamics(
                self.robot_id,
                link_id,
                lateralFriction=friction,
                physicsClientId=self._connection,
            )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Build the world once; subsequent resets just rewind state. Saves
        # ~tens of ms per episode (no resetSimulation/loadURDF/createMultiBody).
        if self.robot_id < 0:
            self._build_world()
        self._apply_domain_randomization()
        self.step_count = 0
        # Reset action latency buffer: pre-fill with zero actions so the first
        # _action_latency_steps steps apply no-op while the buffer warms up.
        self._action_buffer.clear()
        noop = np.zeros(self.action_space.shape, dtype=np.float32)
        for _ in range(self._action_latency_steps):
            self._action_buffer.append(noop)

        reset_cfg = get_section(self.cfg, "reset_randomization")
        pos_noise = reset_cfg.get("position_xy_noise", 0.02)
        yaw_noise = reset_cfg.get("yaw_noise", 0.1)
        # +5 mm clearance avoids the knife-edge ground contact at z=0 (foot
        # bottom and plane top would otherwise coincide and produce solver pop).
        start_pos = [
            float(self._rng.uniform(-pos_noise, pos_noise)),
            float(self._rng.uniform(-pos_noise, pos_noise)),
            self.TORSO_STAND_Z + 0.005,
        ]
        yaw = float(self._rng.uniform(-yaw_noise, yaw_noise))
        quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(
            self.robot_id, start_pos, quat, physicsClientId=self._connection
        )
        # Zero out residual base velocity from the prior episode (resetBase­…
        # does not touch velocity, so a mid-fall robot would carry over).
        p.resetBaseVelocity(
            self.robot_id,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            physicsClientId=self._connection,
        )
        # Start in the slightly-crouched rest pose, not locked-straight pillars.
        for j in JOINT_INDICES:
            p.resetJointState(
                self.robot_id,
                j,
                targetValue=float(REST_POSE[j]),
                targetVelocity=0.0,
                physicsClientId=self._connection,
            )
        # resetJointState re-engages the implicit motor; silence it again
        # before issuing any PD or torque command.
        self.dynamics.disable_velocity_motors(
            self.robot_id, physicsClientId=self._connection
        )

        # Settle phase: let the robot reach equilibrium under PD before the
        # first observation, so step 0 isn't mid-fall from a transient.
        if self._settle_steps > 0:
            self.dynamics.hold_rest_pose(
                self.robot_id, physicsClientId=self._connection
            )
            for _ in range(self._settle_steps):
                p.stepSimulation(physicsClientId=self._connection)

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # Action latency: buffer the current action and apply the delayed one.
        if self._action_latency_steps > 0:
            self._action_buffer.append(action)
            action = self._action_buffer.popleft()
        # Frame-skip: hold the same command across `frame_skip` physics ticks
        # so the policy operates at ~60 Hz while physics runs at ~240 Hz.
        push_applied = False
        if (
            self._push_force > 0.0
            and self.step_count >= self._push_start_step
            and self._push_interval_steps > 0
            and self.step_count % self._push_interval_steps == 0
        ):
            direction = -1.0 if self._rng.integers(0, 2) else 1.0
            p.applyExternalForce(
                self.robot_id, -1, [0.0, direction * self._push_force, 0.0], [0.0, 0.0, 0.0],
                p.WORLD_FRAME, physicsClientId=self._connection,
            )
            push_applied = True
        self.dynamics.apply_action(
            self.robot_id, action, physicsClientId=self._connection
        )
        for _ in range(max(1, self._frame_skip)):
            p.stepSimulation(physicsClientId=self._connection)
        self.step_count += 1

        pos, quat = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self._connection
        )
        lin_vel, ang_vel = p.getBaseVelocity(
            self.robot_id, physicsClientId=self._connection
        )
        # Rare PyBullet solver divergence (stiff PD + contact impulse blowup)
        # can return NaN/Inf here. _get_obs() sanitizes its own copy, but pos/
        # lin_vel/quat below feed termination and reward directly — without
        # this guard a NaN reward poisons GAE for the whole episode's rollout
        # rows and corrupts the next PPO gradient update (NaN action means).
        diverged = not (
            np.all(np.isfinite(pos))
            and np.all(np.isfinite(quat))
            and np.all(np.isfinite(lin_vel))
            and np.all(np.isfinite(ang_vel))
        )
        if diverged:
            pos = tuple(np.nan_to_num(pos, nan=0.0, posinf=1e6, neginf=-1e6))
            quat = (0.0, 0.0, 0.0, 1.0)  # identity; avoids Euler conversion on garbage
            lin_vel = tuple(np.nan_to_num(lin_vel, nan=0.0, posinf=1e6, neginf=-1e6))
            ang_vel = tuple(np.nan_to_num(ang_vel, nan=0.0, posinf=1e6, neginf=-1e6))

        obs = self._get_obs(pos=pos, quat=quat, lin_vel=lin_vel, ang_vel=ang_vel)
        roll, pitch, _ = p.getEulerFromQuaternion(quat)

        # Torso (base, linkIndexA=-1) touching the floor = the robot has fallen.
        # Divergence counts as an immediate fall so a corrupted physics state
        # never lingers until max_steps truncation.
        torso_contact = diverged or bool(
            p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=self.plane_id,
                linkIndexA=-1,
                physicsClientId=self._connection,
            )
        )
        terminated, truncated = self.termination.check(
            pos[2], self.step_count, torso_contact
        )
        reward = self.reward_fn.compute(
            lin_vel_x=lin_vel[0],
            pitch_roll_penalty=abs(roll) + abs(pitch),
            action=action,
            alive=not terminated,
            fell=terminated,
        )
        info = {
            "x_position": pos[0],
            "lin_vel_x": lin_vel[0],
            "torso_contact": torso_contact,
            "push_applied": push_applied,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self.robot_id >= 0:
                pos, _ = p.getBasePositionAndOrientation(
                    self.robot_id, physicsClientId=self._connection
                )
                # Follow robot's actual z so unusual behavior (jumping, falling)
                # stays in frame; clamp low so we never look underground.
                target = [pos[0], pos[1], max(0.35, pos[2] * 0.6)]
            else:
                target = [0.0, 0.0, 0.35]
            view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target,
                distance=2.2,
                yaw=45,
                pitch=-20,
                roll=0,
                upAxisIndex=2,
            )
            proj = p.computeProjectionMatrixFOV(
                fov=60, aspect=640 / 480, nearVal=0.1, farVal=100
            )
            _, _, px, _, _ = p.getCameraImage(
                640,
                480,
                viewMatrix=view,
                projectionMatrix=proj,
                physicsClientId=self._connection,
            )
            img = np.array(px, dtype=np.uint8).reshape(480, 640, 4)
            return img[:, :, :3]
        return None

    def update_live_params(self, params: dict[str, Any]) -> None:
        """Apply dotted-key parameter overrides to live env state.

        Called by CurriculumCallback and LiveTuningCallback via
        ``VecEnv.env_method("update_live_params", params)`` so that changes take
        effect immediately rather than waiting for the next env reset.

        Supported prefixes mirror the env config sections:
        - ``reward.*``       → ``self.reward_fn.<attr>``
        - ``termination.*``  → ``self.termination.<attr>``
        - ``domain_randomization.*`` → config + sensor/latency fields
        - ``sim.gravity`` / ``sim.timestep`` / timing fields → live physics/config

        The underlying ``self.cfg`` dict is also updated so that future resets
        reconstruct objects with the same values.
        """
        _SECTION_OBJ_MAP = {
            "reward": self.reward_fn,
            "termination": self.termination,
        }
        for dotted_key, value in params.items():
            parts = dotted_key.split(".", 1)
            if len(parts) != 2:
                continue
            section, attr = parts
            if section not in {
                "reward",
                "termination",
                "domain_randomization",
                "sim",
            }:
                continue
            live_obj = _SECTION_OBJ_MAP.get(section)
            if live_obj is not None and hasattr(live_obj, attr):
                try:
                    cast_val = type(getattr(live_obj, attr))(value)
                    setattr(live_obj, attr, cast_val)
                    # Mirror into cfg so resets rebuild with the updated
                    # value. setdefault leaves an explicit `section: null`
                    # untouched (it only fills in a genuinely absent key);
                    # get_section replaces it with {} first.
                    section_cfg = get_section(self.cfg, section)
                    section_cfg[attr] = cast_val
                except (TypeError, ValueError):
                    pass
                continue

            section_cfg = get_section(self.cfg, section)
            current = section_cfg.get(attr)
            try:
                cast_val = type(current)(value) if current is not None else value
            except (TypeError, ValueError):
                continue
            section_cfg[attr] = cast_val

            if section == "domain_randomization":
                if attr == "sensor_noise_std":
                    self._sensor_noise_std = float(cast_val)
                elif attr == "action_latency_steps":
                    self._action_latency_steps = int(cast_val)
                    self._action_buffer.clear()
            elif section == "sim":
                if attr == "gravity":
                    p.setGravity(
                        0, 0, float(cast_val), physicsClientId=self._connection
                    )
                elif attr == "timestep":
                    self._sim_timestep = float(cast_val)
                    p.setTimeStep(self._sim_timestep, physicsClientId=self._connection)
                elif attr == "frame_skip":
                    self._frame_skip = int(cast_val)
                elif attr == "settle_steps":
                    self._settle_steps = int(cast_val)

    def close(self) -> None:
        if p.isConnected(self._connection):
            p.disconnect(self._connection)
