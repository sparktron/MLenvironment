"""Bipedal walker environment (PyBullet, PD position control).

This module only orchestrates ``reset`` and ``step``. The parts that used to
live in one 1,400-line class are separate modules:

- ``config``         typed config + the derived live-parameter set
- ``terrain``        spec parsing, per-episode mix, pooled bodies, heightfield
- ``robot``          body geometry and construction
- ``sensing``        per-step contact/foot snapshot
- ``perturbations``  pushes, domain randomisation, action latency
- ``observation``    named observation layout and ``info`` schema
- ``dynamics`` / ``rewards`` / ``terminations`` / ``gait``  (unchanged)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from rl_framework.core.config.config_merge import get_section, set_nested
from rl_framework.physics.locomotion.config import (
    LIVE_PARAM_SET,
    LIVE_PARAMS,
    VALID_CLEARANCE_POINTS,
    DomainRandomizationConfig,
    GaitConfig,
    ObservationConfig,
    ResetRandomizationConfig,
    SimConfig,
    build_dataclass_section,
)
from rl_framework.physics.locomotion.dynamics import (
    JOINT_INDICES,
    REST_POSE,
    WalkerDynamics,
)
from rl_framework.physics.locomotion.gait import WalkerGaitTracker
from rl_framework.physics.locomotion.observation import ObservationLayout
from rl_framework.physics.locomotion.perturbations import (
    ActionLatency,
    DomainRandomizer,
    PushController,
)
from rl_framework.physics.locomotion.rewards import WalkerReward
from rl_framework.physics.locomotion.robot import (
    TORSO_STAND_Z,
    build_robot,
)
from rl_framework.physics.locomotion.sensing import FootSensor
from rl_framework.physics.locomotion.terminations import WalkerTermination
from rl_framework.physics.locomotion.terrain import (
    TerrainManager,
    parse_terrain_mix,
    uneven_profile,
)

__all__ = ["LIVE_PARAMS", "VALID_CLEARANCE_POINTS", "WalkerBulletEnv"]

_REWARD_COMPONENT_NAMES = (
    "alive",
    "velocity",
    "orientation",
    "action",
    "fall",
    "gait_step_progress",
    "swing_touchdown_progress",
    "stance_slip",
    "action_rate",
    "swing_clearance",
    "touchdown_clearance",
    "bilateral_clearance",
    "phase_contact",
    "phase_double_support",
    "touchdown_rate",
    "flight",
    "double_support",
    "gait_symmetry",
)


class WalkerBulletEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    TORSO_STAND_Z = TORSO_STAND_Z
    uneven_profile = staticmethod(uneven_profile)

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.render_mode = cfg.get("render_mode")
        self._connection = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
        try:
            self._configure(cfg)
        except Exception:
            p.disconnect(self._connection)
            raise

    def _configure(self, cfg: dict[str, Any]) -> None:
        seed = cfg.get("seed", 0)
        self._rng = np.random.default_rng(seed)

        # get_section everywhere: the GUI wizard sometimes writes
        # `key: null` for nested groups, which would otherwise crash
        # .get(...) chains.
        self.sim = SimConfig.from_dict(get_section(cfg, "sim"))
        self.dynamics = WalkerDynamics(
            max_torque=self.sim.max_torque,
            control_mode=self.sim.control.mode,
            position_gain=self.sim.control.position_gain,
            velocity_gain=self.sim.control.velocity_gain,
        )
        self.reward_fn = build_dataclass_section(cfg, "reward", WalkerReward)
        self.termination = build_dataclass_section(
            cfg, "termination", WalkerTermination
        )
        self.obs_config = ObservationConfig.from_dict(get_section(cfg, "observation"))
        self.layout = ObservationLayout.from_config(self.obs_config)
        self._obs_size = self.layout.size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        self._previous_action = np.zeros(self.action_space.shape, dtype=np.float32)

        self.domain_rand = DomainRandomizationConfig.from_dict(
            get_section(cfg, "domain_randomization")
        )
        self._reset_rand = ResetRandomizationConfig.from_dict(
            cfg.get("reset_randomization") or {}
        )
        self._latency = ActionLatency()
        self._randomizer = DomainRandomizer(self._connection)
        self._pushes = PushController(self._connection)

        self.terrain = TerrainManager(
            self._connection,
            get_section(cfg, "terrain"),
            seed=seed,
            friction=lambda: self.sim.friction,
        )
        # A live terrain update staged until the next reset().
        self._pending_terrain: dict[str, Any] | None = None

        self.gait = GaitConfig.from_dict(get_section(cfg, "gait"))
        self._gait_phase = 0.0
        self._gait_tracker = WalkerGaitTracker(
            touchdown_debounce_steps=self.gait.touchdown_debounce_steps,
            control_timestep=self.sim.timestep * int(self.sim.frame_skip),
            min_swing_duration=self.gait.min_swing_duration,
            min_foot_clearance=self.gait.min_foot_clearance,
        )
        self.sensor = FootSensor(
            self._connection,
            clearance_point=self.gait.clearance_point,
            support_body_ids=lambda: {self.plane_id, *self.terrain.body_ids},
        )

        self.step_count = 0
        self.robot_id = -1
        self.plane_id = -1
        # Nominal masses captured after createMultiBody; domain randomization
        # scales each link from its own baseline.
        self._nominal_masses: dict[int, float] = {}
        # Pre-allocated obs buffer; _get_obs() writes in-place and returns a
        # copy so VecEnv cannot mutate the buffer through the returned array.
        self._obs_buf = np.zeros(self._obs_size, dtype=np.float32)
        self._episode_start_x = 0.0
        self._episode_abs_tilt = 0.0
        self._episode_action_l2 = 0.0
        self._episode_reward_components: dict[str, float] = {}

    # -- read-only views kept for tests/diagnostics ---------------------
    @property
    def _action_scale(self) -> float:
        return self.sim.action_scale

    @property
    def _terrain_preset(self) -> str:
        return self.terrain.preset

    @property
    def _terrain_body_ids(self) -> list[int]:
        return self.terrain.body_ids

    @property
    def _push_force(self) -> float:
        return self.terrain.push.force

    @property
    def _push_interval_steps(self) -> int:
        return self.terrain.push.interval_steps

    @property
    def _push_start_step(self) -> int:
        return self.terrain.push.start_step

    def _foot_link_states(self) -> list[tuple]:
        return self.sensor.link_states()

    def _sole_heights(self, states: list[tuple] | None = None):
        return self.sensor.sole_heights(states)

    def _foot_positions(self, states: list[tuple] | None = None):
        return self.sensor.positions(states)

    def _expected_stance(self, phase: float) -> tuple[bool, bool]:
        """Reference contact schedule at a gait phase.

        Right leg is in stance over ``[0, stance_duty)``; the left leg follows
        the same window half a cycle later. With ``stance_duty`` above 0.5 the
        two windows overlap, producing double support and no flight phase.
        """
        duty = self.gait.stance_duty
        right = (phase % 1.0) < duty
        left = ((phase + 0.5) % 1.0) < duty
        return bool(right), bool(left)

    # -- world ------------------------------------------------------------
    def _build_world(self) -> None:
        cid = self._connection
        p.resetSimulation(physicsClientId=cid)
        p.setTimeStep(self.sim.timestep, physicsClientId=cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.sim.gravity, physicsClientId=cid)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=cid)
        self.terrain.reset_pools()
        self.terrain.build()
        self.robot_id, self._nominal_masses = build_robot(
            cid,
            mass=self.sim.mass,
            friction=self.sim.friction,
            limb_collision=self.sim.limb_collision,
            dynamics=self.dynamics,
        )
        self.sensor.robot_id = self.robot_id

    def _get_obs(
        self,
        pos=None,
        quat=None,
        lin_vel=None,
        ang_vel=None,
        foot_contacts: tuple[bool, bool] | None = None,
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
        cfg = self.obs_config
        # Order here must match ObservationLayout.from_config.
        values: list[float] = []
        values.extend((pos[2],) if cfg.coordinate_free else pos)
        values.extend(quat)
        values.extend(lin_vel)
        values.extend(ang_vel)
        values.extend(s[0] for s in joint_states)
        values.extend(s[1] for s in joint_states)
        values.extend((self._randomizer.mass_scale, self._randomizer.friction_scale))
        if cfg.version == "v2":
            if foot_contacts is None:
                foot_contacts = self.sensor.foot_contacts()
            values.extend(float(contact) for contact in foot_contacts)
        if cfg.include_previous_action:
            values.extend(self._previous_action)
        if cfg.include_gait_phase:
            angle = 2.0 * np.pi * self._gait_phase
            values.extend((float(np.sin(angle)), float(np.cos(angle))))
        self._obs_buf[:] = values
        noise = self.domain_rand.sensor_noise_std
        if noise > 0.0:
            obs = self._obs_buf + self._rng.normal(
                0.0, noise, size=self._obs_size
            ).astype(np.float32)
        else:
            obs = self._obs_buf.copy()
        # Guard against PyBullet solver divergence poisoning VecNormalize stats.
        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

    # -- reset ------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.terrain.reseed(seed)

        # Draw this episode's terrain before anything builds it, so the first
        # reset and every later one go through the same path.
        pending = self._pending_terrain
        terrain_changed = pending is not None
        if pending is not None:
            self.terrain.set_mix_from(pending)
            self._pending_terrain = None
        self.terrain.sample(self._rng)

        # Build the world once; subsequent resets just rewind state. Saves
        # ~tens of ms per episode (no resetSimulation/loadURDF/createMultiBody).
        if self.robot_id < 0:
            self._build_world()
        elif self.terrain.mix or terrain_changed or self.terrain.varies_per_episode():
            # World already exists: swap only the terrain bodies.
            self.terrain.rebuild()
        self._randomizer.apply(
            self.domain_rand,
            self._rng,
            self.robot_id,
            self._nominal_masses,
            self.sim.friction,
        )
        self.step_count = 0
        self._pushes.reset()
        action_shape = self.action_space.shape
        if action_shape is None:
            raise RuntimeError("Walker action space must have a fixed shape")
        self._latency.reset(self.domain_rand.action_latency_steps, action_shape)

        pos_noise = self._reset_rand.position_xy_noise
        yaw_noise = self._reset_rand.yaw_noise
        # +5 mm clearance avoids the knife-edge ground contact at z=0 (foot
        # bottom and plane top would otherwise coincide and produce solver pop).
        start_pos = [
            float(self._rng.uniform(-pos_noise, pos_noise)),
            float(self._rng.uniform(-pos_noise, pos_noise)),
            self.TORSO_STAND_Z + 0.005,
        ]
        yaw = float(self._rng.uniform(-yaw_noise, yaw_noise))
        quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        cid = self._connection
        p.resetBasePositionAndOrientation(
            self.robot_id, start_pos, quat, physicsClientId=cid
        )
        # Zero out residual base velocity from the prior episode.
        p.resetBaseVelocity(
            self.robot_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=cid
        )
        # Start in the slightly-crouched rest pose, not locked-straight pillars.
        for j in JOINT_INDICES:
            p.resetJointState(
                self.robot_id,
                j,
                targetValue=float(REST_POSE[j]),
                targetVelocity=0.0,
                physicsClientId=cid,
            )
        # resetJointState re-engages the implicit motor; silence it again
        # before issuing any PD or torque command.
        self.dynamics.disable_velocity_motors(self.robot_id, physicsClientId=cid)

        # Settle phase: let the robot reach equilibrium under PD before the
        # first observation, so step 0 isn't mid-fall from a transient.
        if self.sim.settle_steps > 0:
            self.dynamics.hold_rest_pose(self.robot_id, physicsClientId=cid)
            for _ in range(self.sim.settle_steps):
                p.stepSimulation(physicsClientId=cid)

        settled_pos, _ = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=cid
        )
        self._episode_start_x = float(settled_pos[0])
        self._episode_abs_tilt = 0.0
        self._episode_action_l2 = 0.0
        self._episode_reward_components = dict.fromkeys(_REWARD_COMPONENT_NAMES, 0.0)
        foot_contacts = self.sensor.foot_contacts()
        action_size = self.action_space.shape[0] if self.action_space.shape else 10
        self._gait_tracker.reset(
            initial_contacts=foot_contacts,
            action_size=action_size,
        )
        self._previous_action.fill(0.0)
        self._gait_phase = 0.0
        return self._get_obs(foot_contacts=foot_contacts), {}

    # -- step -------------------------------------------------------------
    def _read_base_state(self):
        """Base pose/velocity, sanitised. Returns (pos, quat, lin_vel, ang_vel, diverged)."""
        cid = self._connection
        pos, quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=cid)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=cid)
        # Rare PyBullet solver divergence (stiff PD + contact impulse blowup)
        # can return NaN/Inf here. _get_obs() sanitizes its own copy, but pos/
        # lin_vel/quat feed termination and reward directly — without this
        # guard a NaN reward poisons GAE for the whole episode's rollout rows
        # and corrupts the next PPO gradient update (NaN action means).
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
        return pos, quat, lin_vel, ang_vel, diverged

    def step(self, action: np.ndarray):
        sim = self.sim
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # Action latency: buffer the current action and apply the delayed one.
        action = self._latency.delay(action, self.domain_rand.action_latency_steps)
        applied_action = action * sim.action_scale
        push_applied = self._pushes.maybe_start(
            self.terrain.push, self.step_count, self._rng, sim.timestep
        )
        self.dynamics.apply_action(
            self.robot_id, applied_action, physicsClientId=self._connection
        )
        # Frame-skip: hold the same command across `frame_skip` physics ticks
        # so the policy operates at ~60 Hz while physics runs at ~240 Hz.
        for _ in range(max(1, sim.frame_skip)):
            if self._pushes.ticks_remaining > 0:
                self._pushes.apply_tick(self.robot_id)
            p.stepSimulation(physicsClientId=self._connection)
        self.step_count += 1

        pos, quat, lin_vel, ang_vel, diverged = self._read_base_state()
        feet = self.sensor.snapshot()
        gait_step = self._gait_tracker.update(
            step=self.step_count,
            contacts=feet.foot_contacts,
            pelvis_x=float(pos[0]),
            foot_slip_speeds=feet.slip_speeds,
            foot_contact_slip_speeds=feet.contact_slip_speeds,
            foot_positions=feet.positions,
            applied_action=applied_action,
        )
        # The next observation receives the command that actually reached the
        # PD controller, including curriculum/action scaling and any latency.
        self._previous_action = applied_action.copy()
        # Score against the schedule the policy actually saw, then advance.
        expected_stance = self._expected_stance(self._gait_phase)
        phase_contact_match = 0.5 * sum(
            float(actual == want)
            for actual, want in zip(feet.foot_contacts, expected_stance)
        )
        phase_expects_double_support = all(expected_stance)
        self._gait_phase = (
            self._gait_phase + sim.control_timestep / self.gait.cycle_period_s
        ) % 1.0
        obs = self._get_obs(
            pos=pos,
            quat=quat,
            lin_vel=lin_vel,
            ang_vel=ang_vel,
            foot_contacts=feet.foot_contacts,
        )
        roll, pitch, _ = p.getEulerFromQuaternion(quat)

        # Torso (base, linkIndexA=-1) touching any supporting world surface =
        # the robot has fallen. Checking only plane_id let policies lie prone
        # on raised uneven/obstacle bodies without terminating. Divergence
        # counts as an immediate fall so a corrupted physics state never
        # lingers until max_steps truncation.
        torso_surface_contacts = feet.torso_contacts
        torso_contact = diverged or bool(torso_surface_contacts)
        torso_contact_body = (
            int(torso_surface_contacts[0][2]) if torso_surface_contacts else None
        )
        terminated, truncated = self.termination.check(
            pos[2], self.step_count, torso_contact
        )
        reward_inputs = {
            "lin_vel_x": lin_vel[0],
            "pitch_roll_penalty": abs(roll) + abs(pitch),
            "action": applied_action,
            "alive": not terminated,
            "fell": terminated,
            "gait_step_progress": gait_step.alternating_step_progress,
            "sustained_swing_progress": gait_step.sustained_swing_progress,
            "stance_slip_speed": gait_step.stance_slip_speed,
            # GaitStep.action_delta_l2 is the L2 *norm*; the penalty is on the
            # squared rate, matching the usual ||a_t - a_{t-1}||^2 form.
            "action_rate_l2": gait_step.action_delta_l2**2,
            "swing_clearance": gait_step.swing_clearance_now,
            "touchdown_interval": gait_step.touchdown_interval,
            "touchdown_clearance": gait_step.touchdown_swing_clearance,
            "bilateral_clearance": gait_step.bilateral_swing_clearance,
            "phase_contact_match": phase_contact_match,
            "phase_expects_double_support": phase_expects_double_support,
            "in_flight": gait_step.in_flight,
            "in_double_support": gait_step.in_double_support,
            "contact_duty_imbalance": gait_step.contact_duty_imbalance,
        }
        reward_components = self.reward_fn.components(**reward_inputs)
        reward = self.reward_fn.compute(**reward_inputs)
        abs_tilt = float(abs(roll) + abs(pitch))
        action_l2 = float(applied_action @ applied_action)
        self._episode_abs_tilt += abs_tilt
        self._episode_action_l2 += action_l2
        for name, value in reward_components.items():
            self._episode_reward_components[name] += value

        termination_reason = self._termination_reason(
            diverged, torso_contact, pos[2], truncated
        )
        right_foot_contact, left_foot_contact = feet.foot_contacts
        info = {
            "x_position": pos[0],
            "z_position": pos[2],
            "lin_vel_x": lin_vel[0],
            "roll": roll,
            "pitch": pitch,
            "abs_tilt": abs_tilt,
            "action_l2": action_l2,
            "action_scale": sim.action_scale,
            "reward_components": reward_components,
            "torso_contact": torso_contact,
            "torso_contact_body": torso_contact_body,
            "right_foot_contact": right_foot_contact,
            "left_foot_contact": left_foot_contact,
            "right_foot_touchdown": gait_step.right_touchdown,
            "left_foot_touchdown": gait_step.left_touchdown,
            "valid_alternating_touchdown": gait_step.valid_alternating_touchdown,
            "gait_step_progress": gait_step.alternating_step_progress,
            "sustained_swing_progress": gait_step.sustained_swing_progress,
            "stance_slip_speed": gait_step.stance_slip_speed,
            "action_delta_l2": gait_step.action_delta_l2,
            "swing_clearance_now": gait_step.swing_clearance_now,
            "gait_phase": self._gait_phase,
            "phase_contact_match": phase_contact_match,
            "touchdown_interval": gait_step.touchdown_interval,
            "contact_duty_imbalance": gait_step.contact_duty_imbalance,
            "target_velocity": self.reward_fn.target_velocity,
            "velocity_error": self.reward_fn.target_velocity - float(lin_vel[0]),
            "right_foot_tangential_velocity": feet.tangential_velocities[0],
            "left_foot_tangential_velocity": feet.tangential_velocities[1],
            "right_foot_normal_force": feet.normal_forces[0],
            "left_foot_normal_force": feet.normal_forces[1],
            "swing_touchdown_event": (
                {
                    "duration": gait_step.touchdown_swing_duration,
                    "clearance": gait_step.touchdown_swing_clearance,
                    "side": gait_step.touchdown_side,
                    "pelvis_x": float(pos[0]),
                    "foot_lead": gait_step.touchdown_foot_lead,
                    "sustained": gait_step.sustained_swing_touchdown,
                }
                if gait_step.touchdown_swing_duration is not None
                else None
            ),
            "stance_completion_event": (
                {
                    "side": gait_step.stance_completion_side,
                    "pelvis_advance": gait_step.stance_pelvis_advance,
                    "duration": gait_step.stance_duration,
                    "mean_slip_speed": gait_step.stance_mean_slip_speed,
                }
                if gait_step.stance_completion_side is not None
                else None
            ),
            "termination_reason": termination_reason,
            "push_applied": push_applied,
            # Which terrain this episode actually ran on. Constant without a
            # mix; with one it is the only way to attribute an episode, since
            # a mixed run's aggregate hides which terrain produced a fall.
            "terrain_preset": self.terrain.preset,
        }
        if terminated or truncated:
            info["walker_episode"] = self._episode_summary(float(pos[0]), terminated)
        return obs, reward, terminated, truncated, info

    def _termination_reason(
        self, diverged: bool, torso_contact: bool, z: float, truncated: bool
    ) -> str | None:
        if diverged:
            return "physics_divergence"
        if torso_contact:
            return "torso_contact"
        if z < self.termination.min_height:
            return "min_height"
        if z > self.termination.max_height:
            return "max_height"
        if truncated:
            return "time_limit"
        return None

    def _episode_summary(self, x: float, terminated: bool) -> dict[str, float]:
        steps = max(self.step_count, 1)
        return {
            "episode_length": float(self.step_count),
            "forward_displacement": float(x - self._episode_start_x),
            "fall": float(terminated),
            "mean_abs_tilt": self._episode_abs_tilt / steps,
            "mean_action_l2": self._episode_action_l2 / steps,
            "action_scale": self.sim.action_scale,
            **self._gait_tracker.episode_metrics(),
            **{
                f"reward_{name}_mean": total / steps
                for name, total in self._episode_reward_components.items()
            },
        }

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

    # -- live parameters ----------------------------------------------------
    def update_live_params(self, params: dict[str, Any]) -> None:
        """Apply dotted-key parameter overrides to live env state.

        Called by CurriculumCallback and LiveTuningCallback via
        ``VecEnv.env_method("update_live_params", params)``. Which keys exist,
        how each is cast and when it applies (immediately, or at the next
        reset for terrain) is declared on the config types
        (``config.LIVE_PARAM_SET``); this method only dispatches.

        The raw ``self.cfg`` dict is updated too, so future resets and
        anything reading ``env.cfg`` see the same values.
        """
        now, at_reset = LIVE_PARAM_SET.split(params)
        if at_reset:
            self._stage_terrain_update(at_reset)
        targets = {
            "reward": self.reward_fn,
            "termination": self.termination,
            "sim": self.sim,
            "domain_randomization": self.domain_rand,
        }
        for key, value in now.items():
            section, attr = key.split(".", 1)
            target = targets[section]
            cast_val = LIVE_PARAM_SET[key].coerce(value, current=getattr(target, attr))
            setattr(target, attr, cast_val)
            get_section(self.cfg, section)[attr] = cast_val
            self._on_live_change(key, cast_val)
        if {"sim.timestep", "sim.frame_skip"} & params.keys():
            self._gait_tracker.set_control_timestep(self.sim.control_timestep)

    def _on_live_change(self, key: str, value: Any) -> None:
        """Side effects for the few live keys that touch PyBullet or buffers."""
        if key == "sim.gravity":
            p.setGravity(0, 0, float(value), physicsClientId=self._connection)
        elif key == "sim.timestep":
            p.setTimeStep(float(value), physicsClientId=self._connection)
        elif key == "domain_randomization.action_latency_steps":
            self._latency.clear()

    def _stage_terrain_update(self, updates: dict[str, Any]) -> None:
        """Stage terrain changes; they take effect at the next ``reset()``."""
        terrain = deepcopy(
            self._pending_terrain
            if self._pending_terrain is not None
            else get_section(self.cfg, "terrain")
        )
        for key, value in updates.items():
            cast_val = LIVE_PARAM_SET[key].coerce(value)
            set_nested(
                terrain, key.removeprefix("terrain."), deepcopy(cast_val), strict=False
            )
        parse_terrain_mix(terrain)
        self._pending_terrain = terrain
        self.cfg["terrain"] = deepcopy(terrain)

    def close(self) -> None:
        if p.isConnected(self._connection):
            p.disconnect(self._connection)
