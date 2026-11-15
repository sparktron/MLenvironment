from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from rl_framework.physics.locomotion.dynamics import (
    JOINT_INDICES,
    JOINT_SPECS,
    REST_POSE,
    WalkerDynamics,
)
from rl_framework.physics.locomotion.gait import WalkerGaitTracker
from rl_framework.physics.locomotion.rewards import WalkerReward
from rl_framework.physics.locomotion.terminations import WalkerTermination
from rl_framework.core.config.config_merge import get_section, set_nested

LIVE_PARAMS = frozenset(
    {f"reward.{key}" for key in WalkerReward.__dataclass_fields__}
    | {f"termination.{key}" for key in WalkerTermination.__dataclass_fields__}
    | {
        f"sim.{key}"
        for key in ("gravity", "timestep", "frame_skip", "settle_steps", "action_scale")
    }
    | {
        f"domain_randomization.{key}"
        for key in (
            "mass_scale_range",
            "friction_range",
            "sensor_noise_std",
            "action_latency_steps",
        )
    }
    | {
        f"terrain.{key}"
        for key in (
            "preset",
            "height",
            "obstacle_height",
            "mix",
            "push_recovery",
            "push_recovery.interval_steps",
            "push_recovery.force",
            "push_recovery.start_step",
            "push_recovery.duration_s",
            "length",
            "width",
            "randomize",
            "wavelength",
            "obstacle_spacing",
            "obstacle_spacing_jitter",
            "obstacle_height_jitter",
        )
    }
)

# Terrain keys whose live values are numeric. Kept beside LIVE_PARAMS so the
# cast table cannot drift from the accepted key set.
_TERRAIN_FLOAT_FIELDS = frozenset(
    {
        "height",
        "obstacle_height",
        "push_recovery.force",
        "push_recovery.duration_s",
        "length",
        "width",
        "wavelength",
        "obstacle_spacing",
        "obstacle_spacing_jitter",
        "obstacle_height_jitter",
    }
)
_TERRAIN_INT_FIELDS = frozenset(
    {"push_recovery.interval_steps", "push_recovery.start_step"}
)
VALID_CLEARANCE_POINTS = frozenset({"com", "sole"})
DEFAULT_PUSH_DURATION_S = 0.1
DEFAULT_TERRAIN_LENGTH = 25.0
DEFAULT_TERRAIN_WIDTH = 3.0
# Terrain starts here so reset randomisation never begins in contact.
TERRAIN_START_X = 0.6
TERRAIN_DX = 0.05
TERRAIN_DY = 0.25
# Pooled terrain bodies not used this episode wait here, under the plane.
TERRAIN_PARK_POSITION = (0.0, 0.0, -50.0)


def _terrain_rng(seed: int | None) -> np.random.Generator:
    """Terrain layout draws from their own stream.

    Sharing the main env RNG would shift every later draw (mix sampling,
    reset noise, push direction) whenever a layout is generated, so adding
    per-episode layouts would silently change flat and push episodes too.
    """
    return np.random.default_rng(None if seed is None else [int(seed), 7919])


class WalkerBulletEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.render_mode = cfg.get("render_mode")
        self._connection = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
        try:
            self._rng = np.random.default_rng(cfg.get("seed", 0))
            self._layout_rng = _terrain_rng(cfg.get("seed", 0))

            # get_section everywhere: the GUI wizard sometimes writes
            # `key: null` for nested groups, which would otherwise crash
            # .get(...) chains (a present key with value None is returned as
            # None, not the .get(..., {}) default).
            sim_cfg = get_section(cfg, "sim")
            # A flattened `control: null` used to fall back to defaults
            # silently, which looks like a working run and is not one.
            if "control" in sim_cfg and not isinstance(sim_cfg["control"], dict):
                raise ValueError(
                    "environment.sim.control must be a mapping with mode, "
                    f"position_gain and velocity_gain; got {sim_cfg['control']!r}"
                )
            # Thighs and shins collide only when enabled. Off by default so
            # existing checkpoints keep their physics; obstacle results with
            # it off are optimistic, because a shin cannot catch an edge.
            self._limb_collision = bool(sim_cfg.get("limb_collision", False))
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
            # Curriculum-controlled multiplier applied after clipping the
            # policy action. Early balance stages use a smaller range so
            # exploration cannot immediately throw the body to the ground.
            self._action_scale = float(sim_cfg.get("action_scale", 1.0))
            # How many sim steps to run with the PD holding the rest pose
            # immediately after reset, so the robot is at equilibrium before
            # the first agent observation.
            self._settle_steps = int(sim_cfg.get("settle_steps", 30))
            reward_cfg = get_section(cfg, "reward")
            self._reject_unknown_section_keys(reward_cfg, WalkerReward, "reward")
            self.reward_fn = WalkerReward(
                **{
                    k: v
                    for k, v in reward_cfg.items()
                    if k in WalkerReward.__annotations__
                }
            )
            term_cfg = get_section(cfg, "termination")
            self._reject_unknown_section_keys(
                term_cfg, WalkerTermination, "termination"
            )
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
            self._include_previous_action = bool(
                obs_cfg.get("include_previous_action", False)
            )
            if self._observation_version == "v1" and self._coordinate_free:
                raise ValueError(
                    "observation.coordinate_free requires observation.version: v2"
                )
            self._include_gait_phase = bool(obs_cfg.get("include_gait_phase", False))
            if self._observation_version == "v1" and self._include_previous_action:
                raise ValueError(
                    "observation.include_previous_action requires observation.version: v2"
                )
            # v1: pos(3)+quat(4)+lin_vel(3)+ang_vel(3)+joints(20)+DR(2) = 35
            # v2 adds binary right/left foot contacts. Coordinate-free v2
            # drops global x/y position, keeping height as the useful local cue.
            self._obs_size = 35 + (2 if self._observation_version == "v2" else 0)
            if self._coordinate_free:
                self._obs_size -= 2
            if self._include_previous_action:
                self._obs_size += 10
            # The gait clock enters as (sin, cos) so it is continuous across
            # the wrap at phase 1 -> 0.
            if self._include_gait_phase:
                self._obs_size += 2
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
            self._previous_action = np.zeros(self.action_space.shape, dtype=np.float32)

            # Domain randomisation: sensor noise
            rand_cfg = get_section(cfg, "domain_randomization")
            self._sensor_noise_std = float(rand_cfg.get("sensor_noise_std", 0.0))

            # Domain randomisation: action latency
            self._action_latency_steps = int(rand_cfg.get("action_latency_steps", 0))
            self._action_buffer: deque[np.ndarray] = deque()

            terrain_cfg = get_section(cfg, "terrain")
            self._terrain_body_ids: list[int] = []
            # `terrain.mix` samples a terrain per episode so one policy trains
            # on the whole distribution. The 2026-08-06 v11 result showed why
            # staged single-terrain training does not work here: continuing a
            # flat policy on uneven improved uneven (174 -> 410 steps) but
            # transferred nothing to obstacles (116 -> 113) and regressed flat
            # (3% -> 22% falls). Terrain types compete rather than compose.
            self._terrain_mix = self._parse_terrain_mix(terrain_cfg)
            # The spec currently in force. With no mix this is the static
            # `terrain` block, so behaviour is byte-identical to before.
            self._active_terrain: dict[str, Any] = dict(terrain_cfg)
            self._apply_terrain_spec(self._active_terrain)
            gait_cfg = get_section(cfg, "gait")
            # Reference gait cycle. Four screens of per-step *property* terms
            # (clearance, cadence, symmetry, support fractions) were each
            # satisfied by a non-walking gait, because no marginal can express
            # the anti-phase relationship that defines a walk. The clock and
            # the schedule below encode the ordering directly: right leg is
            # expected in stance over phase [0, stance_duty), left leg over the
            # same window offset by half a cycle.
            self._gait_cycle_period_s = float(gait_cfg.get("cycle_period_s", 1.0))
            if self._gait_cycle_period_s <= 0:
                raise ValueError("gait.cycle_period_s must be positive")
            # 0.6 gives 2*0.6-1 = 20% double support and no flight phase.
            self._gait_stance_duty = float(gait_cfg.get("stance_duty", 0.6))
            if not 0.5 < self._gait_stance_duty < 1.0:
                raise ValueError(
                    "gait.stance_duty must be in (0.5, 1.0); below 0.5 leaves a "
                    "flight phase, at/above 1.0 no foot ever swings"
                )
            self._gait_phase = 0.0
            # Which point of the foot swing clearance is measured at. "com"
            # is the historical link centre of mass, which sits 4 cm ahead
            # of the ankle and mid-height in the foot box: a toe-down foot
            # reports 2.5-3.7 cm more clearance than its sole really has.
            # "sole" uses the lowest point of the foot collision box.
            # "com" stays the default so existing reward configs keep their
            # semantics; prefer "sole" for new work.
            self._clearance_point = str(gait_cfg.get("clearance_point", "com"))
            if self._clearance_point not in VALID_CLEARANCE_POINTS:
                raise ValueError(
                    "gait.clearance_point must be one of "
                    f"{sorted(VALID_CLEARANCE_POINTS)}, got {self._clearance_point!r}"
                )
            self._push_ticks_remaining = 0
            self._push_vector = (0.0, 0.0, 0.0)
            self._reset_terrain_pools()
            self._gait_tracker = WalkerGaitTracker(
                touchdown_debounce_steps=int(
                    gait_cfg.get("touchdown_debounce_steps", 3)
                ),
                control_timestep=float(sim_cfg.get("timestep", 1.0 / 240.0))
                * int(sim_cfg.get("frame_skip", 4)),
                min_swing_duration=float(gait_cfg.get("min_swing_duration", 0.0)),
                min_foot_clearance=float(gait_cfg.get("min_foot_clearance", 0.0)),
            )

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
            self._episode_start_x = 0.0
            self._episode_abs_tilt = 0.0
            self._episode_action_l2 = 0.0
            self._episode_reward_components: dict[str, float] = {}
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
    def _reject_unknown_section_keys(
        section: dict[str, Any], cls: type, name: str
    ) -> None:
        # A warning is easy to miss in a 24-worker log, and a typo'd reward
        # weight trains the wrong objective for hours. Fail at construction.
        unknown = sorted(set(section) - set(cls.__annotations__))
        if unknown:
            raise ValueError(
                f"Unknown {name} keys {unknown}; valid keys are "
                f"{sorted(cls.__annotations__)}"
            )

    def _build_world(self) -> None:
        cid = self._connection
        p.resetSimulation(physicsClientId=cid)
        p.setTimeStep(self._sim_timestep, physicsClientId=cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        sim_cfg = get_section(self.cfg, "sim")
        p.setGravity(0, 0, sim_cfg.get("gravity", -9.81), physicsClientId=cid)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=cid)
        self._reset_terrain_pools()
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
        # Foot visuals sit exactly on the collision box, centred on the ankle.
        # They used to be drawn 4 cm forward of it, so videos showed the foot
        # where it does not collide. The inertial frame keeps its 4 cm offset.
        r_thigh_vis = _vis(th, RED, (0, 0, -th[2]))
        r_shin_vis = _vis(sh, BLUE, (0, 0, -sh[2]))
        r_foot_col = _col(fh)
        r_foot_vis = _vis(fh, DGRAY)
        l_thigh_vis = _vis(th, RED, (0, 0, -th[2]))
        l_shin_vis = _vis(sh, BLUE, (0, 0, -sh[2]))
        l_foot_col = _col(fh)
        l_foot_vis = _vis(fh, DGRAY)

        def _limb_col(half):
            if not self._limb_collision:
                return -1
            return p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half,
                collisionFramePosition=[0, 0, -half[2]],
                physicsClientId=cid,
            )

        r_thigh_col, r_shin_col = _limb_col(th), _limb_col(sh)
        l_thigh_col, l_shin_col = _limb_col(th), _limb_col(sh)

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
                r_thigh_col,
                r_shin_col,
                r_foot_col,  # right leg (thigh/shin only with limb_collision)
                l_thigh_col,
                l_shin_col,
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

    @staticmethod
    def _parse_terrain_mix(
        terrain_cfg: dict[str, Any],
    ) -> list[tuple[dict[str, Any], float]] | None:
        """Parse ``terrain.mix`` into (spec, weight) pairs, or None when absent.

        Each entry is a full terrain spec — the same keys the static ``terrain``
        block accepts — plus an optional ``weight`` (default 1.0). Weights are
        relative, not probabilities; they are normalised at sample time.
        """
        raw = terrain_cfg.get("mix")
        if raw is None:
            return None
        if not isinstance(raw, (list, tuple)):
            raise ValueError("terrain.mix must be a list of terrain specs")
        # An explicit-but-empty mix is a config error, not "no mix". Silently
        # falling back to the default terrain is the failure mode that made the
        # terrain curriculum a no-op: the run looks fine and trains on the
        # wrong distribution.
        if len(raw) == 0:
            raise ValueError("terrain.mix must contain at least one entry")
        parsed: list[tuple[dict[str, Any], float]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                raise ValueError(f"terrain.mix entries must be mappings, got {entry!r}")
            spec = {k: v for k, v in entry.items() if k != "weight"}
            weight = float(entry.get("weight", 1.0))
            if weight < 0:
                raise ValueError("terrain.mix weights must be non-negative")
            spec.setdefault("preset", "flat")
            parsed.append((spec, weight))
        if not parsed:
            raise ValueError("terrain.mix must contain at least one entry")
        if sum(w for _, w in parsed) <= 0:
            raise ValueError("terrain.mix weights must not sum to zero")
        return parsed

    def _apply_terrain_spec(self, spec: dict[str, Any]) -> None:
        """Point the env's terrain-derived state at ``spec``.

        Covers the push-recovery parameters as well as the preset, because
        ``push_recovery`` is a terrain whose behaviour lives entirely in those
        fields — sampling the preset without them would silently produce an
        unpushed flat episode.
        """
        self._active_terrain = dict(spec)
        self._terrain_preset = str(spec.get("preset", "flat"))
        push_cfg = get_section(spec, "push_recovery")
        self._push_interval_steps = int(push_cfg.get("interval_steps", 0))
        self._push_force = float(push_cfg.get("force", 0.0))
        self._push_start_step = int(push_cfg.get("start_step", 0))
        # How long each push lasts. PyBullet clears external forces after
        # every physics tick, so a push applied once lasted 1/240 s: 180 N
        # gave 0.75 N*s, about 0.01 m/s on this 65.9 kg body.
        self._push_duration_s = float(
            push_cfg.get("duration_s", DEFAULT_PUSH_DURATION_S)
        )
        if self._push_duration_s <= 0:
            raise ValueError("terrain.push_recovery.duration_s must be positive")

    def _sample_terrain(self) -> None:
        """Draw the next episode's terrain from ``terrain.mix``.

        Uses the env's seeded RNG, so a seeded run replays the same terrain
        sequence.
        """
        if not self._terrain_mix:
            return
        weights = np.array([w for _, w in self._terrain_mix], dtype=np.float64)
        index = int(self._rng.choice(len(self._terrain_mix), p=weights / weights.sum()))
        self._apply_terrain_spec(self._terrain_mix[index][0])

    def _terrain_varies_per_episode(self) -> bool:
        return self._terrain_preset in {"uneven", "obstacles"} and bool(
            self._active_terrain.get("randomize", True)
        )

    def _rebuild_terrain(self) -> None:
        """Swap this episode's terrain in, without rebuilding the world.

        ``_build_world`` runs once per env and later resets only rewind state,
        which saves tens of ms per episode. Terrain bodies are pooled rather
        than recreated: PyBullet cannot free collision or visual shapes, so
        creating them per episode leaks memory for the life of the process.
        Unused pool bodies are parked below the ground plane.
        """
        cid = self._connection
        for body_id in self._terrain_body_ids:
            p.resetBasePositionAndOrientation(
                body_id, TERRAIN_PARK_POSITION, (0, 0, 0, 1), physicsClientId=cid
            )
        self._terrain_body_ids = []
        self._build_terrain()

    def _reset_terrain_pools(self) -> None:
        # resetSimulation destroys every body and shape.
        self._terrain_body_ids = []
        self._heightfields: dict[tuple[int, int, float], tuple[int, int]] = {}
        self._obstacle_shapes: dict[tuple[float, float], int] = {}
        self._obstacle_pools: dict[tuple[float, float], list[int]] = {}

    def _terrain_layout_rng(self) -> np.random.Generator:
        # randomize: false gives one fixed layout per spec, the same in every
        # episode and every env, for controlled comparisons.
        if self._active_terrain.get("randomize", True):
            return self._layout_rng
        return np.random.default_rng(0)

    @staticmethod
    def uneven_profile(
        length: float, height: float, wavelength: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Smooth ground heights in [0, height], one every ``TERRAIN_DX``.

        Control points every half wavelength take uniform heights and are
        joined by cosine interpolation, so the surface undulates without step
        edges. The first and last points are zero, so the terrain ramps up
        from the spawn plane and back down to it, and one interior point is
        exactly ``height``, so every layout spans the full [0, height] range.
        """
        if wavelength <= 0 or length <= 0 or height < 0:
            raise ValueError(
                "terrain.length and terrain.wavelength must be positive and "
                "terrain.height non-negative"
            )
        spacing = wavelength / 2.0
        knots = max(3, int(np.ceil(length / spacing)) + 1)
        values = rng.uniform(0.0, height, size=knots)
        values[0] = values[-1] = 0.0
        values[int(rng.integers(1, knots - 1))] = height
        xs = np.arange(0.0, length + 1e-9, TERRAIN_DX)
        position = xs / spacing
        left = np.clip(np.floor(position).astype(int), 0, knots - 2)
        frac = np.clip(position - left, 0.0, 1.0)
        weight = 0.5 - 0.5 * np.cos(np.pi * frac)
        return values[left] * (1.0 - weight) + values[left + 1] * weight

    def _new_terrain_body(self, collision: int) -> int:
        cid = self._connection
        friction = float(get_section(self.cfg, "sim").get("friction", 0.8))
        # No separate visual shape: PyBullet draws the collision shape.
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            basePosition=TERRAIN_PARK_POSITION,
            physicsClientId=cid,
        )
        p.changeDynamics(body_id, -1, lateralFriction=friction, physicsClientId=cid)
        p.changeVisualShape(
            body_id, -1, rgbaColor=[0.25, 0.45, 0.25, 1.0], physicsClientId=cid
        )
        return body_id

    def _place_terrain_body(
        self, body_id: int, position: tuple[float, float, float]
    ) -> None:
        p.resetBasePositionAndOrientation(
            body_id, position, (0, 0, 0, 1), physicsClientId=self._connection
        )
        self._terrain_body_ids.append(body_id)

    def _build_terrain(self) -> None:
        """Place static terrain along the whole track, starting clear of spawn.

        Every surface sits on or above the ground plane, and the terrain runs
        ``length`` metres (default 25 m), past where an 800-1000 step episode
        at walking speed can reach. Layouts are redrawn every episode from a
        dedicated RNG unless ``randomize: false``.
        """
        if self._terrain_preset == "flat" or self._terrain_preset == "push_recovery":
            return

        cid = self._connection
        terrain_cfg = self._active_terrain
        length = float(terrain_cfg.get("length", DEFAULT_TERRAIN_LENGTH))
        width = float(terrain_cfg.get("width", DEFAULT_TERRAIN_WIDTH))
        if width <= 0:
            raise ValueError("terrain.width must be positive")
        rng = self._terrain_layout_rng()

        if self._terrain_preset == "uneven":
            height = float(terrain_cfg.get("height", 0.025))
            wavelength = float(terrain_cfg.get("wavelength", 1.0))
            profile = self.uneven_profile(length, height, wavelength, rng)
            columns = max(2, int(round(width / TERRAIN_DY)) + 1)
            # Row-major with x varying fastest along each row.
            data = np.tile(profile, columns).tolist()
            key = (len(profile), columns, height)
            existing = self._heightfields.get(key)
            shape_kwargs = dict(
                meshScale=[TERRAIN_DX, TERRAIN_DY, 1.0],
                heightfieldData=data,
                numHeightfieldRows=len(profile),
                numHeightfieldColumns=columns,
                physicsClientId=cid,
            )
            if existing is None:
                shape = p.createCollisionShape(p.GEOM_HEIGHTFIELD, **shape_kwargs)
                body_id = self._new_terrain_body(shape)
                self._heightfields[key] = (shape, body_id)
            else:
                shape, body_id = existing
                p.createCollisionShape(
                    p.GEOM_HEIGHTFIELD, replaceHeightfieldIndex=shape, **shape_kwargs
                )
            # Bullet centres a heightfield on its height range, which is
            # always [0, height] here, so local height h lands at world z = h.
            span = (len(profile) - 1) * TERRAIN_DX
            self._place_terrain_body(
                body_id, (TERRAIN_START_X + 0.5 * span, 0.0, 0.5 * height)
            )
        elif self._terrain_preset == "obstacles":
            obstacle_height = float(terrain_cfg.get("obstacle_height", 0.10))
            spacing = float(terrain_cfg.get("obstacle_spacing", 1.2))
            spacing_jitter = float(terrain_cfg.get("obstacle_spacing_jitter", 0.3))
            height_jitter = float(terrain_cfg.get("obstacle_height_jitter", 0.0))
            if obstacle_height <= 0:
                raise ValueError("terrain.obstacle_height must be positive")
            if spacing_jitter < 0 or spacing <= 2 * spacing_jitter:
                raise ValueError(
                    "terrain.obstacle_spacing must exceed twice a non-negative "
                    "obstacle_spacing_jitter"
                )
            if not 0 <= height_jitter < 1:
                raise ValueError("terrain.obstacle_height_jitter must be in [0, 1)")
            # One shape at the tallest possible height; shorter obstacles are
            # the same box sunk into the ground, so jitter needs no new shapes.
            tallest = obstacle_height * (1.0 + height_jitter)
            obstacle_key = (width, tallest)
            if obstacle_key not in self._obstacle_shapes:
                self._obstacle_shapes[obstacle_key] = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=(0.10, 0.5 * width, 0.5 * tallest),
                    physicsClientId=cid,
                )
                self._obstacle_pools[obstacle_key] = []
            pool = self._obstacle_pools[obstacle_key]
            # The first obstacle keeps its historical position, clear of spawn.
            x, index = 1.2, 0
            while x <= TERRAIN_START_X + length:
                top = obstacle_height * (
                    1.0 + rng.uniform(-height_jitter, height_jitter)
                )
                if index == len(pool):
                    pool.append(self._new_terrain_body(self._obstacle_shapes[obstacle_key]))
                self._place_terrain_body(pool[index], (x, 0.0, top - 0.5 * tallest))
                x += spacing + rng.uniform(-spacing_jitter, spacing_jitter)
                index += 1

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
        values: list[float] = []
        values.extend((pos[2],) if self._coordinate_free else pos)
        values.extend(quat)
        values.extend(lin_vel)
        values.extend(ang_vel)
        values.extend(s[0] for s in joint_states)
        values.extend(s[1] for s in joint_states)
        values.extend((self._mass_scale, self._friction_scale))
        if self._observation_version == "v2":
            if foot_contacts is None:
                foot_contacts = self._foot_contacts()
            values.extend(float(contact) for contact in foot_contacts)
        if self._include_previous_action:
            values.extend(self._previous_action)
        if self._include_gait_phase:
            angle = 2.0 * np.pi * self._gait_phase
            values.extend((float(np.sin(angle)), float(np.cos(angle))))
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

    _FOOT_LINKS = (2, 5)

    def _surface_contacts(self) -> dict[int, list[tuple]]:
        """Robot contacts with supporting surfaces, grouped by robot link.

        One ``getContactPoints`` call per step replaces the five per-link
        calls the step used to make; contact queries were about half of each
        env step against physics alone.
        """
        support_body_ids = {self.plane_id, *self._terrain_body_ids}
        grouped: dict[int, list[tuple]] = {}
        for contact in p.getContactPoints(
            bodyA=self.robot_id, physicsClientId=self._connection
        ):
            if contact[2] in support_body_ids:
                grouped.setdefault(int(contact[3]), []).append(contact)
        return grouped

    def _foot_link_states(self) -> list[tuple]:
        return [
            p.getLinkState(
                self.robot_id,
                link,
                computeLinkVelocity=1,
                physicsClientId=self._connection,
            )
            for link in self._FOOT_LINKS
        ]

    @staticmethod
    def _finite(values: Any) -> np.ndarray:
        return np.nan_to_num(
            np.asarray(values, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6
        )

    def _foot_contacts(
        self, grouped: dict[int, list[tuple]] | None = None
    ) -> tuple[bool, bool]:
        grouped = self._surface_contacts() if grouped is None else grouped
        right, left = (bool(grouped.get(link)) for link in self._FOOT_LINKS)
        return right, left

    def _expected_stance(self, phase: float) -> tuple[bool, bool]:
        """Reference contact schedule at a gait phase.

        Right leg is in stance over ``[0, stance_duty)``; the left leg follows
        the same window half a cycle later. With ``stance_duty`` above 0.5 the
        two windows overlap, producing double support and no flight phase.
        """
        duty = self._gait_stance_duty
        right = (phase % 1.0) < duty
        left = ((phase + 0.5) % 1.0) < duty
        return bool(right), bool(left)

    def _foot_contact_point_slip(
        self,
        grouped: dict[int, list[tuple]] | None = None,
        states: list[tuple] | None = None,
    ) -> tuple[float, float]:
        """True sliding speed of each foot: the velocity of the material point
        at the contact, not of the link origin.

        ``_foot_slip_speeds`` reports the link origin, which moves whenever the
        foot rotates about a stationary contact edge -- ordinary heel-off and
        toe-off. Measured across the reward_v2..v7 checkpoints, rotation
        accounts for 18-55% of what the origin reports, and the artifact grows
        with speed. Slip proper is

            v_contact = v_com + omega x (r_contact - r_com)

        evaluated at the slowest contact point, the one anchoring the foot.
        Feet out of contact report 0.0.
        """
        grouped = self._surface_contacts() if grouped is None else grouped
        states = self._foot_link_states() if states is None else states
        speeds: list[float] = []
        for link, state in zip(self._FOOT_LINKS, states):
            points = grouped.get(link)
            if not points:
                speeds.append(0.0)
                continue
            com = np.asarray(state[0], dtype=np.float64)
            vel = self._finite(state[6])
            omega = self._finite(state[7])
            speeds.append(
                min(
                    float(
                        np.linalg.norm(
                            (
                                vel
                                + np.cross(
                                    omega,
                                    np.asarray(contact[5], dtype=np.float64) - com,
                                )
                            )[:2]
                        )
                    )
                    for contact in points
                )
            )
        return speeds[0], speeds[1]

    def _foot_slip_speeds(self) -> tuple[float, float]:
        velocities = self._foot_tangential_velocities()
        return (
            float(np.linalg.norm(velocities[0])),
            float(np.linalg.norm(velocities[1])),
        )

    def _foot_tangential_velocities(
        self, states: list[tuple] | None = None
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        states = self._foot_link_states() if states is None else states
        velocities = []
        for state in states:
            velocity = self._finite(state[6])
            velocities.append((float(velocity[0]), float(velocity[1])))
        return velocities[0], velocities[1]

    def _foot_normal_forces(
        self, grouped: dict[int, list[tuple]] | None = None
    ) -> tuple[float, float]:
        """Return total supporting-surface normal force for each foot."""
        grouped = self._surface_contacts() if grouped is None else grouped
        forces = []
        for link in self._FOOT_LINKS:
            force = sum(float(contact[9]) for contact in grouped.get(link, ()))
            forces.append(force if np.isfinite(force) else 0.0)
        return forces[0], forces[1]

    def _sole_heights(self, states: list[tuple] | None = None) -> tuple[float, float]:
        """World z of the lowest point of each foot's collision box.

        The box is centred on the link frame (the ankle). Its lowest corner
        sits ``sum_i |R[2, i]| * half_i`` below that centre.
        """
        states = self._foot_link_states() if states is None else states
        heights = []
        half = np.asarray(self._FOOT_H, dtype=np.float64)
        for state in states:
            centre = self._finite(state[4])
            rotation = np.asarray(
                p.getMatrixFromQuaternion(state[5]), dtype=np.float64
            ).reshape(3, 3)
            heights.append(float(centre[2] - np.abs(rotation[2]) @ half))
        return heights[0], heights[1]

    def _foot_positions(
        self, states: list[tuple] | None = None
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return sagittal foot positions used for stride/swing telemetry.

        x is always the link centre of mass. z follows ``gait.clearance_point``.
        """
        states = self._foot_link_states() if states is None else states
        com_positions = [self._finite(state[0]) for state in states]
        if self._clearance_point == "sole":
            heights = self._sole_heights(states)
        else:
            heights = (float(com_positions[0][2]), float(com_positions[1][2]))
        return (
            (float(com_positions[0][0]), heights[0]),
            (float(com_positions[1][0]), heights[1]),
        )

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
            self._layout_rng = _terrain_rng(seed)

        # Draw this episode's terrain before anything builds it, so the first
        # reset and every later one go through the same path.
        terrain_changed = hasattr(self, "_pending_terrain")
        if terrain_changed:
            terrain_cfg = self._pending_terrain
            self._terrain_mix = self._parse_terrain_mix(terrain_cfg)
            self._apply_terrain_spec(terrain_cfg)
            del self._pending_terrain
        self._sample_terrain()

        # Build the world once; subsequent resets just rewind state. Saves
        # ~tens of ms per episode (no resetSimulation/loadURDF/createMultiBody).
        if self.robot_id < 0:
            self._build_world()
        elif self._terrain_mix or terrain_changed or self._terrain_varies_per_episode():
            # World already exists: swap only the terrain bodies, preserving
            # the optimisation above.
            self._rebuild_terrain()
        self._apply_domain_randomization()
        self.step_count = 0
        self._push_ticks_remaining = 0
        # Reset action latency buffer: pre-fill with zero actions so the first
        # _action_latency_steps steps apply no-op while the buffer warms up.
        self._action_buffer.clear()
        action_shape = self.action_space.shape
        if action_shape is None:
            raise RuntimeError("Walker action space must have a fixed shape")
        noop = np.zeros(action_shape, dtype=np.float32)
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

        settled_pos, _ = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self._connection
        )
        self._episode_start_x = float(settled_pos[0])
        self._episode_abs_tilt = 0.0
        self._episode_action_l2 = 0.0
        self._episode_reward_components = {
            name: 0.0
            for name in (
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
        }
        foot_contacts = self._foot_contacts()
        action_size = self.action_space.shape[0] if self.action_space.shape else 10
        self._gait_tracker.reset(
            initial_contacts=foot_contacts,
            action_size=action_size,
        )
        self._previous_action.fill(0.0)
        self._gait_phase = 0.0
        obs = self._get_obs(foot_contacts=foot_contacts)
        return obs, {}

    def _apply_push_tick(self) -> None:
        """Apply one physics tick of the active push at the torso's COM.

        With ``WORLD_FRAME`` the position argument is a world coordinate. It
        used to be the world origin, so the push was a yaw torque whose lever
        arm was the distance walked: 1.2 rad/s at 2 m, 11.4 rad/s at 10 m,
        on a robot with no yaw actuator. Applying it at the current COM makes
        it a pure lateral push wherever the robot is.
        """
        com, _ = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self._connection
        )
        p.applyExternalForce(
            self.robot_id,
            -1,
            list(self._push_vector),
            list(com),
            p.WORLD_FRAME,
            physicsClientId=self._connection,
        )
        self._push_ticks_remaining -= 1

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # Action latency: buffer the current action and apply the delayed one.
        if self._action_latency_steps > 0:
            self._action_buffer.append(action)
            action = self._action_buffer.popleft()
        applied_action = action * self._action_scale
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
            self._push_vector = (0.0, direction * self._push_force, 0.0)
            self._push_ticks_remaining = max(
                1, int(round(self._push_duration_s / self._sim_timestep))
            )
            push_applied = True
        self.dynamics.apply_action(
            self.robot_id, applied_action, physicsClientId=self._connection
        )
        for _ in range(max(1, self._frame_skip)):
            if self._push_ticks_remaining > 0:
                self._apply_push_tick()
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

        surface_contacts = self._surface_contacts()
        foot_states = self._foot_link_states()
        foot_contacts = self._foot_contacts(surface_contacts)
        foot_tangential_velocities = self._foot_tangential_velocities(foot_states)
        foot_contact_slip_speeds = self._foot_contact_point_slip(
            surface_contacts, foot_states
        )
        foot_slip_speeds = (
            float(np.linalg.norm(foot_tangential_velocities[0])),
            float(np.linalg.norm(foot_tangential_velocities[1])),
        )
        foot_normal_forces = self._foot_normal_forces(surface_contacts)
        foot_positions = self._foot_positions(foot_states)
        gait_step = self._gait_tracker.update(
            step=self.step_count,
            contacts=foot_contacts,
            pelvis_x=float(pos[0]),
            foot_slip_speeds=foot_slip_speeds,
            foot_contact_slip_speeds=foot_contact_slip_speeds,
            foot_positions=foot_positions,
            applied_action=applied_action,
        )
        # The next observation receives the command that actually reached the
        # PD controller, including curriculum/action scaling and any latency.
        self._previous_action = applied_action.copy()
        # Score against the schedule the policy actually saw, then advance.
        expected_stance = self._expected_stance(self._gait_phase)
        phase_contact_match = 0.5 * sum(
            float(actual == want)
            for actual, want in zip(foot_contacts, expected_stance)
        )
        phase_expects_double_support = all(expected_stance)
        self._gait_phase = (
            self._gait_phase
            + (self._sim_timestep * max(1, self._frame_skip))
            / self._gait_cycle_period_s
        ) % 1.0
        obs = self._get_obs(
            pos=pos,
            quat=quat,
            lin_vel=lin_vel,
            ang_vel=ang_vel,
            foot_contacts=foot_contacts,
        )
        roll, pitch, _ = p.getEulerFromQuaternion(quat)

        # Torso (base, linkIndexA=-1) touching any supporting world surface =
        # the robot has fallen. Checking only plane_id let policies lie prone
        # on raised uneven/obstacle bodies without terminating.
        # Divergence counts as an immediate fall so a corrupted physics state
        # never lingers until max_steps truncation.
        torso_surface_contacts = surface_contacts.get(-1, [])
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

        termination_reason: str | None = None
        if diverged:
            termination_reason = "physics_divergence"
        elif torso_contact:
            termination_reason = "torso_contact"
        elif pos[2] < self.termination.min_height:
            termination_reason = "min_height"
        elif pos[2] > self.termination.max_height:
            termination_reason = "max_height"
        elif truncated:
            termination_reason = "time_limit"
        right_foot_contact, left_foot_contact = foot_contacts
        info = {
            "x_position": pos[0],
            "z_position": pos[2],
            "lin_vel_x": lin_vel[0],
            "roll": roll,
            "pitch": pitch,
            "abs_tilt": abs_tilt,
            "action_l2": action_l2,
            "action_scale": self._action_scale,
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
            "right_foot_tangential_velocity": foot_tangential_velocities[0],
            "left_foot_tangential_velocity": foot_tangential_velocities[1],
            "right_foot_normal_force": foot_normal_forces[0],
            "left_foot_normal_force": foot_normal_forces[1],
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
            "terrain_preset": self._terrain_preset,
        }
        if terminated or truncated:
            steps = max(self.step_count, 1)
            info["walker_episode"] = {
                "episode_length": float(self.step_count),
                "forward_displacement": float(pos[0] - self._episode_start_x),
                "fall": float(terminated),
                "mean_abs_tilt": self._episode_abs_tilt / steps,
                "mean_action_l2": self._episode_action_l2 / steps,
                "action_scale": self._action_scale,
                **self._gait_tracker.episode_metrics(),
                **{
                    f"reward_{name}_mean": total / steps
                    for name, total in self._episode_reward_components.items()
                },
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
        effect immediately. Terrain changes take effect on the next reset.

        Supported prefixes mirror the env config sections:
        - ``reward.*``       → ``self.reward_fn.<attr>``
        - ``termination.*``  → ``self.termination.<attr>``
        - ``domain_randomization.*`` → config + sensor/latency fields
        - ``sim.gravity`` / ``sim.timestep`` / timing fields → live physics/config

        The underlying ``self.cfg`` dict is also updated so that future resets
        reconstruct objects with the same values.
        """
        unknown = set(params) - LIVE_PARAMS
        if unknown:
            raise ValueError(
                f"Unknown live parameters {sorted(unknown)}; valid keys: {sorted(LIVE_PARAMS)}"
            )
        terrain_params = {
            key: value for key, value in params.items() if key.startswith("terrain.")
        }
        if terrain_params:
            terrain = deepcopy(
                getattr(self, "_pending_terrain", get_section(self.cfg, "terrain"))
            )
            for key, value in terrain_params.items():
                field = key.removeprefix("terrain.")
                if field in _TERRAIN_FLOAT_FIELDS:
                    value = float(value)
                elif field in _TERRAIN_INT_FIELDS:
                    value = int(value)
                elif field == "randomize":
                    value = bool(value)
                elif field == "preset":
                    value = str(value)
                elif field == "push_recovery" and not isinstance(value, dict):
                    raise ValueError("terrain.push_recovery must be a mapping")
                set_nested(terrain, field, deepcopy(value), strict=False)
            self._parse_terrain_mix(terrain)
            self._pending_terrain = terrain
            self.cfg["terrain"] = deepcopy(terrain)
        _SECTION_OBJ_MAP = {
            "reward": self.reward_fn,
            "termination": self.termination,
        }
        for dotted_key, value in params.items():
            if dotted_key.startswith("terrain."):
                continue
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
                cast_val = type(getattr(live_obj, attr))(value)
                setattr(live_obj, attr, cast_val)
                get_section(self.cfg, section)[attr] = cast_val
                continue

            section_cfg = get_section(self.cfg, section)
            if attr in {"mass_scale_range", "friction_range"}:
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError(f"{dotted_key} must be a pair of numbers")
                cast_val = [float(item) for item in value]
            elif attr in {"frame_skip", "settle_steps", "action_latency_steps"}:
                cast_val = int(value)
            else:
                cast_val = float(value)
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
                elif attr == "action_scale":
                    self._action_scale = float(cast_val)
        if {"sim.timestep", "sim.frame_skip"} & params.keys():
            self._gait_tracker.set_control_timestep(
                self._sim_timestep * max(1, self._frame_skip)
            )

    def close(self) -> None:
        if p.isConnected(self._connection):
            p.disconnect(self._connection)
