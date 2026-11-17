"""Typed walker configuration: the one place raw YAML dicts are interpreted.

``WalkerBulletEnv`` used to read its raw config with ~60 ``.get`` calls, and
the set of valid keys was restated in ``validation.py``, the GUI schema and a
hand-written ``LIVE_PARAMS`` set. Parsing once into these dataclasses gives
the env typed attributes, and the live-parameter key set is derived from the
fields that declare themselves live (``live_field``), so it cannot drift.

Parsing is behaviour-preserving: defaults, casts and error messages match the
inline code they replaced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rl_framework.core.config.config_merge import get_section
from rl_framework.core.config.live_params import (
    AT_RESET,
    LiveParam,
    LiveParamSet,
    cast_mapping,
    cast_pair,
    live_field,
    params_from_dataclass,
)
from rl_framework.physics.locomotion.rewards import WalkerReward
from rl_framework.physics.locomotion.terminations import WalkerTermination

VALID_CLEARANCE_POINTS = frozenset({"com", "sole"})
# Accepted by validation; the env only branches on "v2", so this is a check, not a parser.
OBSERVATION_VERSIONS = frozenset({"v1", "v2"})


@dataclass
class ControlConfig:
    mode: str = "pd"
    position_gain: float = 0.1
    velocity_gain: float = 1.0


@dataclass
class SimConfig:
    gravity: float = live_field(-9.81)
    timestep: float = live_field(1.0 / 240.0)
    frame_skip: int = live_field(4)
    settle_steps: int = live_field(30)
    # Curriculum multiplier applied after clipping the policy action.
    action_scale: float = live_field(1.0)
    # Thighs/shins collide only when enabled; off by default so existing
    # checkpoints keep their physics.
    limb_collision: bool = False
    max_torque: float = 35.0
    mass: float = 3.0
    friction: float = 0.8
    control: ControlConfig = field(default_factory=ControlConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SimConfig:
        # A flattened `control: null` used to fall back to defaults silently,
        # which looks like a working run and is not one.
        if "control" in raw and not isinstance(raw["control"], dict):
            raise ValueError(
                "environment.sim.control must be a mapping with mode, "
                f"position_gain and velocity_gain; got {raw['control']!r}"
            )
        ctrl = get_section(raw, "control")
        return cls(
            gravity=raw.get("gravity", -9.81),
            timestep=float(raw.get("timestep", 1.0 / 240.0)),
            frame_skip=int(raw.get("frame_skip", 4)),
            settle_steps=int(raw.get("settle_steps", 30)),
            action_scale=float(raw.get("action_scale", 1.0)),
            limb_collision=bool(raw.get("limb_collision", False)),
            # Accept both max_torque and legacy max_force keys.
            max_torque=raw.get("max_torque", raw.get("max_force", 35.0)),
            mass=raw.get("mass", 3.0),
            friction=raw.get("friction", 0.8),
            control=ControlConfig(
                mode=str(ctrl.get("mode", "pd")),
                position_gain=float(ctrl.get("position_gain", 0.1)),
                velocity_gain=float(ctrl.get("velocity_gain", 1.0)),
            ),
        )

    @property
    def control_timestep(self) -> float:
        return self.timestep * max(1, self.frame_skip)


@dataclass
class ObservationConfig:
    version: str = "v1"
    coordinate_free: bool = False
    include_previous_action: bool = False
    include_gait_phase: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ObservationConfig:
        out = cls(
            version=str(raw.get("version", "v1")),
            coordinate_free=bool(raw.get("coordinate_free", False)),
            include_previous_action=bool(raw.get("include_previous_action", False)),
            include_gait_phase=bool(raw.get("include_gait_phase", False)),
        )
        if out.version == "v1" and out.coordinate_free:
            raise ValueError(
                "observation.coordinate_free requires observation.version: v2"
            )
        if out.version == "v1" and out.include_previous_action:
            raise ValueError(
                "observation.include_previous_action requires observation.version: v2"
            )
        return out


@dataclass
class GaitConfig:
    """Reference gait cycle and foot-measurement options.

    Four screens of per-step *property* terms were each satisfied by a
    non-walking gait; the clock and stance schedule encode the anti-phase
    relationship directly. The right leg is expected in stance over phase
    ``[0, stance_duty)``, the left over the same window offset by half a cycle.
    """

    cycle_period_s: float = 1.0
    # 0.6 gives 20% double support and no flight phase.
    stance_duty: float = 0.6
    # "com" is the historical link centre of mass (reads 2.5-3.7 cm more
    # clearance than the sole really has); "sole" is the lowest foot-box point.
    clearance_point: str = "com"
    touchdown_debounce_steps: int = 3
    min_swing_duration: float = 0.0
    min_foot_clearance: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GaitConfig:
        out = cls(
            cycle_period_s=float(raw.get("cycle_period_s", 1.0)),
            stance_duty=float(raw.get("stance_duty", 0.6)),
            clearance_point=str(raw.get("clearance_point", "com")),
            touchdown_debounce_steps=int(raw.get("touchdown_debounce_steps", 3)),
            min_swing_duration=float(raw.get("min_swing_duration", 0.0)),
            min_foot_clearance=float(raw.get("min_foot_clearance", 0.0)),
        )
        if out.cycle_period_s <= 0:
            raise ValueError("gait.cycle_period_s must be positive")
        if not 0.5 < out.stance_duty < 1.0:
            raise ValueError(
                "gait.stance_duty must be in (0.5, 1.0); below 0.5 leaves a "
                "flight phase, at/above 1.0 no foot ever swings"
            )
        if out.clearance_point not in VALID_CLEARANCE_POINTS:
            raise ValueError(
                "gait.clearance_point must be one of "
                f"{sorted(VALID_CLEARANCE_POINTS)}, got {out.clearance_point!r}"
            )
        return out


@dataclass
class DomainRandomizationConfig:
    mass_scale_range: list[float] = live_field([1.0, 1.0], cast=cast_pair)
    friction_range: list[float] = live_field([1.0, 1.0], cast=cast_pair)
    sensor_noise_std: float = live_field(0.0)
    action_latency_steps: int = live_field(0)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DomainRandomizationConfig:
        return cls(
            mass_scale_range=list(raw.get("mass_scale_range", [1.0, 1.0])),
            friction_range=list(raw.get("friction_range", [1.0, 1.0])),
            sensor_noise_std=float(raw.get("sensor_noise_std", 0.0)),
            action_latency_steps=int(raw.get("action_latency_steps", 0)),
        )


@dataclass
class ResetRandomizationConfig:
    position_xy_noise: float = 0.02
    yaw_noise: float = 0.1

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ResetRandomizationConfig:
        return cls(
            position_xy_noise=raw.get("position_xy_noise", 0.02),
            yaw_noise=raw.get("yaw_noise", 0.1),
        )


def _reject_unknown_keys(section: dict[str, Any], cls: type, name: str) -> None:
    # A warning is easy to miss in a 24-worker log, and a typo'd reward
    # weight trains the wrong objective for hours. Fail at construction.
    valid = set(cls.__annotations__)
    unknown = sorted(set(section) - valid)
    if unknown:
        raise ValueError(
            f"Unknown {name} keys {unknown}; valid keys are {sorted(valid)}"
        )


def build_dataclass_section(cfg: dict[str, Any], name: str, cls: type):
    """Build a strict reward/termination dataclass from ``cfg[name]``."""
    section = get_section(cfg, name)
    _reject_unknown_keys(section, cls, name)
    return cls(**{k: v for k, v in section.items() if k in cls.__annotations__})


# Terrain stays a spec dict (``mix`` entries are themselves full specs), but its
# live keys are declared here, once. Every terrain change applies at the next
# reset because terrain bodies can only be swapped between episodes.
_TERRAIN_FLOATS = (
    "height",
    "obstacle_height",
    "length",
    "width",
    "wavelength",
    "obstacle_spacing",
    "obstacle_spacing_jitter",
    "obstacle_height_jitter",
    "push_recovery.force",
    "push_recovery.duration_s",
)
_TERRAIN_INTS = ("push_recovery.interval_steps", "push_recovery.start_step")

TERRAIN_LIVE_PARAMS: tuple[LiveParam, ...] = (
    *(LiveParam(f"terrain.{k}", float, AT_RESET) for k in _TERRAIN_FLOATS),
    *(LiveParam(f"terrain.{k}", int, AT_RESET) for k in _TERRAIN_INTS),
    LiveParam("terrain.randomize", bool, AT_RESET),
    LiveParam("terrain.preset", str, AT_RESET),
    LiveParam("terrain.push_recovery", cast_mapping, AT_RESET),
    # `mix` is validated as a whole by the terrain parser, not cast here.
    LiveParam("terrain.mix", lambda value: value, AT_RESET),
)

# Keys a terrain spec may carry, read off the declarations above. Every terrain
# key is reset-applied, so the declared set and the accepted spec set are the same
# by construction; `validation.py` uses these instead of restating them.
_TERRAIN_PREFIX = "terrain."
_PUSH_PREFIX = "terrain.push_recovery."
TERRAIN_SPEC_KEYS = frozenset(
    p.key[len(_TERRAIN_PREFIX) :].split(".", 1)[0] for p in TERRAIN_LIVE_PARAMS
)
PUSH_RECOVERY_KEYS = frozenset(
    p.key[len(_PUSH_PREFIX) :]
    for p in TERRAIN_LIVE_PARAMS
    if p.key.startswith(_PUSH_PREFIX)
)


def build_live_params() -> LiveParamSet:
    """Every key ``update_live_params`` accepts, derived from the config types."""
    return LiveParamSet(
        [
            *params_from_dataclass("reward", WalkerReward, all_live=True),
            *params_from_dataclass("termination", WalkerTermination, all_live=True),
            *params_from_dataclass("sim", SimConfig),
            *params_from_dataclass("domain_randomization", DomainRandomizationConfig),
            *TERRAIN_LIVE_PARAMS,
        ]
    )


LIVE_PARAM_SET = build_live_params()
LIVE_PARAMS = LIVE_PARAM_SET.keys
