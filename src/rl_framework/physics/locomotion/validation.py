"""Task-specific experiment configuration validation.

Key sets come from the config types, not from literals here (ADR-005):
reward/termination fields from their dataclasses, terrain and push-recovery
keys from the terrain live-parameter declarations, presets from ``terrain``,
clearance points and observation versions from ``config``. Only the rules the
types cannot express (ranges, signs, cross-field checks) are written out.
"""

from dataclasses import fields
from typing import Any

from rl_framework.core.config.config import _ensure_int
from rl_framework.physics.locomotion.config import (
    OBSERVATION_VERSIONS,
    PUSH_RECOVERY_KEYS,
    TERRAIN_SPEC_KEYS,
    VALID_CLEARANCE_POINTS,
)
from rl_framework.physics.locomotion.rewards import VELOCITY_MODES, WalkerReward
from rl_framework.physics.locomotion.terminations import WalkerTermination
from rl_framework.physics.locomotion.terrain import TERRAIN_PRESETS

# Reward fields that may legitimately be negative.
_SIGNED_REWARD_KEYS = {"target_velocity"}
# Terrain `mix` entries add a relative sampling weight to a full terrain spec.
_MIX_ENTRY_KEYS = {"weight"}
_POSITIVE_TERRAIN_KEYS = ("length", "width", "wavelength", "obstacle_spacing")
_NON_NEGATIVE_TERRAIN_KEYS = (
    "height",
    "obstacle_height",
    "obstacle_spacing_jitter",
    "obstacle_height_jitter",
)


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _reject_unknown(section: dict[str, Any], valid: set[str], name: str) -> None:
    unknown = sorted(set(section) - valid)
    if unknown:
        raise ValueError(
            f"Unknown {name} keys {unknown}; valid keys are {sorted(valid)}"
        )


def _validate_terrain_spec(spec: dict[str, Any], name: str) -> None:
    _reject_unknown(spec, set(TERRAIN_SPEC_KEYS) | _MIX_ENTRY_KEYS, name)
    preset = spec.get("preset", "flat")
    if preset not in TERRAIN_PRESETS:
        raise ValueError(
            f"{name}.preset must be one of {sorted(TERRAIN_PRESETS)}, got {preset!r}"
        )
    for key in _POSITIVE_TERRAIN_KEYS:
        if key in spec and (not _is_number(spec[key]) or spec[key] <= 0):
            raise ValueError(f"{name}.{key} must be a positive number")
    for key in _NON_NEGATIVE_TERRAIN_KEYS:
        if key in spec and (not _is_number(spec[key]) or spec[key] < 0):
            raise ValueError(f"{name}.{key} must be a non-negative number")
    if "randomize" in spec and not isinstance(spec["randomize"], bool):
        raise TypeError(f"{name}.randomize must be bool")
    push = spec.get("push_recovery") or {}
    if not isinstance(push, dict):
        raise ValueError(f"{name}.push_recovery must be a mapping")
    _reject_unknown(push, set(PUSH_RECOVERY_KEYS), f"{name}.push_recovery")
    for key in ("interval_steps", "start_step"):
        if key in push:
            _ensure_int(push[key], f"{name}.push_recovery.{key}", min_value=0)
    if "force" in push and (not _is_number(push["force"]) or push["force"] < 0):
        raise ValueError(f"{name}.push_recovery.force must be non-negative")
    if "duration_s" in push and (
        not _is_number(push["duration_s"]) or push["duration_s"] <= 0
    ):
        raise ValueError(f"{name}.push_recovery.duration_s must be positive")


def validate_config(cfg: dict[str, Any]) -> None:
    env_cfg = cfg.get("environment", {})
    sim = env_cfg.get("sim") or {}
    action_scale = sim.get("action_scale", 1.0)
    if not _is_number(action_scale) or not 0 < action_scale <= 1:
        raise ValueError(
            "environment.sim.action_scale must be a number in (0, 1], "
            f"got {action_scale!r}"
        )
    # The wizard used to flatten this to `control: null`, which the env then
    # replaced with defaults: a run that looks configured and is not.
    if "control" in sim and not isinstance(sim["control"], dict):
        raise ValueError(
            "environment.sim.control must be a mapping (mode, position_gain, "
            f"velocity_gain), got {sim['control']!r}"
        )
    if "limb_collision" in sim and not isinstance(sim["limb_collision"], bool):
        raise TypeError("environment.sim.limb_collision must be bool")
    observation = env_cfg.get("observation") or {}
    version = observation.get("version", "v1")
    if version not in OBSERVATION_VERSIONS:
        raise ValueError("environment.observation.version must be 'v1' or 'v2'")
    coordinate_free = observation.get("coordinate_free", False)
    if not isinstance(coordinate_free, bool):
        raise TypeError("environment.observation.coordinate_free must be bool")
    if coordinate_free and version != "v2":
        raise ValueError(
            "environment.observation.coordinate_free requires version 'v2'"
        )

    reward = env_cfg.get("reward") or {}
    reward_fields = {field.name: field for field in fields(WalkerReward)}
    _reject_unknown(reward, set(reward_fields), "environment.reward")
    for key, value in reward.items():
        if key == "velocity_mode":
            if value not in VELOCITY_MODES:
                raise ValueError(
                    "environment.reward.velocity_mode must be one of "
                    f"{sorted(VELOCITY_MODES)}, got {value!r}"
                )
            continue
        if not _is_number(value):
            raise ValueError(
                f"environment.reward.{key} must be a number, got {value!r}"
            )
        if key not in _SIGNED_REWARD_KEYS and value < 0:
            raise ValueError(
                f"environment.reward.{key} must be a non-negative number, got {value!r}"
            )
    if "velocity_sigma" in reward and reward["velocity_sigma"] <= 0:
        raise ValueError("environment.reward.velocity_sigma must be positive")

    termination = env_cfg.get("termination") or {}
    _reject_unknown(
        termination,
        {field.name for field in fields(WalkerTermination)},
        "environment.termination",
    )

    gait = env_cfg.get("gait") or {}
    if "touchdown_debounce_steps" in gait:
        _ensure_int(
            gait["touchdown_debounce_steps"],
            "environment.gait.touchdown_debounce_steps",
            min_value=1,
        )
    for key in ("min_swing_duration", "min_foot_clearance"):
        if key in gait:
            value = gait[key]
            if not _is_number(value) or value < 0:
                raise ValueError(
                    f"environment.gait.{key} must be a non-negative number, got {value!r}"
                )
    clearance_point = gait.get("clearance_point", "com")
    if clearance_point not in VALID_CLEARANCE_POINTS:
        raise ValueError(
            "environment.gait.clearance_point must be one of "
            f"{sorted(VALID_CLEARANCE_POINTS)}, got {clearance_point!r}"
        )

    terrain = env_cfg.get("terrain") or {}
    _validate_terrain_spec(terrain, "environment.terrain")
    mix = terrain.get("mix")
    if mix is not None:
        if not isinstance(mix, list) or not mix:
            raise ValueError("environment.terrain.mix must be a non-empty list")
        for index, entry in enumerate(mix):
            if not isinstance(entry, dict):
                raise ValueError(f"environment.terrain.mix[{index}] must be a mapping")
            _validate_terrain_spec(entry, f"environment.terrain.mix[{index}]")
