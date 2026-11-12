"""Task-specific experiment configuration validation."""

from typing import Any

from rl_framework.utils.config import _ensure_int


def validate_config(cfg: dict[str, Any]) -> None:
    env_cfg = cfg.get("environment", {})
    sim = env_cfg.get("sim", {})
    action_scale = sim.get("action_scale", 1.0)
    if (
        isinstance(action_scale, bool)
        or not isinstance(action_scale, (int, float))
        or not 0 < action_scale <= 1
    ):
        raise ValueError(
            "environment.sim.action_scale must be a number in (0, 1], "
            f"got {action_scale!r}"
        )
    observation = env_cfg.get("observation", {})
    version = observation.get("version", "v1")
    if version not in {"v1", "v2"}:
        raise ValueError("environment.observation.version must be 'v1' or 'v2'")
    coordinate_free = observation.get("coordinate_free", False)
    if not isinstance(coordinate_free, bool):
        raise TypeError("environment.observation.coordinate_free must be bool")
    if coordinate_free and version != "v2":
        raise ValueError("environment.observation.coordinate_free requires version 'v2'")
    reward = env_cfg.get("reward", {})
    for key in (
        "alive_bonus",
        "forward_velocity_weight",
        "orientation_penalty_weight",
        "torque_penalty_weight",
        "gait_step_progress_weight",
        "gait_step_progress_clip",
        "swing_touchdown_progress_weight",
        "swing_touchdown_progress_clip",
        "stance_slip_penalty_weight",
        "stance_slip_penalty_clip",
    ):
        if key in reward:
            val = reward[key]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or val < 0:
                raise ValueError(
                    f"environment.reward.{key} must be a non-negative number, got {val!r}"
                )
    gait = env_cfg.get("gait", {})
    if "touchdown_debounce_steps" in gait:
        _ensure_int(
            gait["touchdown_debounce_steps"],
            "environment.gait.touchdown_debounce_steps",
            min_value=1,
        )
    for key in ("min_swing_duration", "min_foot_clearance"):
        if key in gait:
            value = gait[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(
                    f"environment.gait.{key} must be a non-negative number, got {value!r}"
                )
    terrain = env_cfg.get("terrain", {})
    preset = terrain.get("preset", "flat")
    valid_presets = {"flat", "uneven", "obstacles", "push_recovery"}
    if preset not in valid_presets:
        raise ValueError(
            "environment.terrain.preset must be one of "
            f"{sorted(valid_presets)}, got {preset!r}"
        )
    push = terrain.get("push_recovery", {})
    for key in ("interval_steps", "start_step"):
        if key in push:
            _ensure_int(push[key], f"environment.terrain.push_recovery.{key}", min_value=0)
    if "force" in push:
        force = push["force"]
        if isinstance(force, bool) or not isinstance(force, (int, float)) or force < 0:
            raise ValueError("environment.terrain.push_recovery.force must be non-negative")

