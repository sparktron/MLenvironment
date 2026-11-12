"""Task-specific experiment configuration validation."""

from typing import Any

from rl_framework.utils.config import _ensure_int, _ensure_positive_number


def validate_config(cfg: dict[str, Any]) -> None:
    env_cfg = cfg.get("environment", {})
    num_envs = int(cfg.get("training", {}).get("num_envs", 1))
    self_play_enabled = bool(cfg.get("self_play", {}).get("enabled", False))
    if num_envs != 1 and not self_play_enabled:
        raise ValueError(
            "Shared-policy organism_arena_parallel training requires "
            f"training.num_envs == 1 (got {num_envs}). Enable self_play "
            "to use the native parallel vector-env path."
        )
    battle = env_cfg.get("battle_rules", {})
    for key in ("damage", "attack_range"):
        if key in battle:
            val = battle[key]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or val <= 0:
                raise ValueError(
                    f"environment.battle_rules.{key} must be a positive number, got {val!r}"
                )
    collision_damage = battle.get("collision_damage", 0.0)
    if (
        isinstance(collision_damage, bool)
        or not isinstance(collision_damage, (int, float))
        or collision_damage < 0
    ):
        raise ValueError(
            "environment.battle_rules.collision_damage must be a "
            f"non-negative number, got {collision_damage!r}"
        )
    morphology = env_cfg.get("morphology", {})
    valid_morph_keys = {"base_size", "episode_growth_scale", "health"}
    unknown = set(morphology) - valid_morph_keys
    if unknown:
        raise ValueError(
            f"Unknown environment.morphology keys: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid_morph_keys)}"
        )
    resources = env_cfg.get("resources", {})
    valid_resource_keys = {
        "initial_energy", "max_energy", "movement_cost", "attack_cost",
        "food_count", "food_energy", "food_radius", "food_respawn_steps",
        "food_placement",
    }
    unknown_resources = set(resources) - valid_resource_keys
    if unknown_resources:
        raise ValueError(
            f"Unknown environment.resources keys: {sorted(unknown_resources)}. "
            f"Valid keys: {sorted(valid_resource_keys)}"
        )
    for key in ("initial_energy", "max_energy", "movement_cost", "attack_cost", "food_energy", "food_radius"):
        if key in resources:
            _ensure_positive_number(resources[key], f"environment.resources.{key}")
    for key in ("food_count", "food_respawn_steps"):
        if key in resources:
            _ensure_int(resources[key], f"environment.resources.{key}", min_value=0)
    if resources.get("initial_energy", 1.0) > resources.get("max_energy", 1.0):
        raise ValueError("environment.resources.initial_energy must be <= max_energy")
    if resources.get("food_placement", "uniform") not in {"uniform", "center"}:
        raise ValueError(
            "environment.resources.food_placement must be 'uniform' or 'center'"
        )
    reward = env_cfg.get("reward") or {}
    approach_weight = reward.get("approach_weight", 0.0)
    if (
        isinstance(approach_weight, bool)
        or not isinstance(approach_weight, (int, float))
        or approach_weight < 0
    ):
        raise ValueError(
            "environment.reward.approach_weight must be a non-negative "
            f"number, got {approach_weight!r}"
        )
