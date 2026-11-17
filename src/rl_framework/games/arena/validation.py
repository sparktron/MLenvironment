"""Task-specific experiment configuration validation.

Resource and morphology key sets, and the defaults used for cross-field checks,
come from the arena rule types (``games/arena/config.py``, ADR-004). The rules
that name which keys must be positive or non-negative stay written out here,
because the types do not carry that information.
"""

from dataclasses import fields
from typing import Any

from rl_framework.core.config.config import _ensure_int, _ensure_positive_number
from rl_framework.games.arena.config import (
    FOOD_PLACEMENTS,
    MORPHOLOGY_KEYS,
    BattleRules,
    ResourceRules,
    RewardRules,
)

_RESOURCE_KEYS = frozenset(f.name for f in fields(ResourceRules))
_RESOURCE_DEFAULTS = ResourceRules()
_BATTLE_DEFAULTS = BattleRules()
_REWARD_DEFAULTS = RewardRules()


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
    collision_damage = battle.get("collision_damage", _BATTLE_DEFAULTS.collision_damage)
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
    unknown = set(morphology) - MORPHOLOGY_KEYS
    if unknown:
        raise ValueError(
            f"Unknown environment.morphology keys: {sorted(unknown)}. "
            f"Valid keys: {sorted(MORPHOLOGY_KEYS)}"
        )
    resources = env_cfg.get("resources", {})
    unknown_resources = set(resources) - _RESOURCE_KEYS
    if unknown_resources:
        raise ValueError(
            f"Unknown environment.resources keys: {sorted(unknown_resources)}. "
            f"Valid keys: {sorted(_RESOURCE_KEYS)}"
        )
    for key in (
        "initial_energy",
        "max_energy",
        "movement_cost",
        "attack_cost",
        "food_energy",
        "food_radius",
    ):
        if key in resources:
            _ensure_positive_number(resources[key], f"environment.resources.{key}")
    for key in ("food_count", "food_respawn_steps"):
        if key in resources:
            _ensure_int(resources[key], f"environment.resources.{key}", min_value=0)
    if resources.get(
        "initial_energy", _RESOURCE_DEFAULTS.initial_energy
    ) > resources.get("max_energy", _RESOURCE_DEFAULTS.max_energy):
        raise ValueError("environment.resources.initial_energy must be <= max_energy")
    if (
        resources.get("food_placement", _RESOURCE_DEFAULTS.food_placement)
        not in FOOD_PLACEMENTS
    ):
        raise ValueError(
            "environment.resources.food_placement must be 'uniform' or 'center'"
        )
    reward = env_cfg.get("reward") or {}
    approach_weight = reward.get("approach_weight", _REWARD_DEFAULTS.approach_weight)
    if (
        isinstance(approach_weight, bool)
        or not isinstance(approach_weight, (int, float))
        or approach_weight < 0
    ):
        raise ValueError(
            "environment.reward.approach_weight must be a non-negative "
            f"number, got {approach_weight!r}"
        )
