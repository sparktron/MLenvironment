"""Typed arena rule configuration and its live-parameter set.

Lives apart from ``arena_parallel`` (which imports matplotlib via the renderer)
so that ``validation.py`` and tooling can derive the accepted keys and defaults
without pulling in the env. The rule dataclasses are the single source for the
arena's key sets, defaults and live declarations (see ADR-004).
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_framework.core.config.live_params import (
    LiveParam,
    LiveParamSet,
    params_from_dataclass,
)

FOOD_PLACEMENTS = frozenset({"uniform", "center"})


@dataclass
class BattleRules:
    damage: float = 0.05
    collision_damage: float = 0.0
    attack_range: float = 0.2
    cooldown_steps: int = 3
    sensing_radius: float = 2.0
    max_steps: int = 400
    win_health_threshold: float = 0.0
    attack_falloff: str = (
        "linear"  # "linear" (distance-graded) or "binary" (hard cliff)
    )


@dataclass
class ResourceRules:
    initial_energy: float = 1.0
    max_energy: float = 1.0
    movement_cost: float = 0.01
    attack_cost: float = 0.04
    food_count: int = 2
    food_energy: float = 0.35
    food_radius: float = 0.10
    food_respawn_steps: int = 40
    food_placement: str = "uniform"


@dataclass
class RewardRules:
    damage_scale: float = 1.0
    approach_weight: float = 0.0


# Every live key, declared once. Rules/resources fields are all tunable and keep
# their runtime type; the rest are explicit. Morphology lives in a raw dict, so a
# key not yet in the config falls back to float.
LIVE_PARAM_SET = LiveParamSet(
    [
        *params_from_dataclass("battle_rules", BattleRules, all_live=True),
        *params_from_dataclass("resources", ResourceRules, all_live=True),
        LiveParam("reward.damage_scale", float),
        LiveParam("reward.approach_weight", float),
        LiveParam("sim.arena_half_extent", float),
        *(
            LiveParam(f"morphology.{key}", fallback=float)
            for key in ("base_size", "health", "episode_growth_scale")
        ),
    ]
)
LIVE_PARAMS = LIVE_PARAM_SET.keys

MORPHOLOGY_PREFIX = "morphology."
# Morphology is a raw dict in the env; its accepted keys are the ones it is live-tuned by.
MORPHOLOGY_KEYS = frozenset(
    key[len(MORPHOLOGY_PREFIX) :]
    for key in LIVE_PARAMS
    if key.startswith(MORPHOLOGY_PREFIX)
)
