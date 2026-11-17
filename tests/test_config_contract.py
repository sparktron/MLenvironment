"""Characterization of walker/arena config acceptance and rejection.

Each row records the behaviour the validators had before key sets were derived
from the config dataclasses (ADR-004). A row changing means an accepted or
rejected config or an error message changed, which must be listed in the PR.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from rl_framework.games.arena import validation as arena_validation
from rl_framework.physics.locomotion import validation as walker_validation

FIXTURES = Path(__file__).parent / "fixtures"

# (env section dict, expected error type or None for accepted, message or None)
WALKER_CASES: list[tuple[str, dict[str, Any], type | None, str | None]] = [
    ("empty", {}, None, None),
    (
        "action_scale zero",
        {"sim": {"action_scale": 0}},
        ValueError,
        "environment.sim.action_scale must be a number in (0, 1], got 0",
    ),
    (
        "action_scale bool",
        {"sim": {"action_scale": True}},
        ValueError,
        "environment.sim.action_scale must be a number in (0, 1], got True",
    ),
    (
        "flattened control null",
        {"sim": {"control": None}},
        ValueError,
        "environment.sim.control must be a mapping (mode, position_gain, velocity_gain), got None",
    ),
    ("control mapping ok", {"sim": {"control": {"mode": "pd"}}}, None, None),
    (
        "limb_collision not bool",
        {"sim": {"limb_collision": "yes"}},
        TypeError,
        "environment.sim.limb_collision must be bool",
    ),
    ("unknown sim key accepted", {"sim": {"not_a_key": 1}}, None, None),
    ("control mode not checked", {"sim": {"control": {"mode": "bogus"}}}, None, None),
    (
        "observation version",
        {"observation": {"version": "v3"}},
        ValueError,
        "environment.observation.version must be 'v1' or 'v2'",
    ),
    (
        "coordinate_free not bool",
        {"observation": {"version": "v2", "coordinate_free": "x"}},
        TypeError,
        "environment.observation.coordinate_free must be bool",
    ),
    (
        "coordinate_free needs v2",
        {"observation": {"coordinate_free": True}},
        ValueError,
        "environment.observation.coordinate_free requires version 'v2'",
    ),
    (
        "reward unknown key",
        {"reward": {"alive_bonuss": 1.0}},
        ValueError,
        "Unknown environment.reward keys ['alive_bonuss']; valid keys are ",
    ),
    (
        "velocity_mode invalid",
        {"reward": {"velocity_mode": "linear"}},
        ValueError,
        "environment.reward.velocity_mode must be one of ['clipped_linear', 'gaussian'], got 'linear'",
    ),
    ("velocity_mode valid", {"reward": {"velocity_mode": "gaussian"}}, None, None),
    (
        "reward negative",
        {"reward": {"alive_bonus": -1}},
        ValueError,
        "environment.reward.alive_bonus must be a non-negative number, got -1",
    ),
    (
        "reward string",
        {"reward": {"alive_bonus": "5"}},
        ValueError,
        "environment.reward.alive_bonus must be a number, got '5'",
    ),
    (
        "target_velocity may be negative",
        {"reward": {"target_velocity": -1.0}},
        None,
        None,
    ),
    (
        "velocity_sigma zero",
        {"reward": {"velocity_sigma": 0}},
        ValueError,
        "environment.reward.velocity_sigma must be positive",
    ),
    (
        "termination unknown key",
        {"termination": {"max_stepz": 5}},
        ValueError,
        "Unknown environment.termination keys ['max_stepz']; valid keys are ",
    ),
    ("termination valid", {"termination": {"max_steps": 800}}, None, None),
    (
        "touchdown_debounce zero",
        {"gait": {"touchdown_debounce_steps": 0}},
        ValueError,
        "Config key 'environment.gait.touchdown_debounce_steps' must be >= 1, got 0",
    ),
    (
        "min_swing_duration negative",
        {"gait": {"min_swing_duration": -1}},
        ValueError,
        "environment.gait.min_swing_duration must be a non-negative number, got -1",
    ),
    (
        "clearance_point invalid",
        {"gait": {"clearance_point": "ankle"}},
        ValueError,
        "environment.gait.clearance_point must be one of ['com', 'sole'], got 'ankle'",
    ),
    ("clearance_point sole", {"gait": {"clearance_point": "sole"}}, None, None),
    (
        "terrain preset invalid",
        {"terrain": {"preset": "hills"}},
        ValueError,
        "environment.terrain.preset must be one of ['flat', 'obstacles', 'push_recovery', 'uneven'], got 'hills'",
    ),
    (
        "terrain length zero",
        {"terrain": {"length": 0}},
        ValueError,
        "environment.terrain.length must be a positive number",
    ),
    (
        "terrain height negative",
        {"terrain": {"height": -1}},
        ValueError,
        "environment.terrain.height must be a non-negative number",
    ),
    (
        "terrain randomize not bool",
        {"terrain": {"randomize": "yes"}},
        TypeError,
        "environment.terrain.randomize must be bool",
    ),
    (
        "terrain unknown key",
        {"terrain": {"heigth": 0.1}},
        ValueError,
        "Unknown environment.terrain keys ['heigth']; valid keys are ",
    ),
    (
        "push unknown key",
        {"terrain": {"push_recovery": {"bogus": 1}}},
        ValueError,
        "Unknown environment.terrain.push_recovery keys ['bogus']; valid keys are ",
    ),
    (
        "push interval negative",
        {"terrain": {"push_recovery": {"interval_steps": -1}}},
        ValueError,
        "Config key 'environment.terrain.push_recovery.interval_steps' must be >= 0, got -1",
    ),
    (
        "push force negative",
        {"terrain": {"push_recovery": {"force": -1}}},
        ValueError,
        "environment.terrain.push_recovery.force must be non-negative",
    ),
    (
        "push duration zero",
        {"terrain": {"push_recovery": {"duration_s": 0}}},
        ValueError,
        "environment.terrain.push_recovery.duration_s must be positive",
    ),
    (
        "push_recovery not mapping",
        {"terrain": {"push_recovery": "x"}},
        ValueError,
        "environment.terrain.push_recovery must be a mapping",
    ),
    (
        "mix empty",
        {"terrain": {"mix": []}},
        ValueError,
        "environment.terrain.mix must be a non-empty list",
    ),
    (
        "mix not list",
        {"terrain": {"mix": "flat"}},
        ValueError,
        "environment.terrain.mix must be a non-empty list",
    ),
    (
        "mix entry not mapping",
        {"terrain": {"mix": [1]}},
        ValueError,
        "environment.terrain.mix[0] must be a mapping",
    ),
    (
        "mix entry weight accepted",
        {"terrain": {"mix": [{"preset": "flat", "weight": 2}]}},
        None,
        None,
    ),
    (
        "mix entry unknown key",
        {"terrain": {"mix": [{"preset": "flat", "wieght": 2}]}},
        ValueError,
        "Unknown environment.terrain.mix[0] keys ['wieght']; valid keys are ",
    ),
    (
        "mix entry bad preset",
        {"terrain": {"mix": [{"preset": "hills"}]}},
        ValueError,
        "environment.terrain.mix[0].preset must be one of ['flat', 'obstacles', 'push_recovery', 'uneven'], got 'hills'",
    ),
]

ARENA_CASES: list[tuple[str, dict[str, Any], type | None, str | None]] = [
    ("empty", {}, None, None),
    (
        "damage zero",
        {"battle_rules": {"damage": 0}},
        ValueError,
        "environment.battle_rules.damage must be a positive number, got 0",
    ),
    (
        "attack_range bool",
        {"battle_rules": {"attack_range": True}},
        ValueError,
        "environment.battle_rules.attack_range must be a positive number, got True",
    ),
    (
        "collision_damage negative",
        {"battle_rules": {"collision_damage": -1}},
        ValueError,
        "environment.battle_rules.collision_damage must be a non-negative number, got -1",
    ),
    ("battle_rules unknown key accepted", {"battle_rules": {"bogus": 1}}, None, None),
    (
        "morphology unknown key",
        {"morphology": {"speed": 1}},
        ValueError,
        "Unknown environment.morphology keys: ['speed']. Valid keys: ['base_size', 'episode_growth_scale', 'health']",
    ),
    (
        "resources unknown key",
        {"resources": {"bogus": 1}},
        ValueError,
        "Unknown environment.resources keys: ['bogus']. Valid keys: ['attack_cost', 'food_count', "
        "'food_energy', 'food_placement', 'food_radius', 'food_respawn_steps', 'initial_energy', "
        "'max_energy', 'movement_cost']",
    ),
    (
        "resources initial_energy zero",
        {"resources": {"initial_energy": 0}},
        ValueError,
        "Config key 'environment.resources.initial_energy' must be > 0, got 0",
    ),
    (
        "resources food_count negative",
        {"resources": {"food_count": -1}},
        ValueError,
        "Config key 'environment.resources.food_count' must be >= 0, got -1",
    ),
    (
        "initial above max",
        {"resources": {"initial_energy": 2.0}},
        ValueError,
        "environment.resources.initial_energy must be <= max_energy",
    ),
    (
        "food_placement invalid",
        {"resources": {"food_placement": "edge"}},
        ValueError,
        "environment.resources.food_placement must be 'uniform' or 'center'",
    ),
    ("food_placement center", {"resources": {"food_placement": "center"}}, None, None),
    (
        "approach_weight negative",
        {"reward": {"approach_weight": -1}},
        ValueError,
        "environment.reward.approach_weight must be a non-negative number, got -1",
    ),
]


def _walker_env(section: dict[str, Any]) -> dict[str, Any]:
    return {"environment": {"type": "walker_bullet", **section}}


def _arena_env(section: dict[str, Any]) -> dict[str, Any]:
    return {"environment": {"type": "organism_arena_parallel", **section}}


def _check(validate, cfg, error, message) -> None:
    if error is None:
        validate(cfg)
        return
    with pytest.raises(error) as info:
        validate(cfg)
    if message.endswith(" "):
        assert str(info.value).startswith(message)
    else:
        assert str(info.value) == message


@pytest.mark.parametrize(
    ("label", "section", "error", "message"),
    WALKER_CASES,
    ids=[c[0] for c in WALKER_CASES],
)
def test_walker_validation_contract(label, section, error, message) -> None:
    _check(walker_validation.validate_config, _walker_env(section), error, message)


@pytest.mark.parametrize(
    ("label", "section", "error", "message"),
    ARENA_CASES,
    ids=[c[0] for c in ARENA_CASES],
)
def test_arena_validation_contract(label, section, error, message) -> None:
    _check(arena_validation.validate_config, _arena_env(section), error, message)


def test_live_param_keys_match_snapshot() -> None:
    from rl_framework.games.arena.arena_parallel import LIVE_PARAMS as arena_live
    from rl_framework.physics.locomotion.config import LIVE_PARAMS as walker_live

    expected = json.loads((FIXTURES / "live_params.json").read_text())
    assert sorted(walker_live) == expected["walker"]
    assert sorted(arena_live) == expected["arena"]


def test_unknown_live_param_raises() -> None:
    from rl_framework.physics.locomotion.config import LIVE_PARAM_SET

    with pytest.raises(
        ValueError, match=re.escape("Unknown live parameters ['sim.nope']")
    ):
        LIVE_PARAM_SET.check(["sim.nope"])


def test_gui_schema_matches_snapshot() -> None:
    from rl_framework.core.registry import get_task, registered_tasks
    import rl_framework.games.arena  # noqa: F401  (registers arena task)
    import rl_framework.physics.locomotion  # noqa: F401  (registers walker task)
    from rl_framework.gui import app as gui_app

    current = {}
    for name in sorted(registered_tasks()):
        task = get_task(name)
        if task.gui_schema is not None:
            current[name] = task.gui_schema(
                gui_app._training_schema, gui_app._eval_schema
            )
    snapshot = json.loads((FIXTURES / "gui_schema.json").read_text())
    current_json = json.loads(json.dumps(current, sort_keys=True, default=str))
    assert current_json == snapshot
