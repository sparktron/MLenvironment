"""Tests for the walker's structural modules: typed config, live-parameter
declarations, observation layout, terrain and perturbations."""

from __future__ import annotations

import numpy as np
import pytest

from rl_framework.core.config.live_params import (
    AT_RESET,
    LiveParam,
    LiveParamSet,
    cast_pair,
    live_field,
    params_from_dataclass,
)
from rl_framework.physics.locomotion.config import (
    LIVE_PARAM_SET,
    DomainRandomizationConfig,
    GaitConfig,
    ObservationConfig,
    SimConfig,
)
from rl_framework.physics.locomotion.observation import (
    STEP_INFO_KEYS,
    ObservationLayout,
)
from rl_framework.physics.locomotion.perturbations import ActionLatency
from rl_framework.physics.locomotion.terrain import (
    PushSpec,
    parse_terrain_mix,
    uneven_profile,
)
from rl_framework.physics.locomotion.walker_bullet import WalkerBulletEnv


def _layout(**kwargs) -> ObservationLayout:
    return ObservationLayout.from_config(ObservationConfig(**kwargs))


# -- live parameters -------------------------------------------------------
def test_live_keys_are_derived_from_config_types():
    keys = LIVE_PARAM_SET.keys
    assert "sim.action_scale" in keys and "sim.limb_collision" not in keys
    assert "domain_randomization.mass_scale_range" in keys
    assert "reward.alive_bonus" in keys and "termination.max_height" in keys
    # Every terrain key applies at the next reset, none immediately.
    terrain = [k for k in keys if k.startswith("terrain.")]
    assert terrain and all(LIVE_PARAM_SET[k].when == AT_RESET for k in terrain)


def test_live_field_mutable_defaults_are_not_shared():
    a, b = DomainRandomizationConfig(), DomainRandomizationConfig()
    a.mass_scale_range.append(9.0)
    assert b.mass_scale_range == [1.0, 1.0]


def test_live_param_set_rejects_unknown_and_duplicates():
    with pytest.raises(ValueError, match="Unknown live parameters"):
        LIVE_PARAM_SET.check(["sim.nope"])
    with pytest.raises(ValueError, match="Duplicate"):
        LiveParamSet([LiveParam("a.b"), LiveParam("a.b")])


def test_live_param_coerce_keeps_current_type_and_names_key_on_error():
    assert LiveParam("reward.x").coerce("2", current=1.5) == 2.0
    with pytest.raises(ValueError, match="domain_randomization.mass_scale_range"):
        LIVE_PARAM_SET["domain_randomization.mass_scale_range"].coerce([1.0])


def test_params_from_dataclass_respects_live_marker():
    from dataclasses import dataclass

    @dataclass
    class C:
        a: float = live_field(1.0)
        b: float = 2.0
        pair: list = live_field([0.0, 1.0], cast=cast_pair)

    keys = {p.key for p in params_from_dataclass("s", C)}
    assert keys == {"s.a", "s.pair"}
    assert {p.key for p in params_from_dataclass("s", C, all_live=True)} == {
        "s.a",
        "s.b",
        "s.pair",
    }


def test_split_separates_immediate_from_reset():
    now, later = LIVE_PARAM_SET.split(
        {"sim.action_scale": 0.5, "terrain.height": 0.03}
    )
    assert set(now) == {"sim.action_scale"} and set(later) == {"terrain.height"}


# -- typed config ------------------------------------------------------------
def test_sim_config_defaults_and_legacy_max_force():
    sim = SimConfig.from_dict({"max_force": 50.0})
    assert sim.max_torque == 50.0
    assert sim.timestep == pytest.approx(1 / 240) and sim.frame_skip == 4
    assert sim.control_timestep == pytest.approx(4 / 240)


def test_sim_control_must_be_mapping():
    with pytest.raises(ValueError, match="sim.control must be a mapping"):
        SimConfig.from_dict({"control": None})


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"cycle_period_s": 0}, "cycle_period_s"),
        ({"stance_duty": 0.5}, "stance_duty"),
        ({"clearance_point": "heel"}, "clearance_point"),
    ],
)
def test_gait_config_validates(raw, match):
    with pytest.raises(ValueError, match=match):
        GaitConfig.from_dict(raw)


def test_observation_config_v1_rejects_v2_only_options():
    with pytest.raises(ValueError, match="coordinate_free"):
        ObservationConfig.from_dict({"coordinate_free": True})
    with pytest.raises(ValueError, match="include_previous_action"):
        ObservationConfig.from_dict({"include_previous_action": True})


# -- observation layout --------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, size",
    [
        ({}, 35),
        ({"version": "v2"}, 37),
        ({"version": "v2", "coordinate_free": True}, 35),
        ({"version": "v2", "coordinate_free": True, "include_previous_action": True}, 45),
        ({"version": "v2", "include_gait_phase": True}, 39),
    ],
)
def test_layout_sizes_match_documented_shapes(kwargs, size):
    assert _layout(**kwargs).size == size


def test_layout_slices_are_contiguous_and_named():
    layout = _layout(version="v2", coordinate_free=True, include_previous_action=True)
    cursor = 0
    for _, sl in layout.slices:
        assert sl.start == cursor
        cursor = sl.stop
    assert cursor == layout.size
    assert layout["foot_contacts"].stop - layout["foot_contacts"].start == 2
    assert "z" in layout and "position" not in layout
    with pytest.raises(KeyError):
        layout["nope"]


def test_layout_mismatch_reports_difference():
    base = _layout(version="v2", coordinate_free=True)
    assert base.describe_mismatch(_layout(version="v2", coordinate_free=True)) is None
    other = _layout(version="v2", coordinate_free=True, include_previous_action=True)
    msg = base.describe_mismatch(other)
    assert msg and "previous_action" in msg and "35" in msg and "45" in msg


def test_env_observation_matches_layout_and_info_matches_schema():
    env = WalkerBulletEnv(
        {"seed": 0, "observation": {"version": "v2", "include_previous_action": True}}
    )
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (env.layout.size,) == env.observation_space.shape
        # foot-contact flags sit where the layout says they do
        assert set(obs[env.layout["foot_contacts"]]) <= {0.0, 1.0}
        _, _, _, _, info = env.step(np.zeros(10, dtype=np.float32))
        assert set(info) - {"walker_episode"} == STEP_INFO_KEYS
    finally:
        env.close()


# -- terrain / perturbations -------------------------------------------------------
def test_parse_terrain_mix_rules():
    assert parse_terrain_mix({}) is None
    with pytest.raises(ValueError, match="at least one"):
        parse_terrain_mix({"mix": []})
    with pytest.raises(ValueError, match="sum to zero"):
        parse_terrain_mix({"mix": [{"weight": 0}]})
    (spec, weight), = parse_terrain_mix({"mix": [{"height": 0.1, "weight": 3}]})
    assert spec == {"height": 0.1, "preset": "flat"} and weight == 3.0


def test_push_spec_rejects_non_positive_duration():
    with pytest.raises(ValueError, match="duration_s"):
        PushSpec.from_terrain({"push_recovery": {"duration_s": 0}})


def test_uneven_profile_spans_full_height_and_is_deterministic():
    a = uneven_profile(5.0, 0.04, 1.0, np.random.default_rng(1))
    b = uneven_profile(5.0, 0.04, 1.0, np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert a[0] == a[-1] == 0.0 and a.max() == pytest.approx(0.04)


def test_action_latency_delays_by_n_steps():
    latency = ActionLatency()
    latency.reset(2, (3,))
    outs = [latency.delay(np.full(3, float(i + 1)), 2) for i in range(4)]
    assert [float(o[0]) for o in outs] == [0.0, 0.0, 1.0, 2.0]
    assert latency.delay(np.ones(3), 0).tolist() == [1.0, 1.0, 1.0]


def test_live_update_stages_terrain_until_reset():
    env = WalkerBulletEnv({"seed": 0, "terrain": {"preset": "flat"}})
    try:
        env.reset(seed=0)
        env.update_live_params({"terrain.preset": "uneven", "terrain.height": 0.03})
        assert env._terrain_preset == "flat"  # not yet
        env.reset()
        assert env._terrain_preset == "uneven" and env._terrain_body_ids
        with pytest.raises(ValueError, match="Unknown live parameters"):
            env.update_live_params({"terrain.bogus": 1})
    finally:
        env.close()
