from __future__ import annotations

import numpy as np
import pytest

from rl_framework.envs.locomotion.rewards import WalkerReward


def test_default_walker_reward_prioritizes_target_velocity_over_survival() -> None:
    reward = WalkerReward()
    action = np.zeros(10, dtype=np.float32)

    standing = reward.compute(0.0, 0.0, action, alive=True)
    walking = reward.compute(reward.target_velocity, 0.0, action, alive=True)

    assert reward.forward_velocity_weight > reward.alive_bonus
    assert walking - standing > reward.alive_bonus


def test_reward_components_sum_to_computed_reward() -> None:
    reward = WalkerReward()
    action = np.linspace(-0.5, 0.5, 10, dtype=np.float32)
    kwargs = {
        "lin_vel_x": 0.4,
        "pitch_roll_penalty": 0.3,
        "action": action,
        "alive": False,
        "fell": True,
    }

    components = reward.components(**kwargs)

    assert set(components) == {
        "alive",
        "velocity",
        "orientation",
        "action",
        "fall",
        "gait_step_progress",
        "swing_touchdown_progress",
        "stance_slip",
    }
    assert sum(components.values()) == reward.compute(**kwargs)
    assert components["fall"] == -reward.fall_penalty


def test_zero_weight_gait_terms_preserve_existing_reward() -> None:
    reward = WalkerReward()
    action = np.zeros(10, dtype=np.float32)
    base = reward.compute(0.2, 0.1, action, alive=True)

    structured = reward.compute(
        0.2,
        0.1,
        action,
        alive=True,
        gait_step_progress=0.2,
        stance_slip_speed=3.0,
    )

    assert structured == base


def test_gait_progress_is_positive_bounded_and_slip_is_penalized() -> None:
    reward = WalkerReward(
        gait_step_progress_weight=2.0,
        gait_step_progress_clip=0.1,
        stance_slip_penalty_weight=0.5,
    )
    action = np.zeros(10, dtype=np.float32)

    components = reward.components(
        0.0,
        0.0,
        action,
        alive=True,
        gait_step_progress=0.4,
        stance_slip_speed=0.6,
    )
    backwards = reward.components(
        0.0,
        0.0,
        action,
        alive=True,
        gait_step_progress=-0.4,
        stance_slip_speed=0.0,
    )

    assert components["gait_step_progress"] == pytest.approx(0.2)
    assert components["stance_slip"] == pytest.approx(-0.3)
    assert backwards["gait_step_progress"] == 0.0


def test_stance_slip_penalty_can_clip_contact_impulse_outliers() -> None:
    reward = WalkerReward(
        stance_slip_penalty_weight=0.5,
        stance_slip_penalty_clip=0.8,
    )

    components = reward.components(
        lin_vel_x=0.0,
        pitch_roll_penalty=0.0,
        action=np.zeros(10, dtype=np.float32),
        alive=True,
        stance_slip_speed=3.0,
    )

    assert components["stance_slip"] == pytest.approx(-0.4)


def test_swing_touchdown_progress_reward_is_opt_in_and_bounded() -> None:
    reward = WalkerReward(
        swing_touchdown_progress_weight=40.0,
        swing_touchdown_progress_clip=0.03,
    )

    components = reward.components(
        lin_vel_x=0.0,
        pitch_roll_penalty=0.0,
        action=np.zeros(10, dtype=np.float32),
        alive=True,
        sustained_swing_progress=0.10,
    )

    assert components["swing_touchdown_progress"] == pytest.approx(1.2)
