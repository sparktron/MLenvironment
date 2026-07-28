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
        "action_rate",
        "swing_clearance",
        "touchdown_rate",
        "flight",
        "gait_symmetry",
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


def test_clipped_linear_velocity_keeps_gradient_where_gaussian_vanishes() -> None:
    """The shipped Gaussian at sigma 0.10 / target 1.0 is a needle.

    Below ~0.7 m/s its reward and gradient underflow to numerical zero, so a
    policy that is not already near target gets no velocity signal. The
    clipped-linear mode holds a constant gradient across the whole range.
    """
    action = np.zeros(10, dtype=np.float32)
    gaussian = WalkerReward(
        forward_velocity_weight=1.0, target_velocity=1.0, velocity_sigma=0.10
    )
    linear = WalkerReward(
        forward_velocity_weight=1.0, target_velocity=1.0, velocity_mode="clipped_linear"
    )

    def velocity_term(reward: WalkerReward, v: float) -> float:
        return reward.components(
            lin_vel_x=v, pitch_roll_penalty=0.0, action=action, alive=True
        )["velocity"]

    # Gaussian: no usable signal anywhere below target.
    assert velocity_term(gaussian, 0.2) < 1e-10
    assert velocity_term(gaussian, 0.6) < 1e-3

    # Clipped linear: proportional credit, saturating at target.
    assert velocity_term(linear, 0.2) == pytest.approx(0.2)
    assert velocity_term(linear, 0.6) == pytest.approx(0.6)
    assert velocity_term(linear, 1.0) == pytest.approx(1.0)
    assert velocity_term(linear, 1.7) == pytest.approx(1.0)
    # Moving backwards earns nothing rather than a negative spike.
    assert velocity_term(linear, -0.5) == pytest.approx(0.0)


def test_unknown_velocity_mode_is_rejected() -> None:
    reward = WalkerReward(velocity_mode="quadratic")
    with pytest.raises(ValueError, match="Unknown velocity_mode"):
        reward.compute(0.5, 0.0, np.zeros(10, dtype=np.float32), alive=True)


def test_action_rate_penalty_charges_change_not_magnitude() -> None:
    """torque_penalty_weight taxes a held stride; this taxes chatter."""
    reward = WalkerReward(
        torque_penalty_weight=0.0, action_rate_penalty_weight=0.05
    )
    held = reward.components(
        lin_vel_x=0.0,
        pitch_roll_penalty=0.0,
        action=np.full(10, 0.9, dtype=np.float32),
        alive=True,
        action_rate_l2=0.0,
    )
    chattering = reward.components(
        lin_vel_x=0.0,
        pitch_roll_penalty=0.0,
        action=np.zeros(10, dtype=np.float32),
        alive=True,
        action_rate_l2=4.0,
    )

    assert held["action_rate"] == pytest.approx(0.0)
    assert chattering["action_rate"] == pytest.approx(-0.2)


def test_swing_clearance_reward_is_opt_in_and_saturates() -> None:
    reward = WalkerReward(swing_clearance_weight=10.0, swing_clearance_target=0.05)
    action = np.zeros(10, dtype=np.float32)

    def clearance_term(height: float) -> float:
        return reward.components(
            lin_vel_x=0.0,
            pitch_roll_penalty=0.0,
            action=action,
            alive=True,
            swing_clearance=height,
        )["swing_clearance"]

    # The observed 0.7 mm shuffle earns almost nothing; a real step saturates.
    assert clearance_term(0.0007) == pytest.approx(0.007)
    assert clearance_term(0.03) == pytest.approx(0.3)
    assert clearance_term(0.05) == pytest.approx(0.5)
    assert clearance_term(0.20) == pytest.approx(0.5)
    assert clearance_term(-0.01) == pytest.approx(0.0)


def test_touchdown_rate_penalty_only_charges_contacts_faster_than_the_band() -> None:
    reward = WalkerReward(
        touchdown_rate_penalty_weight=1.0, max_cycle_frequency_hz=2.5
    )
    action = np.zeros(10, dtype=np.float32)
    assert reward.min_touchdown_interval == pytest.approx(0.2)

    def rate_term(interval: float | None) -> float:
        return reward.components(
            lin_vel_x=0.0,
            pitch_roll_penalty=0.0,
            action=action,
            alive=True,
            touchdown_interval=interval,
        )["touchdown_rate"]

    assert rate_term(None) == pytest.approx(0.0)      # no touchdown this step
    assert rate_term(0.4) == pytest.approx(0.0)       # 1.25 Hz cycle: fine
    assert rate_term(0.2) == pytest.approx(0.0)       # exactly at the band edge
    assert rate_term(0.1) == pytest.approx(-0.5)      # 5 Hz cycle
    # The measured 6 Hz chatter (~41 ms swings) is charged near-maximally.
    assert rate_term(0.041) == pytest.approx(-0.795)


def test_new_gait_terms_are_zero_by_default() -> None:
    """Existing configs and checkpoints keep bit-for-bit reward semantics."""
    reward = WalkerReward()
    components = reward.components(
        lin_vel_x=0.3,
        pitch_roll_penalty=0.1,
        action=np.full(10, 0.4, dtype=np.float32),
        alive=True,
        action_rate_l2=9.0,
        swing_clearance=0.08,
        touchdown_interval=0.01,
        in_flight=True,
        contact_duty_imbalance=0.9,
    )

    assert reward.velocity_mode == "gaussian"
    assert components["action_rate"] == 0.0
    assert components["swing_clearance"] == 0.0
    assert components["touchdown_rate"] == 0.0
    assert components["flight"] == 0.0
    assert components["gait_symmetry"] == 0.0


def test_flight_penalty_distinguishes_a_walk_from_a_run() -> None:
    """Nothing else in the reward tells a walk from a bound.

    The 10M reward_v2 candidate reached 0.41 m strides and 63 mm clearance
    while spending 54% of steps airborne with 0.7% double support -- it
    satisfied every stride and clearance term by running.
    """
    reward = WalkerReward(flight_penalty_weight=0.5)
    action = np.zeros(10, dtype=np.float32)

    def total(in_flight: bool) -> float:
        return reward.components(
            lin_vel_x=1.0,
            pitch_roll_penalty=0.0,
            action=action,
            alive=True,
            in_flight=in_flight,
        )["flight"]

    assert total(in_flight=False) == pytest.approx(0.0)
    assert total(in_flight=True) == pytest.approx(-0.5)


def test_gait_symmetry_penalty_scales_with_duty_imbalance() -> None:
    reward = WalkerReward(gait_symmetry_penalty_weight=0.5)
    action = np.zeros(10, dtype=np.float32)

    def symmetry(imbalance: float) -> float:
        return reward.components(
            lin_vel_x=1.0,
            pitch_roll_penalty=0.0,
            action=action,
            alive=True,
            contact_duty_imbalance=imbalance,
        )["gait_symmetry"]

    assert symmetry(0.0) == pytest.approx(0.0)
    # The measured candidate ran 30.7% right / 16.0% left -> 0.315 imbalance.
    assert symmetry(0.315) == pytest.approx(-0.1575)
    assert symmetry(1.0) == pytest.approx(-0.5)
    # Out-of-range inputs are clamped rather than extrapolated.
    assert symmetry(4.0) == pytest.approx(-0.5)
    assert symmetry(-1.0) == pytest.approx(0.0)
