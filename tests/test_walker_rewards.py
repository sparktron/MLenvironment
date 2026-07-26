from __future__ import annotations

import numpy as np

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

    assert set(components) == {"alive", "velocity", "orientation", "action", "fall"}
    assert sum(components.values()) == reward.compute(**kwargs)
    assert components["fall"] == -reward.fall_penalty
