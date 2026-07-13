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

