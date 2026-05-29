from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class RewardAnnealingCallback(BaseCallback):
    """Linearly anneals the arena damage reward scale from 1.0 to 0.0.

    After anneal_steps total env steps, damage rewards are zeroed and only
    the terminal win/loss signal (+1 / -1) drives learning.
    """

    def __init__(self, anneal_steps: int = 500_000) -> None:
        super().__init__()
        self.anneal_steps = anneal_steps

    def _on_step(self) -> bool:
        scale = max(0.0, 1.0 - self.num_timesteps / self.anneal_steps)
        self.training_env.env_method(
            "update_live_params", {"reward.damage_scale": scale}
        )
        return True
