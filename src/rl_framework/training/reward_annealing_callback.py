"""Anneal the arena's dense per-hit reward toward the sparse win/loss signal."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class RewardAnnealingCallback(BaseCallback):
    """Linearly anneal the arena damage-reward scale from 1.0 to 0.0.

    The dense per-hit reward teaches agents to spam attacks for immediate
    reward rather than to actually win matches. This callback ramps that dense
    reward down over the first ``anneal_steps`` timesteps, after which only the
    terminal ``+1 / -1`` win/loss signal drives learning. The health damage
    itself is unaffected, so combat still resolves throughout.

    Pushes the current scale once at each rollout boundary. This keeps the
    schedule smooth at training timescales while avoiding a cross-process
    ``env_method`` call for every vectorized environment step.
    """

    def __init__(self, anneal_steps: int = 500_000, verbose: int = 0) -> None:
        super().__init__(verbose)
        if anneal_steps <= 0:
            raise ValueError(f"anneal_steps must be > 0, got {anneal_steps}")
        self.anneal_steps = int(anneal_steps)
        self._last_scale: float | None = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        scale = max(0.0, 1.0 - self.num_timesteps / self.anneal_steps)
        # Avoid redundant calls once fully annealed or when no timesteps moved.
        if scale == self._last_scale:
            return
        if self.training_env is not None:
            self.training_env.env_method(
                "update_live_params", {"reward.damage_scale": scale}
            )
        self._last_scale = scale
        if self.verbose >= 1 and self.logger is not None:
            self.logger.record("arena/damage_reward_scale", scale)
