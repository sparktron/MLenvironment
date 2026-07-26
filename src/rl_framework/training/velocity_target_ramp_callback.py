"""Fixed-time target-velocity schedule for controlled walker studies."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class VelocityTargetRampCallback(BaseCallback):
    """Linearly ramp a walker's velocity target at rollout boundaries.

    The callback deliberately uses training timesteps rather than reward or
    performance gates, keeping the schedule identical for every seed.
    """

    def __init__(self, *, start: float, end: float, ramp_steps: int) -> None:
        super().__init__(verbose=0)
        if start < 0 or end < start:
            raise ValueError("velocity targets must satisfy 0 <= start <= end")
        if ramp_steps <= 0:
            raise ValueError("ramp_steps must be positive")
        self.start = float(start)
        self.end = float(end)
        self.ramp_steps = int(ramp_steps)
        self._last_target: float | None = None
        self._start_timestep: int | None = None

    def _on_training_start(self) -> None:
        # Resumed SB3 models retain their prior global timestep count. Measure
        # from this continuation's start so a 50k ramp remains a 50k ramp.
        self._start_timestep = int(self.num_timesteps)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        start = self._start_timestep if self._start_timestep is not None else 0
        elapsed = max(0, int(self.num_timesteps) - start)
        fraction = min(elapsed / self.ramp_steps, 1.0)
        target = self.start + (self.end - self.start) * fraction
        if target == self._last_target:
            return
        if self.training_env is not None:
            self.training_env.env_method(
                "update_live_params", {"reward.target_velocity": target}
            )
        self.logger.record("walker/target_velocity", target)
        self._last_target = target
