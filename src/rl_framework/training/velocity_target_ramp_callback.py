"""Fixed-time target-velocity schedule for controlled walker studies."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class VelocityTargetRampCallback(BaseCallback):
    """Linearly ramp a walker's velocity target at rollout boundaries.

    The callback deliberately uses training timesteps rather than reward or
    performance gates, keeping the schedule identical for every seed.
    """

    def __init__(
        self,
        *,
        start: float,
        end: float,
        ramp_steps: int,
        sigma_fraction: float | None = None,
    ) -> None:
        super().__init__(verbose=0)
        if start < 0 or end < start:
            raise ValueError("velocity targets must satisfy 0 <= start <= end")
        if ramp_steps <= 0:
            raise ValueError("ramp_steps must be positive")
        if sigma_fraction is not None and sigma_fraction <= 0:
            raise ValueError("sigma_fraction must be positive when set")
        self.start = float(start)
        self.end = float(end)
        self.ramp_steps = int(ramp_steps)
        # A Gaussian velocity reward whose sigma is fixed while the target
        # moves becomes unreachable: sigma 0.10 calibrated at a 0.15 m/s
        # target pays 3.3e-4 once the target reaches 1.0 m/s and the policy is
        # still at 0.6, so the velocity gradient vanishes mid-ramp. Setting
        # sigma_fraction rescales sigma to `fraction * target` at every step,
        # keeping the basin of attraction proportional to the target.
        # Leave it None for `velocity_mode: clipped_linear`, which has no sigma.
        self.sigma_fraction = None if sigma_fraction is None else float(sigma_fraction)
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
        params: dict[str, float] = {"reward.target_velocity": target}
        if self.sigma_fraction is not None:
            params["reward.velocity_sigma"] = self.sigma_fraction * target
        if self.training_env is not None:
            self.training_env.env_method("update_live_params", params)
        self.logger.record("walker/target_velocity", target)
        if self.sigma_fraction is not None:
            self.logger.record("walker/velocity_sigma", params["reward.velocity_sigma"])
        self._last_target = target
