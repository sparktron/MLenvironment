from __future__ import annotations

from rl_framework.training.velocity_target_ramp_callback import VelocityTargetRampCallback


class _Env:
    def __init__(self) -> None:
        self.calls: list[dict[str, float]] = []

    def env_method(self, _method: str, params: dict[str, float]) -> None:
        self.calls.append(params)


class _Model:
    def __init__(self, env: _Env) -> None:
        self._env = env
        self.logger = _Logger()

    def get_env(self) -> _Env:
        return self._env


class _Logger:
    def record(self, _key: str, _value: float) -> None:
        pass


def test_velocity_ramp_measures_elapsed_continuation_steps() -> None:
    callback = VelocityTargetRampCallback(start=0.15, end=0.25, ramp_steps=50_000)
    env = _Env()
    callback.init_callback(_Model(env))
    callback.num_timesteps = 100_000
    callback._on_training_start()

    callback.num_timesteps = 125_000
    callback._on_rollout_end()

    assert env.calls == [{"reward.target_velocity": 0.2}]


def test_velocity_ramp_rescales_sigma_with_the_moving_target() -> None:
    """A fixed sigma makes the Gaussian unreachable once the target moves.

    At sigma 0.10 with the target ramped to 1.0 the reward at 0.6 m/s is
    3.3e-4, so the velocity gradient vanishes mid-ramp. Scaling sigma with the
    target keeps the basin of attraction proportional.
    """
    callback = VelocityTargetRampCallback(
        start=0.15, end=1.0, ramp_steps=100_000, sigma_fraction=0.5
    )
    env = _Env()
    callback.init_callback(_Model(env))
    callback.num_timesteps = 0
    callback._on_training_start()

    callback.num_timesteps = 100_000
    callback._on_rollout_end()

    assert env.calls == [{"reward.target_velocity": 1.0, "reward.velocity_sigma": 0.5}]


def test_velocity_ramp_omits_sigma_when_not_requested() -> None:
    """clipped_linear has no sigma, so the ramp must not invent one."""
    callback = VelocityTargetRampCallback(start=0.15, end=1.0, ramp_steps=100_000)
    env = _Env()
    callback.init_callback(_Model(env))
    callback.num_timesteps = 0
    callback._on_training_start()

    callback.num_timesteps = 100_000
    callback._on_rollout_end()

    assert env.calls == [{"reward.target_velocity": 1.0}]
