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
