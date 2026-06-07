from __future__ import annotations


def test_live_tuning_forwards_all_accepted_env_sections() -> None:
    from rl_framework.training.live_tuning_callback import LiveTuningCallback

    class DummyTrainingEnv:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def env_method(self, method_name: str, params: dict) -> None:
            self.calls.append((method_name, params))

    class DummyModel:
        def __init__(self, env: DummyTrainingEnv) -> None:
            self._env = env

        def get_env(self) -> DummyTrainingEnv:
            return self._env

    event = {
        "reward.alive_bonus": 2.0,
        "termination.max_steps": 25,
        "domain_randomization.sensor_noise_std": 0.1,
        "battle_rules.damage": 0.2,
        "morphology.base_size": 1.3,
        "sim.arena_half_extent": 2.0,
    }
    env_cfg = {
        "reward": {"alive_bonus": 1.0},
        "termination": {"max_steps": 10},
        "domain_randomization": {"sensor_noise_std": 0.0},
        "battle_rules": {"damage": 0.1},
        "morphology": {"base_size": 1.0},
        "sim": {"arena_half_extent": 1.0},
    }
    training_env = DummyTrainingEnv()
    callback = LiveTuningCallback(env_cfg, pop_tuning_event=lambda: event)
    callback.model = DummyModel(training_env)

    callback._on_rollout_end()

    assert training_env.calls == [("update_live_params", event)]
    assert env_cfg["domain_randomization"]["sensor_noise_std"] == 0.1
    assert env_cfg["battle_rules"]["damage"] == 0.2
    assert env_cfg["morphology"]["base_size"] == 1.3
    assert env_cfg["sim"]["arena_half_extent"] == 2.0
