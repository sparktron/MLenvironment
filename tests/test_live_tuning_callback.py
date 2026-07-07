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


def test_live_tuning_status_reads_ep_info_buffer_and_omits_absent_metrics() -> None:
    """Status snapshots must take episode stats from the model's
    ep_info_buffer (the logger never holds rollout/* at rollout end) and
    omit absent metrics instead of reporting the defaultdict's 0.0."""
    from collections import defaultdict
    from types import SimpleNamespace

    from rl_framework.training.live_tuning_callback import LiveTuningCallback

    published: list[dict] = []
    callback = LiveTuningCallback(
        {}, pop_tuning_event=lambda: None, publish_status=published.append
    )
    logger = SimpleNamespace(name_to_value=defaultdict(float, {"train/loss": 0.5}))
    callback.model = SimpleNamespace(
        get_env=lambda: None,
        logger=logger,
        ep_info_buffer=[{"r": 12.0, "l": 30}, {"r": 18.0, "l": 50}],
    )
    callback.num_timesteps = 123

    callback._on_rollout_end()

    assert published, "a status snapshot must publish even with no tuning event"
    status = published[-1]
    assert status["timesteps"] == 123
    assert status["rollout/ep_rew_mean"] == 15.0
    assert status["rollout/ep_len_mean"] == 40.0
    assert status["train/loss"] == 0.5
    assert "train/value_loss" not in status, "absent metrics are omitted, not 0.0"
    assert "train/value_loss" not in logger.name_to_value, (
        "probing a metric must not insert it into the logger map"
    )
