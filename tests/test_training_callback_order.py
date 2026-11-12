"""Pin the callback order at the train() boundary before moving task code."""

import pytest

from rl_framework.training.sb3_runner import PPO, train


@pytest.mark.parametrize(
    ("env_type", "options", "expected"),
    [
        pytest.param(
            "walker_bullet",
            {
                "evaluation": {
                    "best_model": {"enabled": True, "episodes": 1, "workers": 1}
                },
                "curriculum": {"enabled": True},
            },
            [
                "VecNormalizeCheckpointCallback",
                "EvalCallback",
                "WalkerMetricsCallback",
                "CurriculumCallback",
            ],
            id="walker",
        ),
        pytest.param(
            "organism_arena_parallel",
            {
                "self_play": {"enabled": True},
                "reward_annealing": {"enabled": True},
                "curriculum": {"enabled": True, "metric": "arena/agent_0_win_rate"},
            },
            [
                "VecNormalizeCheckpointCallback",
                "ArenaMetricsCallback",
                "RewardAnnealingCallback",
                "CurriculumCallback",
                "SelfPlayCallback",
            ],
            id="arena-self-play",
        ),
        pytest.param(
            "organism_arena_parallel",
            {"self_play": {"enabled": False}},
            ["VecNormalizeCheckpointCallback", "ArenaMetricsCallback"],
            id="arena-shared-policy",
        ),
    ],
)
def test_training_callback_order(tmp_path, monkeypatch, env_type, options, expected):
    actual = []

    def capture_learn(self, *, callback, **kwargs):
        actual.extend(type(item).__name__ for item in callback)
        return self

    monkeypatch.setattr(PPO, "learn", capture_learn)
    train(
        {
            "experiment_name": "callback_order",
            "seed": 11,
            "environment": {"type": env_type},
            "output": {"base_dir": str(tmp_path)},
            "training": {
                "total_timesteps": 32,
                "num_envs": 1,
                "n_steps": 32,
                "batch_size": 32,
                "device": "cpu",
                "torch_num_threads": 1,
            },
            **options,
        }
    )

    assert actual == expected
