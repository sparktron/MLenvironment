from __future__ import annotations

import re
from pathlib import Path

import gymnasium as gym
import numpy as np

from rl_framework.training import eval_runner
from rl_framework.training.evaluation_workers import (
    EpisodeSeedWrapper,
    episode_quotas,
    resolve_evaluation_workers,
)
from rl_framework.training.walker_diagnostics import _rollouts


class _SeedRecordingEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self) -> None:
        self.seeds: list[int | None] = []

    def reset(self, *, seed=None, options=None):
        self.seeds.append(seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}


class _OneStepVecEnv:
    num_envs = 3

    def __init__(self) -> None:
        self.seed_base: int | None = None
        self.step_calls = 0
        self.closed = False

    def env_method(self, method_name, *args, **kwargs):
        assert method_name == "set_episode_seed_base"
        self.seed_base = int(args[0])
        return [None] * self.num_envs

    def reset(self):
        return np.zeros((self.num_envs, 1), dtype=np.float32)

    def step(self, action):
        self.step_calls += 1
        return (
            np.zeros((self.num_envs, 1), dtype=np.float32),
            np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            np.ones(self.num_envs, dtype=bool),
            [{}, {"TimeLimit.truncated": True}, {}],
        )

    def close(self):
        self.closed = True


def test_evaluation_workers_are_bounded_by_episode_count(monkeypatch) -> None:
    monkeypatch.setattr(
        "rl_framework.training.evaluation_workers.os.cpu_count", lambda: 12
    )
    assert resolve_evaluation_workers({}, episodes=5) == 5
    assert resolve_evaluation_workers({"evaluation": {"workers": 3}}, episodes=5) == 3
    assert episode_quotas(8, 3) == [3, 3, 2]


def test_episode_seed_wrapper_uses_strided_seed_stream() -> None:
    env = _SeedRecordingEnv()
    wrapped = EpisodeSeedWrapper(env, worker_index=1, worker_count=3)
    wrapped.set_episode_seed_base(10)
    wrapped.reset()
    wrapped.reset()
    assert env.seeds == [11, 14]


def test_eval_runner_uses_parallel_workers_and_exact_episode_count(
    tmp_path: Path, monkeypatch
) -> None:
    vec_env = _OneStepVecEnv()
    created: dict[str, int] = {}

    def _make_vec_env(env_type, env_cfg, workers, training_cfg):
        created["workers"] = workers
        return vec_env

    class _ZeroPolicy:
        def predict(self, obs, deterministic=True):
            return np.zeros((len(obs), 1), dtype=np.float32), None

    monkeypatch.setattr(eval_runner, "make_seeded_vec_env", _make_vec_env)
    monkeypatch.setattr(eval_runner.PPO, "load", lambda *args, **kwargs: _ZeroPolicy())
    monkeypatch.setattr(
        eval_runner, "find_vecnormalize_path_for_model", lambda model_path: None
    )
    cfg = {
        "experiment_name": "parallel_eval",
        "seed": 7,
        "output": {"base_dir": str(tmp_path)},
        "environment": {"type": "walker_bullet"},
        "training": {"algorithm": "PPO", "device": "cpu"},
        "evaluation": {"episodes": 5, "workers": 3},
    }

    metrics = eval_runner.evaluate(cfg, "unused.zip")

    assert created["workers"] == 3
    assert vec_env.seed_base == 7
    assert vec_env.step_calls == 2
    assert vec_env.closed is True
    assert metrics["mean_return"] == 1.8
    assert metrics["terminated_rate"] == 0.6
    assert metrics["truncated_rate"] == 0.4


def test_walker_diagnostics_collects_each_parallel_episode_once() -> None:
    vec_env = _OneStepVecEnv()
    result = _rollouts(
        vec_env,
        episodes=5,
        seed=9,
        model=None,
        deterministic=True,
    )
    assert vec_env.seed_base == 9
    assert result["episode_length_mean"] == 1.0
    assert result["return_mean"] == 1.8


def test_transfer_suite_resolves_obstacles_below_the_saturation_floor() -> None:
    """The obstacle heights must include the band where a policy can respond.

    Measured 2026-08-13 across six checkpoints (v9 and v12, three seeds each):
    the failure cliff sits between 2 and 3 cm and every policy saturates at
    ~115 steps / 100% falls from 4 cm upward. A suite that probes obstacles
    only at 10 cm reports "unchanged" for a policy that genuinely improved --
    which is what happened to v11 and v12 -- because 10 cm is deep in the
    floor. At least one obstacle height must sit at or below 3 cm.
    """
    import inspect

    from rl_framework.training import walker_diagnostics

    source = inspect.getsource(walker_diagnostics.evaluate_walker_transfer_suite)
    heights = [
        float(m)
        for m in re.findall(r'"obstacle_height":\s*([0-9.]+)', source)
    ]
    assert heights, "transfer suite defines no obstacle terrains"
    assert min(heights) <= 0.03, (
        "every obstacle height is above the 2-3 cm cliff, so the suite cannot "
        f"resolve obstacle progress; got {sorted(heights)}"
    )
    # The 10 cm point is the historical series shared with v9/v11/v12 and must
    # survive, or those results stop being comparable.
    assert 0.10 in heights
