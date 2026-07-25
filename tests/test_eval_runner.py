from pathlib import Path

import numpy as np

from rl_framework.training import eval_runner
from rl_framework.training.eval_runner import _was_truncated


def test_was_truncated_true_when_timelimit_flag_present() -> None:
    assert _was_truncated([{"TimeLimit.truncated": True}]) is True


def test_was_truncated_true_for_arena_timeout_outcome() -> None:
    assert _was_truncated(
        [{"episode_outcome": {"winner": None, "outcome": "timeout", "step": 2}}]
    ) is True


def test_was_truncated_false_for_missing_or_false_flags() -> None:
    assert _was_truncated([{}]) is False
    assert _was_truncated([{"TimeLimit.truncated": False}]) is False
    assert _was_truncated({}) is False


def test_arena_evaluate_counts_max_step_timeout_as_truncated(
    tmp_path: Path, monkeypatch
) -> None:
    loaded: dict[str, str] = {}
    predict_calls = 0

    class _ZeroPolicy:
        def predict(self, obs, deterministic=True):
            nonlocal predict_calls
            predict_calls += 1
            obs_array = np.asarray(obs)
            if obs_array.ndim >= 2:
                return np.zeros((obs_array.shape[0], 3), dtype=np.float32), None
            return np.zeros(3, dtype=np.float32), None

    def _load(_path, *, device):
        loaded["device"] = device
        return _ZeroPolicy()

    monkeypatch.setattr(eval_runner.PPO, "load", _load)
    cfg = {
        "experiment_name": "arena_eval_timeout",
        "seed": 0,
        "output": {"base_dir": str(tmp_path)},
        "environment": {
            "type": "organism_arena_parallel",
            "battle_rules": {
                "max_steps": 2,
                "damage": 0.2,
                "attack_range": 0.5,
                "cooldown_steps": 0,
            },
        },
    }

    metrics = eval_runner.evaluate(cfg, "unused.zip")

    assert metrics["truncated_rate"] == 1.0
    assert metrics["terminated_rate"] == 0.0
    assert loaded["device"] == "cpu"
    assert predict_calls == 10  # five default episodes x two steps
