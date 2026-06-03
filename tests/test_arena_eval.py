"""Tests for the head-to-head arena eval harness and FrozenPolicy normalisation."""

from __future__ import annotations

import numpy as np

from rl_framework.training.arena_eval import run_arena_eval
from rl_framework.training.self_play_env_wrapper import (
    FrozenPolicy,
    RandomPolicy,
    load_frozen_policy,
)


def _cfg() -> dict:
    return {
        "experiment_name": "eval_arena",
        "seed": 0,
        "environment": {
            "type": "organism_arena_parallel",
            "sim": {"arena_half_extent": 1.0},
            "battle_rules": {
                "max_steps": 12,
                "damage": 0.3,
                "attack_range": 0.6,
                "cooldown_steps": 0,
            },
        },
    }


# -- run_arena_eval ----------------------------------------------------------


def test_arena_eval_returns_expected_keys() -> None:
    result = run_arena_eval("random", "random", _cfg(), n_episodes=5, swap_roles=False)
    assert {
        "policy_win_rate",
        "opponent_win_rate",
        "timeout_rate",
        "policy_mean_return",
        "opponent_mean_return",
        "n_episodes",
    }.issubset(result.keys())


def test_arena_eval_rates_are_a_partition() -> None:
    result = run_arena_eval("random", "random", _cfg(), n_episodes=10, swap_roles=False)
    total = (
        result["policy_win_rate"] + result["opponent_win_rate"] + result["timeout_rate"]
    )
    assert abs(total - 1.0) < 1e-6
    assert result["policy_win_rate"] + result["opponent_win_rate"] <= 1.0 + 1e-6


def test_arena_eval_role_swap_doubles_episode_count() -> None:
    result = run_arena_eval("random", "random", _cfg(), n_episodes=10, swap_roles=True)
    assert result["n_episodes"] == 20


def test_arena_eval_no_swap_runs_n_episodes() -> None:
    result = run_arena_eval("random", "random", _cfg(), n_episodes=7, swap_roles=False)
    assert result["n_episodes"] == 7


def test_arena_eval_writes_output_json(tmp_path) -> None:
    import json

    out = tmp_path / "evals" / "result.json"
    run_arena_eval(
        "random", "random", _cfg(), n_episodes=3, swap_roles=False, output_path=str(out)
    )
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["n_episodes"] == 3


# -- FrozenPolicy / loaders --------------------------------------------------


def test_load_frozen_policy_random_returns_random_policy() -> None:
    from rl_framework.envs.registry import make_env

    env = make_env("organism_arena_parallel", {"type": "organism_arena_parallel"})
    actor = load_frozen_policy("random", env.action_space("agent_0"))
    assert isinstance(actor, RandomPolicy)
    action, state = actor.predict(np.zeros(7, dtype=np.float32))
    assert action.shape == (3,)
    assert state is None


def test_frozen_policy_applies_normalizer_before_predict() -> None:
    seen = {}

    class _Norm:
        def normalize_obs(self, obs):
            seen["normalized"] = True
            return obs + 100.0

    class _Model:
        def predict(self, obs, deterministic=True):
            seen["obs"] = np.asarray(obs)
            return np.zeros(3, dtype=np.float32), None

    fp = FrozenPolicy(_Model(), _Norm())
    fp.predict(np.zeros(7, dtype=np.float32))
    assert seen.get("normalized") is True
    assert float(seen["obs"][0]) == 100.0  # normaliser ran before predict


def test_frozen_policy_without_normalizer_passes_raw_obs() -> None:
    captured = {}

    class _Model:
        def predict(self, obs, deterministic=True):
            captured["obs"] = np.asarray(obs)
            return np.zeros(3, dtype=np.float32), None

    fp = FrozenPolicy(_Model(), None)
    raw = np.arange(7, dtype=np.float32)
    fp.predict(raw)
    assert np.array_equal(captured["obs"], raw)
