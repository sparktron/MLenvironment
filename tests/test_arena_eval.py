from unittest.mock import MagicMock, patch

import numpy as np

from rl_framework.training.arena_eval import run_arena_eval


def _cfg(max_steps=5):
    return {
        "environment": {
            "type": "organism_arena_parallel",
            "seed": 0,
            "battle_rules": {"max_steps": max_steps},
        }
    }


def _mock_policy():
    p = MagicMock()
    p.predict.return_value = (np.zeros((1, 3), dtype=np.float32), None)
    return p


def _make_mock_env():
    """Factory function to create fresh mock environments for each test."""
    def _mock_env():
        env = MagicMock()
        obs_dict = {
            "agent_0": np.zeros(8, dtype=np.float32),
            "agent_1": np.zeros(8, dtype=np.float32),
        }

        # Track call count for side effects
        env._reset_called = False

        def reset_impl(*args, **kwargs):
            env._reset_called = True
            env.agents = ["agent_0", "agent_1"]
            return (obs_dict, {})

        def step_impl(actions):
            env.agents = []
            return (
                obs_dict,
                {"agent_0": 0.0, "agent_1": 0.0},
                {},
                {},
                {"agent_0": {"episode_outcome": {"outcome": "ko", "winner": "agent_0"}}},
            )

        env.reset.side_effect = reset_impl
        env.step.side_effect = step_impl
        env.action_space.return_value.sample.return_value = np.zeros(
            3, dtype=np.float32
        )
        env.close = MagicMock(return_value=None)
        return env

    return _mock_env()


@patch("rl_framework.training.arena_eval.PPO")
@patch("rl_framework.training.arena_eval.make_env")
def test_arena_eval_returns_correct_keys(mock_make_env, mock_ppo):
    mock_ppo.load.return_value = _mock_policy()
    mock_make_env.side_effect = lambda *args, **kwargs: _make_mock_env()
    result = run_arena_eval(
        "mock_policy", "random", _cfg(), n_episodes=4, swap_roles=False
    )
    assert {
        "policy_win_rate",
        "opponent_win_rate",
        "timeout_rate",
        "policy_mean_return",
        "opponent_mean_return",
        "n_episodes",
    }.issubset(result.keys())


@patch("rl_framework.training.arena_eval.PPO")
@patch("rl_framework.training.arena_eval.make_env")
def test_arena_eval_win_rates_sum_to_lte_one(mock_make_env, mock_ppo):
    mock_ppo.load.return_value = _mock_policy()
    mock_make_env.side_effect = lambda *args, **kwargs: _make_mock_env()
    result = run_arena_eval(
        "mock_policy", "random", _cfg(), n_episodes=10, swap_roles=False
    )
    assert result["policy_win_rate"] + result["opponent_win_rate"] <= 1.0 + 1e-6


@patch("rl_framework.training.arena_eval.PPO")
@patch("rl_framework.training.arena_eval.make_env")
def test_arena_eval_role_swap_doubles_episode_count(mock_make_env, mock_ppo):
    mock_ppo.load.return_value = _mock_policy()
    mock_make_env.side_effect = lambda *args, **kwargs: _make_mock_env()
    result = run_arena_eval(
        "mock_policy", "random", _cfg(), n_episodes=10, swap_roles=True
    )
    assert result["n_episodes"] == 20


@patch("rl_framework.training.arena_eval.PPO")
@patch("rl_framework.training.arena_eval.make_env")
def test_arena_eval_random_opponent_no_load(mock_make_env, mock_ppo):
    """Passing opponent_path='random' must not call PPO.load for the opponent."""
    mock_ppo.load.return_value = _mock_policy()
    mock_make_env.side_effect = lambda *args, **kwargs: _make_mock_env()
    run_arena_eval("mock_policy", "random", _cfg(), n_episodes=2, swap_roles=False)
    # PPO.load called exactly once (for the policy, not the opponent)
    assert mock_ppo.load.call_count == 1
