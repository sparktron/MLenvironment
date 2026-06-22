"""Tests for arena evaluation."""

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


@patch("rl_framework.training.arena_eval.PPO")
@patch("rl_framework.training.arena_eval.make_env")
def test_arena_eval_returns_dict_with_required_keys(mock_make_env, mock_ppo):
    """Test that run_arena_eval returns a dict with required keys."""
    # Setup mocks
    mock_policy = MagicMock()
    mock_policy.predict.return_value = (
        np.zeros((1, 3), dtype=np.float32),
        None,
    )
    mock_ppo.load.return_value = mock_policy

    mock_env = MagicMock()
    mock_env.agents = ["agent_0", "agent_1"]
    mock_env.reset.return_value = (
        {
            "agent_0": np.zeros(8, dtype=np.float32),
            "agent_1": np.zeros(8, dtype=np.float32),
        },
        {},
    )
    # Step returns immediately with no agents to end episode
    mock_env.step.return_value = (
        {},
        {"agent_0": 0.0, "agent_1": 0.0},
        {},
        {},
        {"agent_0": {"episode_outcome": {"outcome": "ko", "winner": "agent_0"}}},
    )
    # Update agents list after step
    def step_side_effect(*args, **kwargs):
        mock_env.agents = []
        return mock_env.step.return_value

    mock_env.step.side_effect = step_side_effect
    mock_env.action_space.return_value.sample.return_value = np.zeros(
        3, dtype=np.float32
    )
    mock_env.close.return_value = None

    mock_make_env.return_value = mock_env

    # Call run_arena_eval
    result = run_arena_eval(
        "mock_policy", "random", _cfg(), n_episodes=1, swap_roles=False
    )

    # Verify result has required keys
    required_keys = {
        "policy_win_rate",
        "opponent_win_rate",
        "timeout_rate",
        "policy_mean_return",
        "opponent_mean_return",
        "n_episodes",
    }
    assert required_keys.issubset(result.keys())


@patch("rl_framework.training.arena_eval.PPO")
@patch("rl_framework.training.arena_eval.make_env")
def test_arena_eval_n_episodes_equals_total(mock_make_env, mock_ppo):
    """Test that n_episodes in result matches requested episodes."""
    mock_policy = MagicMock()
    mock_policy.predict.return_value = (np.zeros((1, 3), dtype=np.float32), None)
    mock_ppo.load.return_value = mock_policy

    mock_env = MagicMock()
    mock_env.agents = ["agent_0", "agent_1"]
    mock_env.reset.return_value = (
        {
            "agent_0": np.zeros(8, dtype=np.float32),
            "agent_1": np.zeros(8, dtype=np.float32),
        },
        {},
    )

    def step_side_effect(*args, **kwargs):
        mock_env.agents = []
        return (
            {},
            {"agent_0": 0.0, "agent_1": 0.0},
            {},
            {},
            {"agent_0": {"episode_outcome": {"outcome": "ko", "winner": "agent_0"}}},
        )

    mock_env.step.side_effect = step_side_effect
    mock_env.action_space.return_value.sample.return_value = np.zeros(
        3, dtype=np.float32
    )
    mock_env.close.return_value = None
    mock_make_env.return_value = mock_env

    # Test with swap_roles=False
    result = run_arena_eval(
        "mock_policy", "random", _cfg(), n_episodes=5, swap_roles=False
    )
    assert result["n_episodes"] == 5

    # Reset mock for next test
    mock_env.reset.return_value = (
        {
            "agent_0": np.zeros(8, dtype=np.float32),
            "agent_1": np.zeros(8, dtype=np.float32),
        },
        {},
    )
    mock_env.step.side_effect = step_side_effect

    # Test with swap_roles=True
    result = run_arena_eval(
        "mock_policy", "random", _cfg(), n_episodes=5, swap_roles=True
    )
    assert result["n_episodes"] == 10
