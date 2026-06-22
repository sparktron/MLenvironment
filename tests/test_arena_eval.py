"""Minimal tests for arena evaluation module."""

from unittest.mock import MagicMock, patch

import numpy as np

from rl_framework.training.arena_eval import run_arena_eval


@patch("rl_framework.training.arena_eval.make_env")
@patch("rl_framework.training.arena_eval.PPO")
def test_arena_eval_basic(mock_ppo, mock_make_env):
    """Basic test that run_arena_eval completes and returns expected structure."""
    # Mock the policy
    mock_policy = MagicMock()
    mock_policy.predict.return_value = (np.array([[0.0, 0.0, 0.0]]), None)
    mock_ppo.load.return_value = mock_policy

    # Mock the environment
    mock_env = MagicMock()

    # Setup reset to return initial observation
    mock_env.reset.return_value = (
        {"agent_0": np.zeros(8), "agent_1": np.zeros(8)},
        {},
    )

    # Setup step to return observations and end episode on first step
    def step_fn(actions):
        mock_env.agents = []
        return (
            {},
            {"agent_0": 1.0, "agent_1": 0.0},
            {},
            {},
            {"agent_0": {"episode_outcome": {"outcome": "ko", "winner": "agent_0"}}},
        )

    mock_env.agents = ["agent_0", "agent_1"]
    mock_env.step.side_effect = step_fn
    mock_env.action_space.return_value.sample.return_value = np.array([0.0, 0.0, 0.0])
    mock_env.close.return_value = None

    # Make make_env return our mock
    mock_make_env.return_value = mock_env

    # Run the evaluation
    result = run_arena_eval(
        "policy.zip",
        "random",
        {"environment": {"type": "organism_arena_parallel"}},
        n_episodes=2,
        swap_roles=False,
    )

    # Check structure
    assert isinstance(result, dict)
    assert "n_episodes" in result
    assert result["n_episodes"] == 2
    assert "policy_win_rate" in result
    assert "opponent_win_rate" in result
    assert "timeout_rate" in result
