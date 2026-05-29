from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from rl_framework.envs.registry import make_env


def run_arena_eval(
    policy_path: str,
    opponent_path: str,
    cfg: dict[str, Any],
    n_episodes: int = 100,
    swap_roles: bool = True,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Evaluate policy vs opponent over N episodes.

    Parameters
    ----------
    policy_path:
        Path to a saved PPO model zip for the evaluated policy.
    opponent_path:
        Path to a saved PPO model zip, or "random" for a random-action baseline.
    cfg:
        Full experiment config dict (must contain an "environment" key) or a
        bare environment config dict.
    n_episodes:
        Total episodes to run.  When swap_roles=True this is split evenly.
    swap_roles:
        When True, runs n_episodes/2 with policy as agent_0 and n_episodes/2
        with policy as agent_1 to control for positional spawn bias.
    output_path:
        Optional JSON path to write results.

    Returns
    -------
    dict with keys: policy_win_rate, opponent_win_rate, timeout_rate,
    policy_mean_return, opponent_mean_return, n_episodes.
    """
    policy = PPO.load(policy_path)
    opponent: PPO | None = (
        None if opponent_path == "random" else PPO.load(opponent_path)
    )

    env_cfg: dict[str, Any] = cfg.get("environment", cfg)

    def _run_block(n: int, policy_as_agent_0: bool) -> tuple[dict, int, list, list]:
        policy_agent = "agent_0" if policy_as_agent_0 else "agent_1"
        opponent_agent = "agent_1" if policy_as_agent_0 else "agent_0"
        wins: dict[str, int] = {"policy": 0, "opponent": 0}
        timeouts = 0
        policy_returns: list[float] = []
        opponent_returns: list[float] = []

        for _ in range(n):
            env = make_env(env_cfg.get("type", "organism_arena_parallel"), env_cfg)
            obs, _ = env.reset()
            ep_ret: dict[str, float] = {"policy": 0.0, "opponent": 0.0}
            last_infos: dict = {}

            while env.agents:
                actions: dict[str, np.ndarray] = {}
                for agent, o in obs.items():
                    if agent == policy_agent:
                        action, _ = policy.predict(o[np.newaxis], deterministic=True)
                        actions[agent] = action[0]
                    elif opponent is not None:
                        action, _ = opponent.predict(o[np.newaxis], deterministic=True)
                        actions[agent] = action[0]
                    else:
                        actions[agent] = env.action_space(agent).sample()

                obs, rewards, terminations, truncations, last_infos = env.step(actions)
                ep_ret["policy"] += rewards.get(policy_agent, 0.0)
                ep_ret["opponent"] += rewards.get(opponent_agent, 0.0)

            for info in last_infos.values():
                if "episode_outcome" in info:
                    outcome = info["episode_outcome"]
                    if outcome["outcome"] == "ko":
                        if outcome.get("winner") == policy_agent:
                            wins["policy"] += 1
                        else:
                            wins["opponent"] += 1
                    else:
                        timeouts += 1
                    break

            policy_returns.append(ep_ret["policy"])
            opponent_returns.append(ep_ret["opponent"])
            env.close()

        return wins, timeouts, policy_returns, opponent_returns

    if swap_roles:
        w1, t1, pr1, or1 = _run_block(n_episodes, policy_as_agent_0=True)
        w2, t2, pr2, or2 = _run_block(n_episodes, policy_as_agent_0=False)
        total = n_episodes * 2
        policy_wins = w1["policy"] + w2["policy"]
        opponent_wins = w1["opponent"] + w2["opponent"]
        all_timeouts = t1 + t2
        all_policy_returns = pr1 + pr2
        all_opponent_returns = or1 + or2
    else:
        wins, timeouts, all_policy_returns, all_opponent_returns = _run_block(
            n_episodes, policy_as_agent_0=True
        )
        total = n_episodes
        policy_wins = wins["policy"]
        opponent_wins = wins["opponent"]
        all_timeouts = timeouts

    result: dict[str, Any] = {
        "policy_win_rate": policy_wins / total if total > 0 else 0.0,
        "opponent_win_rate": opponent_wins / total if total > 0 else 0.0,
        "timeout_rate": all_timeouts / total if total > 0 else 0.0,
        "policy_mean_return": float(np.mean(all_policy_returns))
        if all_policy_returns
        else 0.0,
        "opponent_mean_return": float(np.mean(all_opponent_returns))
        if all_opponent_returns
        else 0.0,
        "n_episodes": total,
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))

    return result
