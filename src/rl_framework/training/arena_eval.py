"""Head-to-head evaluation for the organism arena.

``eval_runner.evaluate`` drives both arena slots with one shared policy, so it
cannot answer "does checkpoint A beat checkpoint B?". :func:`run_arena_eval`
pits a policy against a specific opponent (another checkpoint or a random
baseline) over many episodes and reports win rates and mean returns, optionally
swapping spawn slots to cancel positional bias.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rl_framework.envs.registry import make_env
from rl_framework.training.self_play_env_wrapper import load_frozen_policy

_AGENTS = ("agent_0", "agent_1")


def _play_episode(env: Any, policy: Any, opponent: Any, policy_slot: str, seed: int):
    """Run one episode; return (outcome_for_policy, policy_return, opponent_return).

    ``outcome_for_policy`` is ``"win"``, ``"loss"``, or ``"timeout"`` from the
    policy's perspective.
    """
    opp_slot = _AGENTS[1] if policy_slot == _AGENTS[0] else _AGENTS[0]
    observations, _ = env.reset(seed=seed)
    returns = {a: 0.0 for a in _AGENTS}
    final_outcome: dict[str, Any] | None = None

    while env.agents:
        actions = {}
        for agent, obs in observations.items():
            actor = policy if agent == policy_slot else opponent
            action, _ = actor.predict(obs, deterministic=True)
            actions[agent] = np.asarray(action, dtype=np.float32)
        observations, rewards, _, _, infos = env.step(actions)
        for agent, reward in rewards.items():
            returns[agent] += float(reward)
        for info in infos.values():
            if "episode_outcome" in info:
                final_outcome = info["episode_outcome"]

    if final_outcome is None or final_outcome.get("outcome") == "timeout":
        result = "timeout"
    elif final_outcome.get("outcome") == "draw":
        # Simultaneous knockout — neither side wins.
        result = "draw"
    elif final_outcome.get("winner") == policy_slot:
        result = "win"
    else:
        result = "loss"
    return result, returns[policy_slot], returns[opp_slot]


def run_arena_eval(
    policy_path: str,
    opponent_path: str,
    cfg: dict,
    n_episodes: int = 100,
    swap_roles: bool = True,
    output_path: str | None = None,
) -> dict:
    """Evaluate *policy_path* against *opponent_path* in the arena.

    Parameters
    ----------
    policy_path, opponent_path:
        Paths to saved PPO checkpoints. Either may be the string ``"random"``
        for a random-action baseline. A sibling ``vecnormalize.pkl`` (or a
        ``<stem>_vecnorm.pkl`` league sidecar) is applied automatically so each
        policy sees the observation distribution it trained under.
    n_episodes:
        Episodes per spawn orientation. With ``swap_roles=True`` the harness
        runs ``n_episodes`` with the policy as ``agent_0`` and another
        ``n_episodes`` with it as ``agent_1`` (``2 * n_episodes`` total) to
        cancel positional bias; otherwise it runs ``n_episodes`` as ``agent_0``.
    output_path:
        If given, the result dict is written there as JSON.

    Returns a dict with ``policy_win_rate``, ``opponent_win_rate``,
    ``draw_rate``, ``timeout_rate``, ``policy_mean_return``,
    ``opponent_mean_return``, and ``n_episodes`` (the total number actually
    run). Draws (simultaneous knockouts) count toward neither side's win rate.
    """
    env_cfg = cfg["environment"]
    env = make_env(env_cfg["type"], env_cfg)
    base_seed = int(cfg.get("seed", 0))

    # Spaces are slot-symmetric, so either agent's action space works.
    action_space = env.action_space(_AGENTS[0])
    policy = load_frozen_policy(policy_path, action_space)
    opponent = load_frozen_policy(opponent_path, action_space)

    slots = [_AGENTS[0], _AGENTS[1]] if swap_roles else [_AGENTS[0]]

    wins = losses = draws = timeouts = 0
    policy_returns: list[float] = []
    opponent_returns: list[float] = []
    episode = 0
    try:
        # Pair the two spawn orientations on identical seeds so role-swapping
        # cancels positional bias exactly: episode i is replayed from the same
        # initial conditions with the policy in each slot, rather than each
        # orientation drawing fresh (unmatched) seeds.
        for i in range(n_episodes):
            for policy_slot in slots:
                result, p_ret, o_ret = _play_episode(
                    env, policy, opponent, policy_slot, base_seed + i
                )
                episode += 1
                policy_returns.append(p_ret)
                opponent_returns.append(o_ret)
                if result == "win":
                    wins += 1
                elif result == "loss":
                    losses += 1
                elif result == "draw":
                    draws += 1
                else:
                    timeouts += 1
    finally:
        env.close()

    total = max(episode, 1)
    result = {
        "policy_win_rate": wins / total,
        "opponent_win_rate": losses / total,
        "draw_rate": draws / total,
        "timeout_rate": timeouts / total,
        "policy_mean_return": float(np.mean(policy_returns)) if policy_returns else 0.0,
        "opponent_mean_return": (
            float(np.mean(opponent_returns)) if opponent_returns else 0.0
        ),
        "n_episodes": episode,
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
