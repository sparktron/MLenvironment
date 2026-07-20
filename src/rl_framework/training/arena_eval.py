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


def run_n_agent_eval(
    policy_paths: list[str],
    cfg: dict,
    n_episodes: int = 100,
    output_path: str | None = None,
) -> dict:
    """Evaluate one policy per arena slot and report N-agent outcomes.

    ``policy_paths`` is ordered by ``env.possible_agents``. Each episode is
    seeded identically for all slot assignments; callers that need role
    balancing should rotate the paths before invoking this primitive.
    """
    env_cfg = cfg["environment"]
    env = make_env(env_cfg["type"], env_cfg)
    agents = list(env.possible_agents)
    if len(policy_paths) != len(agents):
        env.close()
        raise ValueError(f"Expected {len(agents)} policy paths, got {len(policy_paths)}")
    policies = [load_frozen_policy(path, env.action_space(agents[0])) for path in policy_paths]
    wins = dict.fromkeys(agents, 0)
    returns = {agent: [] for agent in agents}
    placements = {agent: [] for agent in agents}
    draws = timeouts = 0
    try:
        for episode in range(n_episodes):
            observations, _ = env.reset(seed=int(cfg.get("seed", 0)) + episode)
            totals = dict.fromkeys(agents, 0.0)
            outcome: dict[str, Any] | None = None
            while env.agents:
                actions = {}
                for agent, obs in observations.items():
                    action, _ = policies[agents.index(agent)].predict(obs, deterministic=True)
                    actions[agent] = np.asarray(action, dtype=np.float32)
                observations, rewards, _, _, infos = env.step(actions)
                for agent, reward in rewards.items():
                    totals[agent] += float(reward)
                outcome = next((info["episode_outcome"] for info in infos.values() if "episode_outcome" in info), outcome)
            for agent in agents:
                returns[agent].append(totals[agent])
            if not outcome or outcome.get("outcome") == "timeout":
                timeouts += 1
                for agent in agents:
                    placements[agent].append(1.0)
            elif outcome.get("outcome") == "draw":
                draws += 1
                for agent in agents:
                    placements[agent].append(1.0)
            else:
                winner = outcome.get("winner")
                for agent in agents:
                    won = agent == winner
                    wins[agent] += int(won)
                    placements[agent].append(1.0 if won else 0.0)
    finally:
        env.close()
    result = {
        "agents": agents,
        "n_episodes": n_episodes,
        "draw_rate": draws / max(n_episodes, 1),
        "timeout_rate": timeouts / max(n_episodes, 1),
        "agent_win_rates": {agent: wins[agent] / max(n_episodes, 1) for agent in agents},
        "agent_mean_returns": {agent: float(np.mean(values)) for agent, values in returns.items()},
        "agent_mean_scores": {agent: float(np.mean(values)) for agent, values in placements.items()},
    }
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _play_episode(
    env: Any,
    policy: Any,
    opponent: Any,
    policy_slot: str,
    seed: int,
    *,
    include_metrics: bool = False,
):
    """Run one episode; return (outcome_for_policy, policy_return, opponent_return).

    ``outcome_for_policy`` is ``"win"``, ``"loss"``, or ``"timeout"`` from the
    policy's perspective.
    """
    opp_slot = _AGENTS[1] if policy_slot == _AGENTS[0] else _AGENTS[0]
    observations, _ = env.reset(seed=seed)
    returns = {a: 0.0 for a in _AGENTS}
    final_outcome: dict[str, Any] | None = None
    final_metrics: dict[str, dict[str, float]] = {}

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
        for agent, info in infos.items():
            if "episode_metrics" in info:
                final_metrics[agent] = info["episode_metrics"]

    if final_outcome is None or final_outcome.get("outcome") == "timeout":
        result = "timeout"
    elif final_outcome.get("outcome") == "draw":
        # Simultaneous knockout — neither side wins.
        result = "draw"
    elif final_outcome.get("winner") == policy_slot:
        result = "win"
    else:
        result = "loss"
    basic = (result, returns[policy_slot], returns[opp_slot])
    if not include_metrics:
        return basic
    return (*basic, final_metrics.get(policy_slot, {}), final_metrics.get(opp_slot, {}))


def _mean_episode_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(np.mean([float(row.get(key, 0.0)) for row in rows]))
        for key in keys
    }


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
    if list(env.possible_agents) != list(_AGENTS):
        env.close()
        raise ValueError(
            "run_arena_eval is a head-to-head harness and requires exactly 2 "
            f"agents ({_AGENTS}); the env has {list(env.possible_agents)}. "
            "Set environment.num_agents: 2 for head-to-head eval."
        )
    base_seed = int(cfg.get("seed", 0))

    # Spaces are slot-symmetric, so either agent's action space works.
    action_space = env.action_space(_AGENTS[0])
    policy = load_frozen_policy(policy_path, action_space)
    opponent = load_frozen_policy(opponent_path, action_space)

    slots = [_AGENTS[0], _AGENTS[1]] if swap_roles else [_AGENTS[0]]

    wins = losses = draws = timeouts = 0
    policy_returns: list[float] = []
    opponent_returns: list[float] = []
    policy_metrics: list[dict[str, float]] = []
    opponent_metrics: list[dict[str, float]] = []
    episode = 0
    try:
        # Pair the two spawn orientations on identical seeds so role-swapping
        # cancels positional bias exactly: episode i is replayed from the same
        # initial conditions with the policy in each slot, rather than each
        # orientation drawing fresh (unmatched) seeds.
        for i in range(n_episodes):
            for policy_slot in slots:
                result, p_ret, o_ret, p_metrics, o_metrics = _play_episode(
                    env,
                    policy,
                    opponent,
                    policy_slot,
                    base_seed + i,
                    include_metrics=True,
                )
                episode += 1
                policy_returns.append(p_ret)
                opponent_returns.append(o_ret)
                policy_metrics.append(p_metrics)
                opponent_metrics.append(o_metrics)
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
        "policy_episode_metrics": _mean_episode_metrics(policy_metrics),
        "opponent_episode_metrics": _mean_episode_metrics(opponent_metrics),
        "n_episodes": episode,
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
