from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


@dataclass
class BattleRules:
    damage: float = 0.05
    attack_range: float = 0.2
    cooldown_steps: int = 3
    sensing_radius: float = 2.0
    max_steps: int = 400
    win_health_threshold: float = 0.0


class OrganismArenaParallelEnv(ParallelEnv):
    metadata = {"name": "organism_arena_v0"}

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = []
        self._rng = np.random.default_rng(cfg.get("seed", 0))
        self.bounds = float(cfg.get("sim", {}).get("arena_half_extent", 1.0))
        self.rules = BattleRules(**{k: v for k, v in cfg.get("battle_rules", {}).items() if k in BattleRules.__annotations__})
        self.morphology = cfg.get("morphology", {})
        self.state: dict[str, dict[str, Any]] = {}
        self.step_count = 0

    def observation_space(self, agent: str):
        # self x,y,health + opp relative x,y,health + cooldown
        return spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def action_space(self, agent: str):
        # move_x, move_y, attack_trigger
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _spawn_agent(self, name: str, sign: float) -> dict[str, Any]:
        base_size = float(self.morphology.get("base_size", 1.0))
        # step_count is 0 at spawn — store base_size and compute current size dynamically.
        size = np.clip(base_size, 0.5, 2.0)
        health = float(self.morphology.get("health", 1.0)) * size
        return {
            "pos": np.array([0.6 * sign, 0.0], dtype=np.float32),
            "health": health,
            "cooldown": 0,
            "size": size,
        }

    def _current_size(self, agent: str) -> float:
        """Compute agent's current size including episode growth."""
        base_size = float(self.morphology.get("base_size", 1.0))
        growth = float(self.morphology.get("episode_growth_scale", 0.0)) * self.step_count
        return float(np.clip(base_size + growth, 0.5, 2.0))

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.state = {
            "agent_0": self._spawn_agent("agent_0", -1.0),
            "agent_1": self._spawn_agent("agent_1", 1.0),
        }
        observations = {agent: self._obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _obs(self, agent: str) -> np.ndarray:
        opp = "agent_1" if agent == "agent_0" else "agent_0"
        me, other = self.state[agent], self.state[opp]
        rel = other["pos"] - me["pos"]
        return np.array([
            me["pos"][0], me["pos"][1], me["health"],
            rel[0], rel[1], other["health"], float(me["cooldown"])
        ], dtype=np.float32)

    def step(self, actions: dict[str, np.ndarray]):
        self.step_count += 1
        # Capture the active agent set at step entry; all returned dicts must share these keys.
        active_agents = list(self.agents)
        rewards = {agent: 0.0 for agent in active_agents}
        terminations = {agent: False for agent in active_agents}
        truncations = {agent: self.step_count >= self.rules.max_steps for agent in active_agents}

        for agent, action in actions.items():
            if agent not in self.state:
                continue
            move = np.asarray(action[:2], dtype=np.float32) * 0.05
            self.state[agent]["pos"] = np.clip(self.state[agent]["pos"] + move, -self.bounds, self.bounds)
            self.state[agent]["cooldown"] = max(0, int(self.state[agent]["cooldown"]) - 1)
            # Apply growth: update size each step so episode_growth_scale takes effect.
            self.state[agent]["size"] = self._current_size(agent)

        for attacker in list(self.agents):
            defender = "agent_1" if attacker == "agent_0" else "agent_0"
            if attacker not in actions:
                continue
            trigger = float(actions[attacker][2]) > 0.5
            if not trigger or self.state[attacker]["cooldown"] > 0:
                continue
            dist = np.linalg.norm(self.state[attacker]["pos"] - self.state[defender]["pos"])
            if dist <= self.rules.attack_range:
                damage = self.rules.damage * self.state[attacker]["size"]
                self.state[defender]["health"] = max(0.0, self.state[defender]["health"] - damage)
                rewards[attacker] += damage
                rewards[defender] -= damage
                self.state[attacker]["cooldown"] = self.rules.cooldown_steps

        for agent in active_agents:
            if self.state[agent]["health"] <= self.rules.win_health_threshold:
                terminations[agent] = True
                winner = "agent_1" if agent == "agent_0" else "agent_0"
                if winner in rewards:
                    rewards[winner] += 1.0
                rewards[agent] -= 1.0

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        # All five dicts must have identical keys (active_agents) per PettingZoo Parallel API.
        observations = {agent: self._obs(agent) for agent in active_agents}
        infos = {agent: {"step": self.step_count} for agent in active_agents}
        return observations, rewards, terminations, truncations, infos
