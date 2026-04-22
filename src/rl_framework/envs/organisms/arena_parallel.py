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
    metadata = {"name": "organism_arena_v0", "render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, cfg: dict[str, Any], render_mode: str | None = None):
        self.cfg = cfg
        self.render_mode = render_mode
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = []
        self._rng = np.random.default_rng(cfg.get("seed", 0))
        self.bounds = float(cfg.get("sim", {}).get("arena_half_extent", 1.0))
        self.rules = BattleRules(**{k: v for k, v in cfg.get("battle_rules", {}).items() if k in BattleRules.__annotations__})
        self.morphology = cfg.get("morphology", {})
        self.state: dict[str, dict[str, Any]] = {}
        self.step_count = 0
        self._fig = None
        self._ax = None

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

    def render(self):
        import matplotlib

        if self.render_mode not in ("human", "rgb_array"):
            return None

        if self.render_mode == "human":
            matplotlib.use("TkAgg")

        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(5, 5))
            self._fig.tight_layout()

        ax = self._ax
        ax.clear()
        ax.set_xlim(-self.bounds, self.bounds)
        ax.set_ylim(-self.bounds, self.bounds)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")
        self._fig.patch.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        # Arena border
        border = mpatches.FancyBboxPatch(
            (-self.bounds, -self.bounds), 2 * self.bounds, 2 * self.bounds,
            boxstyle="square,pad=0", linewidth=2, edgecolor="#888", fill=False,
        )
        ax.add_patch(border)

        _colors = {"agent_0": ("#4fc3f7", "#0288d1"), "agent_1": ("#ef9a9a", "#c62828")}
        _labels = {"agent_0": "A0", "agent_1": "A1"}

        for agent in self.possible_agents:
            if agent not in self.state:
                continue
            s = self.state[agent]
            pos = s["pos"]
            size = s["size"]
            health = s["health"]
            max_health = float(self.morphology.get("health", 1.0)) * size
            frac = np.clip(health / max_health, 0.0, 1.0) if max_health > 0 else 0.0
            fill_color, edge_color = _colors[agent]

            # Agent body circle — radius scales with size, normalised to arena
            radius = 0.07 * size
            circle = plt.Circle(pos, radius, color=fill_color, zorder=3)
            ax.add_patch(circle)
            circle_edge = plt.Circle(pos, radius, fill=False, edgecolor=edge_color, linewidth=2, zorder=4)
            ax.add_patch(circle_edge)

            # Attack range indicator (faint ring) when off cooldown
            if s["cooldown"] == 0:
                atk_ring = plt.Circle(pos, self.rules.attack_range, fill=False,
                                      edgecolor=edge_color, linewidth=0.8, linestyle="--",
                                      alpha=0.4, zorder=2)
                ax.add_patch(atk_ring)

            # Health bar above agent
            bar_w = 0.18
            bar_h = 0.025
            bar_x = pos[0] - bar_w / 2
            bar_y = pos[1] + radius + 0.03
            # Background
            ax.add_patch(mpatches.Rectangle((bar_x, bar_y), bar_w, bar_h,
                                            color="#333", zorder=5))
            # Fill
            ax.add_patch(mpatches.Rectangle((bar_x, bar_y), bar_w * frac, bar_h,
                                            color=fill_color, zorder=6))

            # Cooldown pip
            if s["cooldown"] > 0:
                ax.text(pos[0], pos[1] - radius - 0.06, f"cd:{s['cooldown']}",
                        color=edge_color, fontsize=6, ha="center", va="top", zorder=7)

            ax.text(pos[0], pos[1], _labels[agent],
                    color="white", fontsize=7, ha="center", va="center",
                    fontweight="bold", zorder=8)

        ax.set_title(f"Organism Arena  step={self.step_count}", color="white", fontsize=9, pad=4)

        if self.render_mode == "human":
            plt.pause(1.0 / self.metadata["render_fps"])
            return None

        # rgb_array
        self._fig.canvas.draw()
        buf = self._fig.canvas.buffer_rgba()
        img = np.asarray(buf, dtype=np.uint8).reshape(
            self._fig.canvas.get_width_height()[::-1] + (4,)
        )
        return img[:, :, :3]

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None
