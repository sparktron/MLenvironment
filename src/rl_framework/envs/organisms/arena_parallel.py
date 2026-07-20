from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from rl_framework.utils.config_merge import get_section


@dataclass
class BattleRules:
    damage: float = 0.05
    collision_damage: float = 0.0
    attack_range: float = 0.2
    cooldown_steps: int = 3
    sensing_radius: float = 2.0
    max_steps: int = 400
    win_health_threshold: float = 0.0
    attack_falloff: str = (
        "linear"  # "linear" (distance-graded) or "binary" (hard cliff)
    )


@dataclass
class ResourceRules:
    initial_energy: float = 1.0
    max_energy: float = 1.0
    movement_cost: float = 0.01
    attack_cost: float = 0.04
    food_count: int = 2
    food_energy: float = 0.35
    food_radius: float = 0.10
    food_respawn_steps: int = 40
    food_placement: str = "uniform"


class OrganismArenaParallelEnv(ParallelEnv):
    metadata = {
        "name": "organism_arena_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, cfg: dict[str, Any], render_mode: str | None = None):
        self.cfg = cfg
        self.render_mode = render_mode
        self.num_agents_cfg = int(cfg.get("num_agents", 2))
        if self.num_agents_cfg < 2:
            raise ValueError(f"num_agents must be >= 2, got {self.num_agents_cfg}")
        self.possible_agents = [f"agent_{i}" for i in range(self.num_agents_cfg)]
        self.agents: list[str] = []
        # Agents that have been knocked out this episode. They linger in the
        # agent set as inert spectators (so the population — and the SuperSuit
        # vec-env — stays a constant size) until the episode ends for everyone.
        self._dead: set[str] = set()
        self._rng = np.random.default_rng(cfg.get("seed", 0))
        # get_section tolerates both a missing key and an explicit `key: null`
        # (the GUI wizard has historically written the latter for empty
        # nested groups) — a plain `.get(key, {})` only covers the former.
        sim_cfg = get_section(cfg, "sim")
        self.bounds = float(sim_cfg.get("arena_half_extent", 1.0))
        # Per-step movement speed cap (arena units / step).
        self.move_speed = float(sim_cfg.get("move_speed", 0.05))
        self.collision_radius = float(sim_cfg.get("collision_radius", 0.08))
        self.speed_size_exponent = float(sim_cfg.get("speed_size_exponent", 1.0))
        # Spawn-position jitter half-width. Non-zero by default: without it the
        # env is fully deterministic and head-to-head eval replays one episode.
        self.spawn_jitter = float(sim_cfg.get("spawn_jitter", 0.1))
        rules_cfg = get_section(cfg, "battle_rules")
        unknown_rules = sorted(set(rules_cfg) - set(BattleRules.__annotations__))
        if unknown_rules:
            warnings.warn(
                f"Ignoring unknown battle_rules keys {unknown_rules}; "
                f"valid keys are {sorted(BattleRules.__annotations__)}",
                stacklevel=2,
            )
        self.rules = BattleRules(
            **{k: v for k, v in rules_cfg.items() if k in BattleRules.__annotations__}
        )
        resource_cfg = get_section(cfg, "resources")
        unknown_resources = sorted(set(resource_cfg) - set(ResourceRules.__annotations__))
        if unknown_resources:
            warnings.warn(
                f"Ignoring unknown resources keys {unknown_resources}; "
                f"valid keys are {sorted(ResourceRules.__annotations__)}",
                stacklevel=2,
            )
        self.resources = ResourceRules(
            **{k: v for k, v in resource_cfg.items() if k in ResourceRules.__annotations__}
        )
        self.morphology = get_section(cfg, "morphology")
        self.state: dict[str, dict[str, Any]] = {}
        # Previous-step positions, used to derive each agent's velocity for the
        # egocentric observation. Repopulated on reset and updated each step.
        self._prev_positions: dict[str, np.ndarray] = {}
        self._food: list[dict[str, Any]] = []
        self._episode_stats: dict[str, dict[str, float]] = {}
        # Scales only the dense per-hit reward (not the health damage itself).
        # Annealed toward 0 by RewardAnnealingCallback so the terminal win/loss
        # signal eventually dominates. See update_live_params().
        self._damage_scale = 1.0
        self.step_count = 0
        self._fig = None
        self._ax = None

    def observation_space(self, agent: str):
        # Egocentric, slot-symmetric obs (13D), every component pre-scaled to
        # roughly [-1, 1]:
        #   [self_vel_x, self_vel_y, health_frac,
        #    energy_frac, size_frac, rel_opp_x, rel_opp_y, opp_health_frac,
        #    cooldown_frac, opp_visible, rel_food_x, rel_food_y, food_visible]
        # Self position is expressed as velocity (displacement since last step,
        # in units of move_speed) rather than absolute coords so the shared
        # policy sees the same input distribution regardless of spawn slot.
        # Health is a fraction of max health; rel components are in units of
        # the arena diameter and gated (with opp_health) by sensing_radius.
        # opp_visible disambiguates "out of sensing range" (all-zero opponent
        # block) from "adjacent opponent with near-zero health".
        return spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

    def action_space(self, agent: str):
        # move_x, move_y, attack_trigger
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _max_health_for_size(self, size: float) -> float:
        """Health pool for an organism of the given *size*.

        Used at spawn and re-applied each step so episode growth scales an
        organism's max health in lockstep with the damage it deals (both scale
        with ``size``); otherwise growth would be a pure offensive buff.
        """
        return float(self.morphology.get("health", 1.0)) * float(size)

    def _spawn_position(self, index: int) -> np.ndarray:
        """Evenly space agent *index* on a spawn circle of radius 0.6.

        ``theta = pi - 2*pi*index/N`` so the 2-agent case reduces exactly to the
        legacy layout (agent_0 at ``(-0.6, 0)``, agent_1 at ``(+0.6, 0)``) while
        N>2 spreads competitors symmetrically around the arena.
        """
        n = self.num_agents_cfg
        theta = np.pi - 2.0 * np.pi * index / n
        return 0.6 * np.array([np.cos(theta), np.sin(theta)])

    def _spawn_agent(self, name: str, index: int) -> dict[str, Any]:
        base_size = float(self.morphology.get("base_size", 1.0))
        # step_count is 0 at spawn — store base_size and compute current size dynamically.
        size = float(np.clip(base_size, 0.5, 2.0))
        health = self._max_health_for_size(size)
        jitter = self._rng.uniform(-self.spawn_jitter, self.spawn_jitter, size=2)
        pos = np.clip(
            self._spawn_position(index) + jitter, -self.bounds, self.bounds
        ).astype(np.float32)
        return {
            "pos": pos,
            "health": health,
            "max_health": health,
            "cooldown": 0,
            "size": size,
            "energy": self.resources.initial_energy,
        }

    def _current_size(self, agent: str) -> float:
        """Compute agent's current size including episode growth."""
        base_size = float(self.morphology.get("base_size", 1.0))
        growth = (
            float(self.morphology.get("episode_growth_scale", 0.0)) * self.step_count
        )
        return float(np.clip(base_size + growth, 0.5, 2.0))

    def _spawn_food_position(self) -> np.ndarray:
        margin = max(self.resources.food_radius, 0.05)
        extent = max(self.bounds - margin, 0.0)
        if self.resources.food_placement == "center":
            # A contested central patch makes food a strategic objective while
            # preserving symmetric access from every spawn slot.
            extent *= 0.3
        return self._rng.uniform(-extent, extent, size=2).astype(np.float32)

    def _reset_food(self) -> None:
        self._food = [
            {"pos": self._spawn_food_position(), "respawn_at": None}
            for _ in range(self.resources.food_count)
        ]

    def _effective_move_speed(self, agent: str) -> float:
        size = max(float(self.state[agent]["size"]), 1e-8)
        return self.move_speed / size**self.speed_size_exponent

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self._dead = set()
        self.step_count = 0
        self.state = {
            name: self._spawn_agent(name, i)
            for i, name in enumerate(self.possible_agents)
        }
        self._reset_food()
        self._episode_stats = {
            agent: {
                "attack_hits": 0.0,
                "collision_contacts": 0.0,
                "damage_dealt": 0.0,
                "energy_depleted_steps": 0.0,
                "food_pickups": 0.0,
            }
            for agent in self.agents
        }
        # Seed previous positions so velocity reads zero on the first observation.
        self._prev_positions = {
            agent: self.state[agent]["pos"].copy() for agent in self.agents
        }
        observations = {agent: self._obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _is_alive(self, agent: str) -> bool:
        return self.state[agent]["health"] > self.rules.win_health_threshold

    def is_alive(self, agent: str) -> bool:
        """Return whether *agent* can still act in the current episode."""
        return self._is_alive(agent)

    def _nearest_opponent(
        self, agent: str, candidates: set[str] | None = None
    ) -> tuple[str | None, float]:
        """Return the nearest opponent of *agent* and the distance to it.

        Generalises the 2-agent "the other one" lookup to N agents: the policy
        always perceives (and, in ``step``, attacks) its closest threat. For N=2
        this is exactly the single opponent, so the observation is unchanged.

        By default candidates are the currently-living opponents (used for
        observations). ``step`` passes an explicit *candidates* set — the agents
        alive at the start of the attack phase — so all attacks in a step
        resolve simultaneously rather than letting an early kill cancel
        retaliation. Returns ``(None, inf)`` when no candidate exists.
        """
        me_pos = self.state[agent]["pos"]
        pool = (
            candidates
            if candidates is not None
            else {o for o in self.possible_agents if self._is_alive(o)}
        )
        nearest: str | None = None
        best = float("inf")
        for other in pool:
            if other == agent:
                continue
            dist = float(np.linalg.norm(me_pos - self.state[other]["pos"]))
            if dist < best:
                best, nearest = dist, other
        return nearest, best

    def _obs(self, agent: str) -> np.ndarray:
        """Build the egocentric observation for *agent* (pure read — no state mutation).

        All components are pre-scaled to roughly ``[-1, 1]`` (see
        ``observation_space``). Self position is reported as velocity
        (displacement since the previous step, in units of ``move_speed``) so
        every agent shares one input distribution. The opponent block describes
        the *nearest living opponent* (relative position in arena-diameter
        units, health fraction), zeroed and flagged invisible when none is
        within ``sensing_radius``.
        """
        me = self.state[agent]
        prev = self._prev_positions.get(agent, me["pos"])
        vel = (me["pos"] - prev) / max(self.move_speed, 1e-8)
        diameter = max(2.0 * self.bounds, 1e-8)
        nearest, dist = self._nearest_opponent(agent)
        visible = nearest is not None and dist <= self.rules.sensing_radius
        if visible:
            other = self.state[nearest]
            rel = other["pos"] - me["pos"]
            rel_x = float(rel[0]) / diameter
            rel_y = float(rel[1]) / diameter
            opp_health = other["health"] / max(other["max_health"], 1e-8)
        else:
            # No living opponent in sensing range — hide the opponent block.
            rel_x, rel_y, opp_health = 0.0, 0.0, 0.0
        health_frac = me["health"] / max(me["max_health"], 1e-8)
        cooldown_frac = float(me["cooldown"]) / max(self.rules.cooldown_steps, 1)
        available_food = [food for food in self._food if food["respawn_at"] is None]
        if available_food:
            food = min(available_food, key=lambda item: float(np.linalg.norm(item["pos"] - me["pos"])))
            food_rel = (food["pos"] - me["pos"]) / diameter
            food_x, food_y, food_visible = float(food_rel[0]), float(food_rel[1]), 1.0
        else:
            food_x = food_y = food_visible = 0.0
        return np.array(
            [
                vel[0],
                vel[1],
                health_frac,
                me["energy"] / max(self.resources.max_energy, 1e-8),
                me["size"] / 2.0,
                rel_x,
                rel_y,
                opp_health,
                cooldown_frac,
                float(visible),
                food_x,
                food_y,
                food_visible,
            ],
            dtype=np.float32,
        )

    def observe(self, agent: str) -> np.ndarray:
        """Public read-only observation accessor for *agent*.

        Wrappers that drive an internal agent slot (e.g. ``SelfPlayEnvWrapper``)
        should use this rather than the private ``_obs``.
        """
        return self._obs(agent)

    def _attack_falloff(self, dist: float) -> float:
        """Return the damage multiplier in ``[0, 1]`` for an attack at *dist*.

        ``"linear"`` (default) grades damage with distance: full damage at
        ``dist == 0``, half at ``attack_range / 2``, zero at ``dist >=
        attack_range``. This gives the policy a continuous gradient that
        rewards closing distance rather than the hard hit/miss cliff of the
        ``"binary"`` mode, which deals full damage inside ``attack_range`` and
        nothing beyond it.
        """
        rng = self.rules.attack_range
        if self.rules.attack_falloff == "binary":
            return 1.0 if dist <= rng else 0.0
        if rng <= 0.0:
            return 1.0 if dist <= 0.0 else 0.0
        return max(0.0, 1.0 - dist / rng)

    def update_live_params(self, params: dict[str, Any]) -> None:
        """Apply live parameter overrides from training callbacks.

        Supported keys:
          * ``reward.damage_scale`` — float multiplier on the dense per-hit
            reward (used by ``RewardAnnealingCallback``).
          * ``battle_rules.<field>`` — overrides a ``BattleRules`` field in place
            (used by ``CurriculumCallback`` to ramp opponent difficulty); the
            value is coerced to the field's existing type.
          * ``morphology.<field>`` — updates morphology config for current/future
            size calculations and subsequent resets.
          * ``sim.arena_half_extent`` — updates arena bounds immediately.

        Unknown keys are ignored so a shared curriculum config can target
        multiple env families without error.
        """
        for key, value in params.items():
            if key == "reward.damage_scale":
                self._damage_scale = float(value)
                # setdefault leaves an explicit `reward: null` untouched (it
                # only fills in a genuinely absent key); get_section replaces
                # it with {} first so the write below cannot crash.
                get_section(self.cfg, "reward")["damage_scale"] = self._damage_scale
            elif key.startswith("battle_rules."):
                field = key.removeprefix("battle_rules.")
                if hasattr(self.rules, field):
                    current = getattr(self.rules, field)
                    cast_val = type(current)(value)
                    setattr(self.rules, field, cast_val)
                    get_section(self.cfg, "battle_rules")[field] = cast_val
            elif key.startswith("morphology."):
                field = key.removeprefix("morphology.")
                current = self.morphology.get(field)
                try:
                    cast_val = type(current)(value) if current is not None else value
                except (TypeError, ValueError):
                    continue
                self.morphology[field] = cast_val
                get_section(self.cfg, "morphology")[field] = cast_val
            elif key.startswith("resources."):
                field = key.removeprefix("resources.")
                if hasattr(self.resources, field):
                    current = getattr(self.resources, field)
                    cast_val = type(current)(value)
                    setattr(self.resources, field, cast_val)
                    get_section(self.cfg, "resources")[field] = cast_val
            elif key == "sim.arena_half_extent":
                self.bounds = float(value)
                get_section(self.cfg, "sim")["arena_half_extent"] = self.bounds

    def step(self, actions: dict[str, np.ndarray]):
        if not self.agents:
            # Episode already over. Without this guard the empty truncations
            # dict below makes all(...) vacuously True and the step fabricates
            # a spurious timeout outcome.
            return {}, {}, {}, {}, {}
        self.step_count += 1
        # Capture the active agent set at step entry; all returned dicts must share these keys.
        active_agents = list(self.agents)
        rewards = {agent: 0.0 for agent in active_agents}
        terminations = {agent: False for agent in active_agents}
        truncations = {
            agent: self.step_count >= self.rules.max_steps for agent in active_agents
        }

        # Movement, cooldown decay, and growth — living agents only; knocked-out
        # spectators are inert until the episode ends for everyone.
        for agent, action in actions.items():
            if agent not in self.state or not self._is_alive(agent):
                continue
            move = np.asarray(action[:2], dtype=np.float32)
            # Clamp the move *norm* (not per component) so diagonal movement
            # is not √2 faster than axis-aligned movement.
            norm = float(np.linalg.norm(move))
            if norm > 1.0:
                move = move / norm
            move_distance = float(np.linalg.norm(move))
            energy = self.state[agent]["energy"]
            if energy <= 0.0:
                move = np.zeros(2, dtype=np.float32)
                move_distance = 0.0
                self._episode_stats[agent]["energy_depleted_steps"] += 1.0
            else:
                move = move * self._effective_move_speed(agent)
                self.state[agent]["energy"] = max(
                    0.0, energy - move_distance * self.resources.movement_cost
                )
            self.state[agent]["pos"] = np.clip(
                self.state[agent]["pos"] + move, -self.bounds, self.bounds
            )
            self.state[agent]["cooldown"] = max(
                0, int(self.state[agent]["cooldown"]) - 1
            )
            # Apply growth: update size each step so episode_growth_scale takes
            # effect. Max health tracks size, and current health is rescaled by
            # the same factor so the health *fraction* is preserved — growth
            # raises both the cap and current health, making a larger organism
            # tankier as well as harder-hitting (its damage already scales with
            # size). Without this, growth would buff offense only.
            new_size = self._current_size(agent)
            prev_max = self.state[agent]["max_health"]
            new_max = self._max_health_for_size(new_size)
            if prev_max > 0 and new_max != prev_max:
                self.state[agent]["health"] *= new_max / prev_max
            self.state[agent]["max_health"] = new_max
            self.state[agent]["size"] = new_size

        self._resolve_collisions(active_agents)
        self._update_food(active_agents, rewards)

        # Attacks — each living attacker strikes its nearest living opponent
        # within attack_range (single target, mirroring the nearest-opponent
        # observation). For N=2 this is exactly the other agent. Targeting and
        # attacker eligibility use a snapshot of the agents alive at the start
        # of the attack phase, so all attacks in a step resolve simultaneously
        # (an early kill does not cancel the victim's retaliation -> mutual KOs
        # are draws, not wins).
        combatants = {agent for agent in active_agents if self._is_alive(agent)}
        for attacker in active_agents:
            if attacker not in combatants or attacker not in actions:
                continue
            trigger = float(actions[attacker][2]) > 0.5
            if (
                not trigger
                or self.state[attacker]["cooldown"] > 0
                or self.state[attacker]["energy"] < self.resources.attack_cost
            ):
                continue
            target, dist = self._nearest_opponent(attacker, candidates=combatants)
            if target is None or dist > self.rules.attack_range:
                continue
            falloff = self._attack_falloff(dist)
            if falloff > 0.0:
                self.state[attacker]["energy"] -= self.resources.attack_cost
                damage = self.rules.damage * falloff * self.state[attacker]["size"]
                damage = min(damage, self.state[target]["health"])
                self.state[target]["health"] = max(
                    0.0, self.state[target]["health"] - damage
                )
                self._episode_stats[attacker]["attack_hits"] += 1.0
                self._episode_stats[attacker]["damage_dealt"] += damage
                # Health always takes full damage so combat resolves; only the
                # dense reward is scaled (annealed toward the sparse win signal).
                dense_reward = damage * self._damage_scale
                rewards[attacker] += dense_reward
                rewards[target] -= dense_reward
                self.state[attacker]["cooldown"] = self.rules.cooldown_steps

        # Newly knocked-out agents take a one-time -1 and become inert.
        for agent in active_agents:
            if agent not in self._dead and not self._is_alive(agent):
                self._dead.add(agent)
                rewards[agent] -= 1.0

        living = [agent for agent in active_agents if self._is_alive(agent)]
        timeout = self.step_count >= self.rules.max_steps
        ended_by_elimination = len(living) <= 1

        # Build a per-episode outcome annotation for instrumentation. The same
        # dict is attached to every agent's info so a metrics callback can read
        # the result from whichever agent slot it observes.
        episode_outcome: dict[str, Any] | None = None
        if ended_by_elimination:
            # Last-organism-standing: elimination *terminates* (not truncates).
            truncations = {agent: False for agent in active_agents}
            for agent in active_agents:
                terminations[agent] = True
            if len(living) == 1:
                survivor = living[0]
                rewards[survivor] += 1.0
                episode_outcome = {
                    "winner": survivor,
                    "outcome": "ko",
                    "step": self.step_count,
                }
            else:
                # Zero survivors — simultaneous wipeout, no winner.
                episode_outcome = {
                    "winner": None,
                    "outcome": "draw",
                    "step": self.step_count,
                }
            self.agents = []
        elif timeout:
            episode_outcome = {
                "winner": None,
                "outcome": "timeout",
                "step": self.step_count,
            }
            self.agents = []

        # All five dicts must have identical keys (active_agents) per PettingZoo Parallel API.
        observations = {agent: self._obs(agent) for agent in active_agents}
        infos = {agent: {"step": self.step_count} for agent in active_agents}
        if episode_outcome is not None:
            for agent in active_agents:
                infos[agent]["episode_outcome"] = episode_outcome
                infos[agent]["episode_metrics"] = dict(self._episode_stats[agent])
        # Record positions *after* building observations so the next step's
        # velocity reflects the displacement that occurs during that step.
        for agent in active_agents:
            self._prev_positions[agent] = self.state[agent]["pos"].copy()
        return observations, rewards, terminations, truncations, infos

    def _resolve_collisions(self, active_agents: list[str]) -> None:
        """Separate overlapping living organisms after movement."""
        for index, left in enumerate(active_agents):
            if not self._is_alive(left):
                continue
            for right in active_agents[index + 1 :]:
                if not self._is_alive(right):
                    continue
                delta = self.state[right]["pos"] - self.state[left]["pos"]
                distance = float(np.linalg.norm(delta))
                minimum = self.collision_radius * (self.state[left]["size"] + self.state[right]["size"])
                if distance >= minimum:
                    continue
                self._episode_stats[left]["collision_contacts"] += 1.0
                self._episode_stats[right]["collision_contacts"] += 1.0
                if self.rules.collision_damage > 0.0:
                    damage_to_left = min(
                        self.rules.collision_damage * self.state[right]["size"],
                        self.state[left]["health"],
                    )
                    damage_to_right = min(
                        self.rules.collision_damage * self.state[left]["size"],
                        self.state[right]["health"],
                    )
                    self.state[left]["health"] -= damage_to_left
                    self.state[right]["health"] -= damage_to_right
                    self._episode_stats[right]["damage_dealt"] += damage_to_left
                    self._episode_stats[left]["damage_dealt"] += damage_to_right
                direction = delta / distance if distance > 1e-8 else np.array([1.0, 0.0], dtype=np.float32)
                correction = (minimum - distance) / 2.0
                self.state[left]["pos"] = np.clip(self.state[left]["pos"] - direction * correction, -self.bounds, self.bounds)
                self.state[right]["pos"] = np.clip(self.state[right]["pos"] + direction * correction, -self.bounds, self.bounds)

    def _update_food(self, active_agents: list[str], rewards: dict[str, float]) -> None:
        """Respawn consumed food and award its energy to the first nearby agent."""
        for food in self._food:
            if food["respawn_at"] is not None:
                if self.step_count >= food["respawn_at"]:
                    food["pos"] = self._spawn_food_position()
                    food["respawn_at"] = None
                continue
            nearby = [agent for agent in active_agents if self._is_alive(agent) and float(np.linalg.norm(self.state[agent]["pos"] - food["pos"])) <= self.resources.food_radius]
            if nearby:
                eater = min(nearby)
                before = self.state[eater]["energy"]
                self.state[eater]["energy"] = min(self.resources.max_energy, before + self.resources.food_energy)
                rewards[eater] += self.state[eater]["energy"] - before
                self._episode_stats[eater]["food_pickups"] += 1.0
                food["respawn_at"] = self.step_count + self.resources.food_respawn_steps

    def render(self):
        import matplotlib

        if self.render_mode not in ("human", "rgb_array"):
            return None

        # Switch backend once, before the first figure exists — calling use()
        # repeatedly after pyplot is live is at best a no-op and at worst
        # clashes with an already-loaded backend. rgb_array never forces a
        # backend so headless (Agg) rendering keeps working.
        if self.render_mode == "human" and self._fig is None:
            try:
                matplotlib.use("TkAgg")
            except ImportError as exc:
                raise RuntimeError(
                    "render_mode='human' needs a TkAgg-capable display; "
                    "use render_mode='rgb_array' for headless rendering"
                ) from exc

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
            (-self.bounds, -self.bounds),
            2 * self.bounds,
            2 * self.bounds,
            boxstyle="square,pad=0",
            linewidth=2,
            edgecolor="#888",
            fill=False,
        )
        ax.add_patch(border)

        # Per-agent fill/edge colours. The first two keep the legacy palette
        # (blue A0, red A1); further agents are sampled from a qualitative
        # colormap so N>2 arenas stay legible.
        import matplotlib.colors as mcolors
        from matplotlib import colormaps

        base_palette = [("#4fc3f7", "#0288d1"), ("#ef9a9a", "#c62828")]

        def _agent_colors(index: int) -> tuple[str, str]:
            if index < len(base_palette):
                return base_palette[index]
            hexc = mcolors.to_hex(colormaps["tab10"](index % 10))
            return hexc, hexc

        for index, agent in enumerate(self.possible_agents):
            if agent not in self.state:
                continue
            s = self.state[agent]
            pos = s["pos"]
            size = s["size"]
            health = s["health"]
            max_health = s["max_health"]
            frac = np.clip(health / max_health, 0.0, 1.0) if max_health > 0 else 0.0
            fill_color, edge_color = _agent_colors(index)
            label = f"A{index}"

            # Agent body circle — radius scales with size, normalised to arena
            radius = 0.07 * size
            circle = plt.Circle(pos, radius, color=fill_color, zorder=3)
            ax.add_patch(circle)
            circle_edge = plt.Circle(
                pos, radius, fill=False, edgecolor=edge_color, linewidth=2, zorder=4
            )
            ax.add_patch(circle_edge)

            # Attack range indicator (faint ring) when off cooldown
            if s["cooldown"] == 0:
                atk_ring = plt.Circle(
                    pos,
                    self.rules.attack_range,
                    fill=False,
                    edgecolor=edge_color,
                    linewidth=0.8,
                    linestyle="--",
                    alpha=0.4,
                    zorder=2,
                )
                ax.add_patch(atk_ring)

            # Health bar above agent
            bar_w = 0.18
            bar_h = 0.025
            bar_x = pos[0] - bar_w / 2
            bar_y = pos[1] + radius + 0.03
            # Background
            ax.add_patch(
                mpatches.Rectangle((bar_x, bar_y), bar_w, bar_h, color="#333", zorder=5)
            )
            # Fill
            ax.add_patch(
                mpatches.Rectangle(
                    (bar_x, bar_y), bar_w * frac, bar_h, color=fill_color, zorder=6
                )
            )

            # Cooldown pip
            if s["cooldown"] > 0:
                ax.text(
                    pos[0],
                    pos[1] - radius - 0.06,
                    f"cd:{s['cooldown']}",
                    color=edge_color,
                    fontsize=6,
                    ha="center",
                    va="top",
                    zorder=7,
                )

            ax.text(
                pos[0],
                pos[1],
                label,
                color="white",
                fontsize=7,
                ha="center",
                va="center",
                fontweight="bold",
                zorder=8,
            )

        ax.set_title(
            f"Organism Arena  step={self.step_count}", color="white", fontsize=9, pad=4
        )

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
