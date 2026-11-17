"""Matplotlib rendering for the organism arena.

Kept out of the env so the rules (``arena_parallel``) carry no drawing code and
can be exercised without matplotlib.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rl_framework.games.arena.arena_parallel import OrganismArenaParallelEnv


class ArenaRenderer:
    """Owns the matplotlib figure for one env."""

    def __init__(self) -> None:
        self._fig: Any = None
        self._ax: Any = None

    def render(self, env: OrganismArenaParallelEnv):
        import matplotlib

        if env.render_mode not in ("human", "rgb_array"):
            return None

        # Switch backend once, before the first figure exists — calling use()
        # repeatedly after pyplot is live is at best a no-op and at worst
        # clashes with an already-loaded backend. rgb_array never forces a
        # backend so headless (Agg) rendering keeps working.
        if env.render_mode == "human" and self._fig is None:
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
        ax.set_xlim(-env.bounds, env.bounds)
        ax.set_ylim(-env.bounds, env.bounds)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")
        self._fig.patch.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        # Arena border
        border = mpatches.FancyBboxPatch(
            (-env.bounds, -env.bounds),
            2 * env.bounds,
            2 * env.bounds,
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

        for index, agent in enumerate(env.possible_agents):
            if agent not in env.state:
                continue
            s = env.state[agent]
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
                    env.rules.attack_range,
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
            f"Organism Arena  step={env.step_count}", color="white", fontsize=9, pad=4
        )

        if env.render_mode == "human":
            plt.pause(1.0 / float(env.metadata["render_fps"]))  # type: ignore[arg-type]
            return None

        # rgb_array
        self._fig.canvas.draw()
        buf = self._fig.canvas.buffer_rgba()
        img = np.asarray(buf, dtype=np.uint8).reshape(
            self._fig.canvas.get_width_height()[::-1] + (4,)
        )
        return img[:, :, :3]

    def close(self) -> None:
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None
