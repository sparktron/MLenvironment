"""Walker terrain: spec parsing, per-episode mix sampling, the body pool and
the heightfield profile.

PyBullet cannot free collision or visual shapes, so terrain bodies are
**pooled**: they are parked under the ground plane between episodes and
re-placed, never recreated per reset.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pybullet as p

from rl_framework.core.config.config_merge import get_section

DEFAULT_PUSH_DURATION_S = 0.1
DEFAULT_TERRAIN_LENGTH = 25.0
DEFAULT_TERRAIN_WIDTH = 3.0
# Terrain starts here so reset randomisation never begins in contact.
TERRAIN_START_X = 0.6
TERRAIN_DX = 0.05
TERRAIN_DY = 0.25
# Pooled terrain bodies not used this episode wait here, under the plane.
TERRAIN_PARK_POSITION = (0.0, 0.0, -50.0)


def terrain_rng(seed: int | None) -> np.random.Generator:
    """Terrain layout draws from their own stream.

    Sharing the main env RNG would shift every later draw (mix sampling,
    reset noise, push direction) whenever a layout is generated, so adding
    per-episode layouts would silently change flat and push episodes too.
    """
    return np.random.default_rng(None if seed is None else [int(seed), 7919])


@dataclass(frozen=True)
class PushSpec:
    """Lateral push schedule. Active when ``force > 0`` and ``interval_steps > 0``."""

    interval_steps: int = 0
    force: float = 0.0
    start_step: int = 0
    # PyBullet clears external forces after every physics tick, so a push
    # applied once lasted 1/240 s: 180 N gave 0.75 N*s, about 0.01 m/s on this
    # 65.9 kg body.
    duration_s: float = DEFAULT_PUSH_DURATION_S

    @classmethod
    def from_terrain(cls, spec: dict[str, Any]) -> PushSpec:
        push_cfg = get_section(spec, "push_recovery")
        out = cls(
            interval_steps=int(push_cfg.get("interval_steps", 0)),
            force=float(push_cfg.get("force", 0.0)),
            start_step=int(push_cfg.get("start_step", 0)),
            duration_s=float(push_cfg.get("duration_s", DEFAULT_PUSH_DURATION_S)),
        )
        if out.duration_s <= 0:
            raise ValueError("terrain.push_recovery.duration_s must be positive")
        return out


def parse_terrain_mix(
    terrain_cfg: dict[str, Any],
) -> list[tuple[dict[str, Any], float]] | None:
    """Parse ``terrain.mix`` into (spec, weight) pairs, or None when absent.

    Each entry is a full terrain spec — the same keys the static ``terrain``
    block accepts — plus an optional ``weight`` (default 1.0). Weights are
    relative, not probabilities; they are normalised at sample time.
    """
    raw = terrain_cfg.get("mix")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError("terrain.mix must be a list of terrain specs")
    # An explicit-but-empty mix is a config error, not "no mix". Silently
    # falling back to the default terrain is the failure mode that made the
    # terrain curriculum a no-op: the run looks fine and trains on the
    # wrong distribution.
    if len(raw) == 0:
        raise ValueError("terrain.mix must contain at least one entry")
    parsed: list[tuple[dict[str, Any], float]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"terrain.mix entries must be mappings, got {entry!r}")
        spec = {k: v for k, v in entry.items() if k != "weight"}
        weight = float(entry.get("weight", 1.0))
        if weight < 0:
            raise ValueError("terrain.mix weights must be non-negative")
        spec.setdefault("preset", "flat")
        parsed.append((spec, weight))
    if not parsed:
        raise ValueError("terrain.mix must contain at least one entry")
    if sum(w for _, w in parsed) <= 0:
        raise ValueError("terrain.mix weights must not sum to zero")
    return parsed


def uneven_profile(
    length: float, height: float, wavelength: float, rng: np.random.Generator
) -> np.ndarray:
    """Smooth ground heights in [0, height], one every ``TERRAIN_DX``.

    Control points every half wavelength take uniform heights and are
    joined by cosine interpolation, so the surface undulates without step
    edges. The first and last points are zero, so the terrain ramps up
    from the spawn plane and back down to it, and one interior point is
    exactly ``height``, so every layout spans the full [0, height] range.
    """
    if wavelength <= 0 or length <= 0 or height < 0:
        raise ValueError(
            "terrain.length and terrain.wavelength must be positive and "
            "terrain.height non-negative"
        )
    spacing = wavelength / 2.0
    knots = max(3, int(np.ceil(length / spacing)) + 1)
    values = rng.uniform(0.0, height, size=knots)
    values[0] = values[-1] = 0.0
    values[int(rng.integers(1, knots - 1))] = height
    xs = np.arange(0.0, length + 1e-9, TERRAIN_DX)
    position = xs / spacing
    left = np.clip(np.floor(position).astype(int), 0, knots - 2)
    frac = np.clip(position - left, 0.0, 1.0)
    weight = 0.5 - 0.5 * np.cos(np.pi * frac)
    return values[left] * (1.0 - weight) + values[left + 1] * weight


class TerrainManager:
    """Owns the terrain spec in force, the mix, and the pooled PyBullet bodies.

    ``terrain.mix`` samples a terrain per episode so one policy trains on the
    whole distribution. The 2026-08-06 v11 result showed why staged
    single-terrain training does not work here: terrain types compete rather
    than compose.
    """

    def __init__(
        self,
        connection: int,
        terrain_cfg: dict[str, Any],
        *,
        seed: int | None,
        friction: Callable[[], float],
    ):
        self._connection = connection
        self._friction = friction
        self._layout_rng = terrain_rng(seed)
        self.body_ids: list[int] = []
        self.mix = parse_terrain_mix(terrain_cfg)
        # With no mix the active spec is the static `terrain` block, so
        # behaviour is byte-identical to before.
        self.active: dict[str, Any] = dict(terrain_cfg)
        self.preset = "flat"
        self.push = PushSpec()
        self.apply_spec(self.active)
        self.reset_pools()

    # -- spec ---------------------------------------------------------
    def apply_spec(self, spec: dict[str, Any]) -> None:
        """Point terrain-derived state at ``spec``.

        Covers push-recovery parameters as well as the preset, because
        ``push_recovery`` is a terrain whose behaviour lives entirely in those
        fields — sampling the preset without them would silently produce an
        unpushed flat episode.
        """
        self.active = dict(spec)
        self.preset = str(spec.get("preset", "flat"))
        self.push = PushSpec.from_terrain(spec)

    def set_mix_from(self, terrain_cfg: dict[str, Any]) -> None:
        """Adopt a whole new terrain block (pending live update) and its mix."""
        self.mix = parse_terrain_mix(terrain_cfg)
        self.apply_spec(terrain_cfg)

    def sample(self, rng: np.random.Generator) -> None:
        """Draw the next episode's terrain from the mix using the env RNG."""
        if not self.mix:
            return
        weights = np.array([w for _, w in self.mix], dtype=np.float64)
        index = int(rng.choice(len(self.mix), p=weights / weights.sum()))
        self.apply_spec(self.mix[index][0])

    def reseed(self, seed: int | None) -> None:
        self._layout_rng = terrain_rng(seed)

    def varies_per_episode(self) -> bool:
        return self.preset in {"uneven", "obstacles"} and bool(
            self.active.get("randomize", True)
        )

    # -- bodies -------------------------------------------------------
    def reset_pools(self) -> None:
        # resetSimulation destroys every body and shape.
        self.body_ids = []
        self._heightfields: dict[tuple[int, int, float], tuple[int, int]] = {}
        self._obstacle_shapes: dict[tuple[float, float], int] = {}
        self._obstacle_pools: dict[tuple[float, float], list[int]] = {}

    def rebuild(self) -> None:
        """Swap this episode's terrain in, without rebuilding the world.

        The world is built once per env and later resets only rewind state,
        which saves tens of ms per episode. Unused pool bodies are parked
        below the ground plane.
        """
        cid = self._connection
        for body_id in self.body_ids:
            p.resetBasePositionAndOrientation(
                body_id, TERRAIN_PARK_POSITION, (0, 0, 0, 1), physicsClientId=cid
            )
        self.body_ids = []
        self.build()

    def _layout_source(self) -> np.random.Generator:
        # randomize: false gives one fixed layout per spec, the same in every
        # episode and every env, for controlled comparisons.
        if self.active.get("randomize", True):
            return self._layout_rng
        return np.random.default_rng(0)

    def _new_body(self, collision: int) -> int:
        cid = self._connection
        # No separate visual shape: PyBullet draws the collision shape.
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            basePosition=TERRAIN_PARK_POSITION,
            physicsClientId=cid,
        )
        p.changeDynamics(
            body_id, -1, lateralFriction=float(self._friction()), physicsClientId=cid
        )
        p.changeVisualShape(
            body_id, -1, rgbaColor=[0.25, 0.45, 0.25, 1.0], physicsClientId=cid
        )
        return body_id

    def _place(self, body_id: int, position: tuple[float, float, float]) -> None:
        p.resetBasePositionAndOrientation(
            body_id, position, (0, 0, 0, 1), physicsClientId=self._connection
        )
        self.body_ids.append(body_id)

    def build(self) -> None:
        """Place static terrain along the whole track, starting clear of spawn.

        Every surface sits on or above the ground plane, and the terrain runs
        ``length`` metres (default 25 m), past where an 800-1000 step episode
        at walking speed can reach. Layouts are redrawn every episode from a
        dedicated RNG unless ``randomize: false``.
        """
        if self.preset in ("flat", "push_recovery"):
            return
        spec = self.active
        length = float(spec.get("length", DEFAULT_TERRAIN_LENGTH))
        width = float(spec.get("width", DEFAULT_TERRAIN_WIDTH))
        if width <= 0:
            raise ValueError("terrain.width must be positive")
        rng = self._layout_source()
        if self.preset == "uneven":
            self._build_uneven(spec, length, width, rng)
        elif self.preset == "obstacles":
            self._build_obstacles(spec, length, width, rng)

    def _build_uneven(
        self,
        spec: dict[str, Any],
        length: float,
        width: float,
        rng: np.random.Generator,
    ) -> None:
        cid = self._connection
        height = float(spec.get("height", 0.025))
        wavelength = float(spec.get("wavelength", 1.0))
        profile = uneven_profile(length, height, wavelength, rng)
        columns = max(2, int(round(width / TERRAIN_DY)) + 1)
        # Row-major with x varying fastest along each row.
        data = np.tile(profile, columns).tolist()
        key = (len(profile), columns, height)
        existing = self._heightfields.get(key)
        shape_kwargs = dict(
            meshScale=[TERRAIN_DX, TERRAIN_DY, 1.0],
            heightfieldData=data,
            numHeightfieldRows=len(profile),
            numHeightfieldColumns=columns,
            physicsClientId=cid,
        )
        if existing is None:
            shape = p.createCollisionShape(p.GEOM_HEIGHTFIELD, **shape_kwargs)
            body_id = self._new_body(shape)
            self._heightfields[key] = (shape, body_id)
        else:
            shape, body_id = existing
            p.createCollisionShape(
                p.GEOM_HEIGHTFIELD, replaceHeightfieldIndex=shape, **shape_kwargs
            )
        # Bullet centres a heightfield on its height range, which is always
        # [0, height] here, so local height h lands at world z = h.
        span = (len(profile) - 1) * TERRAIN_DX
        self._place(body_id, (TERRAIN_START_X + 0.5 * span, 0.0, 0.5 * height))

    def _build_obstacles(
        self,
        spec: dict[str, Any],
        length: float,
        width: float,
        rng: np.random.Generator,
    ) -> None:
        cid = self._connection
        obstacle_height = float(spec.get("obstacle_height", 0.10))
        spacing = float(spec.get("obstacle_spacing", 1.2))
        spacing_jitter = float(spec.get("obstacle_spacing_jitter", 0.3))
        height_jitter = float(spec.get("obstacle_height_jitter", 0.0))
        if obstacle_height <= 0:
            raise ValueError("terrain.obstacle_height must be positive")
        if spacing_jitter < 0 or spacing <= 2 * spacing_jitter:
            raise ValueError(
                "terrain.obstacle_spacing must exceed twice a non-negative "
                "obstacle_spacing_jitter"
            )
        if not 0 <= height_jitter < 1:
            raise ValueError("terrain.obstacle_height_jitter must be in [0, 1)")
        # One shape at the tallest possible height; shorter obstacles are the
        # same box sunk into the ground, so jitter needs no new shapes.
        tallest = obstacle_height * (1.0 + height_jitter)
        obstacle_key = (width, tallest)
        if obstacle_key not in self._obstacle_shapes:
            self._obstacle_shapes[obstacle_key] = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=(0.10, 0.5 * width, 0.5 * tallest),
                physicsClientId=cid,
            )
            self._obstacle_pools[obstacle_key] = []
        pool = self._obstacle_pools[obstacle_key]
        # The first obstacle keeps its historical position, clear of spawn.
        x, index = 1.2, 0
        while x <= TERRAIN_START_X + length:
            top = obstacle_height * (1.0 + rng.uniform(-height_jitter, height_jitter))
            if index == len(pool):
                pool.append(self._new_body(self._obstacle_shapes[obstacle_key]))
            self._place(pool[index], (x, 0.0, top - 0.5 * tallest))
            x += spacing + rng.uniform(-spacing_jitter, spacing_jitter)
            index += 1
