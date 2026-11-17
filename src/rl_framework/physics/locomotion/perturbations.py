"""Perturbations: lateral pushes, domain randomisation and action latency."""

from __future__ import annotations

from collections import deque

import numpy as np
import pybullet as p

from rl_framework.physics.locomotion.config import DomainRandomizationConfig
from rl_framework.physics.locomotion.robot import FRICTION_LINK_IDS
from rl_framework.physics.locomotion.terrain import PushSpec


class PushController:
    """Schedules and applies lateral pushes at the torso COM."""

    def __init__(self, connection: int):
        self._connection = connection
        self.ticks_remaining = 0
        self.vector = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self.ticks_remaining = 0

    def maybe_start(
        self,
        spec: PushSpec,
        step_count: int,
        rng: np.random.Generator,
        sim_timestep: float,
    ) -> bool:
        """Begin a push if this step is on schedule. Returns True if started."""
        if not (
            spec.force > 0.0
            and step_count >= spec.start_step
            and spec.interval_steps > 0
            and step_count % spec.interval_steps == 0
        ):
            return False
        direction = -1.0 if rng.integers(0, 2) else 1.0
        self.vector = (0.0, direction * spec.force, 0.0)
        self.ticks_remaining = max(1, int(round(spec.duration_s / sim_timestep)))
        return True

    def apply_tick(self, robot_id: int) -> None:
        """Apply one physics tick of the active push at the torso's COM.

        With ``WORLD_FRAME`` the position argument is a world coordinate. It
        used to be the world origin, so the push was a yaw torque whose lever
        arm was the distance walked: 1.2 rad/s at 2 m, 11.4 rad/s at 10 m,
        on a robot with no yaw actuator. Applying it at the current COM makes
        it a pure lateral push wherever the robot is.
        """
        com, _ = p.getBasePositionAndOrientation(
            robot_id, physicsClientId=self._connection
        )
        p.applyExternalForce(
            robot_id,
            -1,
            list(self.vector),
            list(com),
            p.WORLD_FRAME,
            physicsClientId=self._connection,
        )
        self.ticks_remaining -= 1


class DomainRandomizer:
    """Per-episode mass and friction scaling.

    The *scales* (not absolute values) are exposed in the observation as
    dimensionless multipliers centred on 1.0; without that, randomisation
    looks like pure noise to the agent and hurts training.
    """

    def __init__(self, connection: int):
        self._connection = connection
        self.mass_scale = 1.0
        self.friction_scale = 1.0

    def apply(
        self,
        cfg: DomainRandomizationConfig,
        rng: np.random.Generator,
        robot_id: int,
        nominal_masses: dict[int, float],
        base_friction: float,
    ) -> None:
        mass_rng, fric_rng = cfg.mass_scale_range, cfg.friction_range
        self.mass_scale = float(rng.uniform(mass_rng[0], mass_rng[1]))
        self.friction_scale = float(rng.uniform(fric_rng[0], fric_rng[1]))
        friction = base_friction * self.friction_scale
        for link_id, nominal_mass in nominal_masses.items():
            p.changeDynamics(
                robot_id,
                link_id,
                mass=nominal_mass * self.mass_scale,
                physicsClientId=self._connection,
            )
        for link_id in FRICTION_LINK_IDS:
            p.changeDynamics(
                robot_id,
                link_id,
                lateralFriction=friction,
                physicsClientId=self._connection,
            )


class ActionLatency:
    """Delays actions by ``steps`` agent steps (zero-filled while warming up)."""

    def __init__(self) -> None:
        self._buffer: deque[np.ndarray] = deque()

    def clear(self) -> None:
        self._buffer.clear()

    def reset(self, steps: int, action_shape: tuple[int, ...]) -> None:
        self._buffer.clear()
        noop = np.zeros(action_shape, dtype=np.float32)
        for _ in range(steps):
            self._buffer.append(noop)

    def delay(self, action: np.ndarray, steps: int) -> np.ndarray:
        if steps <= 0:
            return action
        self._buffer.append(action)
        return self._buffer.popleft()
