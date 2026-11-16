"""Contact and foot sensing, computed once per step as a snapshot.

``step`` used to call a dozen ``_foot_*`` helpers that each re-queried
PyBullet or took optional pre-fetched arguments. ``FootSensor.snapshot``
makes the two PyBullet queries (supporting-surface contacts, foot link
states) once and derives everything else from them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pybullet as p

from rl_framework.physics.locomotion.robot import FOOT_H, FOOT_LINKS

Contacts = dict[int, list[tuple]]
Pair = tuple[float, float]


def finite(values: Any) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(values, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6
    )


@dataclass(frozen=True)
class FootSnapshot:
    """Everything the gait tracker, rewards and telemetry need about the feet."""

    surface_contacts: Contacts
    link_states: list[tuple]
    foot_contacts: tuple[bool, bool]
    tangential_velocities: tuple[Pair, Pair]
    contact_slip_speeds: Pair
    slip_speeds: Pair
    normal_forces: Pair
    positions: tuple[Pair, Pair]

    @property
    def torso_contacts(self) -> list[tuple]:
        """Torso (base, link -1) contacts with any supporting surface."""
        return self.surface_contacts.get(-1, [])


class FootSensor:
    """Queries a walker body's contacts and foot states.

    ``support_body_ids`` is evaluated lazily because the terrain body pool
    changes every episode.
    """

    def __init__(
        self,
        connection: int,
        *,
        clearance_point: str,
        support_body_ids: Callable[[], Iterable[int]],
    ):
        self._connection = connection
        self.clearance_point = clearance_point
        self._support_body_ids = support_body_ids
        self.robot_id = -1

    # -- raw PyBullet queries ----------------------------------------
    def surface_contacts(self) -> Contacts:
        """Robot contacts with supporting surfaces, grouped by robot link.

        One ``getContactPoints`` call per step replaces the five per-link
        calls the step used to make; contact queries were about half of each
        env step against physics alone.
        """
        support = set(self._support_body_ids())
        grouped: Contacts = {}
        for contact in p.getContactPoints(
            bodyA=self.robot_id, physicsClientId=self._connection
        ):
            if contact[2] in support:
                grouped.setdefault(int(contact[3]), []).append(contact)
        return grouped

    def link_states(self) -> list[tuple]:
        return [
            p.getLinkState(
                self.robot_id,
                link,
                computeLinkVelocity=1,
                physicsClientId=self._connection,
            )
            for link in FOOT_LINKS
        ]

    # -- derived quantities ------------------------------------------
    def foot_contacts(self, grouped: Contacts | None = None) -> tuple[bool, bool]:
        grouped = self.surface_contacts() if grouped is None else grouped
        right, left = (bool(grouped.get(link)) for link in FOOT_LINKS)
        return right, left

    def contact_point_slip(
        self, grouped: Contacts | None = None, states: list[tuple] | None = None
    ) -> Pair:
        """True sliding speed of each foot: the velocity of the material point
        at the contact, not of the link origin.

        ``slip_speeds`` reports the link origin, which moves whenever the foot
        rotates about a stationary contact edge -- ordinary heel-off and
        toe-off. Measured across the reward_v2..v7 checkpoints, rotation
        accounts for 18-55% of what the origin reports, and the artifact grows
        with speed. Slip proper is

            v_contact = v_com + omega x (r_contact - r_com)

        evaluated at the slowest contact point, the one anchoring the foot.
        Feet out of contact report 0.0.
        """
        grouped = self.surface_contacts() if grouped is None else grouped
        states = self.link_states() if states is None else states
        speeds: list[float] = []
        for link, state in zip(FOOT_LINKS, states):
            points = grouped.get(link)
            if not points:
                speeds.append(0.0)
                continue
            com = np.asarray(state[0], dtype=np.float64)
            vel = finite(state[6])
            omega = finite(state[7])
            speeds.append(
                min(
                    float(
                        np.linalg.norm(
                            (
                                vel
                                + np.cross(
                                    omega,
                                    np.asarray(contact[5], dtype=np.float64) - com,
                                )
                            )[:2]
                        )
                    )
                    for contact in points
                )
            )
        return speeds[0], speeds[1]

    def tangential_velocities(
        self, states: list[tuple] | None = None
    ) -> tuple[Pair, Pair]:
        states = self.link_states() if states is None else states
        velocities = []
        for state in states:
            velocity = finite(state[6])
            velocities.append((float(velocity[0]), float(velocity[1])))
        return velocities[0], velocities[1]

    def slip_speeds(self) -> Pair:
        velocities = self.tangential_velocities()
        return (
            float(np.linalg.norm(velocities[0])),
            float(np.linalg.norm(velocities[1])),
        )

    def normal_forces(self, grouped: Contacts | None = None) -> Pair:
        """Total supporting-surface normal force for each foot."""
        grouped = self.surface_contacts() if grouped is None else grouped
        forces = []
        for link in FOOT_LINKS:
            force = sum(float(contact[9]) for contact in grouped.get(link, ()))
            forces.append(force if np.isfinite(force) else 0.0)
        return forces[0], forces[1]

    def sole_heights(self, states: list[tuple] | None = None) -> Pair:
        """World z of the lowest point of each foot's collision box.

        The box is centred on the link frame (the ankle). Its lowest corner
        sits ``sum_i |R[2, i]| * half_i`` below that centre.
        """
        states = self.link_states() if states is None else states
        heights = []
        half = np.asarray(FOOT_H, dtype=np.float64)
        for state in states:
            centre = finite(state[4])
            rotation = np.asarray(
                p.getMatrixFromQuaternion(state[5]), dtype=np.float64
            ).reshape(3, 3)
            heights.append(float(centre[2] - np.abs(rotation[2]) @ half))
        return heights[0], heights[1]

    def positions(self, states: list[tuple] | None = None) -> tuple[Pair, Pair]:
        """Sagittal foot positions used for stride/swing telemetry.

        x is always the link centre of mass. z follows ``gait.clearance_point``.
        """
        states = self.link_states() if states is None else states
        com_positions = [finite(state[0]) for state in states]
        if self.clearance_point == "sole":
            heights = self.sole_heights(states)
        else:
            heights = (float(com_positions[0][2]), float(com_positions[1][2]))
        return (
            (float(com_positions[0][0]), heights[0]),
            (float(com_positions[1][0]), heights[1]),
        )

    def snapshot(self) -> FootSnapshot:
        grouped = self.surface_contacts()
        states = self.link_states()
        tangential = self.tangential_velocities(states)
        return FootSnapshot(
            surface_contacts=grouped,
            link_states=states,
            foot_contacts=self.foot_contacts(grouped),
            tangential_velocities=tangential,
            contact_slip_speeds=self.contact_point_slip(grouped, states),
            slip_speeds=(
                float(np.linalg.norm(tangential[0])),
                float(np.linalg.norm(tangential[1])),
            ),
            normal_forces=self.normal_forces(grouped),
            positions=self.positions(states),
        )
