"""Walker robot geometry and PyBullet body construction.

Atlas-DRC-class biped: 28 kg torso, 65.9 kg total. Link order (11 total)::

    0 rThigh  1 rShin  2 rFoot  3 lThigh  4 lShin  5 lFoot
    6 rUpperArm  7 rForearm  8 lUpperArm  9 lForearm  10 head
"""

from __future__ import annotations

import pybullet as p

from rl_framework.physics.locomotion.dynamics import (
    JOINT_INDICES,
    JOINT_SPECS,
    WalkerDynamics,
)

# Geometry constants (half-extents in metres)
TORSO_H = [0.14, 0.09, 0.17]
THIGH_H = [0.055, 0.055, 0.125]
SHIN_H = [0.045, 0.045, 0.115]
FOOT_H = [0.085, 0.05, 0.028]
UPPER_ARM_H = [0.04, 0.04, 0.11]
FORE_ARM_H = [0.035, 0.035, 0.09]
HEAD_H = [0.08, 0.07, 0.10]
LEG_DY = 0.085  # lateral (Y) hip offset — legs are side-by-side along Y
DYNAMIC_LINK_IDS = [-1, *range(11)]
FRICTION_LINK_IDS = [-1, 2, 5]
FOOT_LINKS = (2, 5)  # right, left
# Torso COM height when both feet rest flat on the ground:
#   ankle_z = foot_h[2], knee = ankle + 2*shin_h[2], hip = knee + 2*thigh_h[2]
#   torso_z = hip + torso_h[2]
TORSO_STAND_Z = FOOT_H[2] + 2 * SHIN_H[2] + 2 * THIGH_H[2] + TORSO_H[2]  # ≈ 0.678 m


def build_robot(
    connection: int,
    *,
    mass: float,
    friction: float,
    limb_collision: bool,
    dynamics: WalkerDynamics,
) -> tuple[int, dict[int, float]]:
    """Create the walker body. Returns ``(robot_id, nominal_link_masses)``.

    Nominal masses are captured after ``createMultiBody`` so domain
    randomisation can scale each link from its own baseline.
    """
    cid = connection
    th, sh, fh = THIGH_H, SHIN_H, FOOT_H
    ua, fa, hd = UPPER_ARM_H, FORE_ARM_H, HEAD_H
    leg_dy = LEG_DY
    arm_dy = TORSO_H[1] + ua[1]  # torso half-width + upper-arm half-width

    def _col(half):
        return p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)

    def _vis(half, color, offset=(0, 0, 0)):
        return p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half,
            rgbaColor=color,
            visualFramePosition=list(offset),
            physicsClientId=cid,
        )

    GRAY = [0.45, 0.45, 0.45, 1.0]
    LGRAY = [0.60, 0.60, 0.60, 1.0]  # lighter gray for head & forearms
    RED = [0.80, 0.15, 0.15, 1.0]
    BLUE = [0.15, 0.15, 0.80, 1.0]
    DGRAY = [0.30, 0.30, 0.30, 1.0]

    torso_col = _col(TORSO_H)
    torso_vis = _vis(TORSO_H, GRAY)

    # ── Legs ──────────────────────────────────────────────────────────────
    # Thigh/shin visuals hang down from their joint (top of segment = joint).
    # Foot visuals sit exactly on the collision box, centred on the ankle.
    # They used to be drawn 4 cm forward of it, so videos showed the foot
    # where it does not collide. The inertial frame keeps its 4 cm offset.
    r_thigh_vis = _vis(th, RED, (0, 0, -th[2]))
    r_shin_vis = _vis(sh, BLUE, (0, 0, -sh[2]))
    r_foot_col = _col(fh)
    r_foot_vis = _vis(fh, DGRAY)
    l_thigh_vis = _vis(th, RED, (0, 0, -th[2]))
    l_shin_vis = _vis(sh, BLUE, (0, 0, -sh[2]))
    l_foot_col = _col(fh)
    l_foot_vis = _vis(fh, DGRAY)

    def _limb_col(half):
        # Thighs and shins collide only when enabled. Off by default so
        # existing checkpoints keep their physics; obstacle results with it
        # off are optimistic, because a shin cannot catch an edge.
        if not limb_collision:
            return -1
        return p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half,
            collisionFramePosition=[0, 0, -half[2]],
            physicsClientId=cid,
        )

    r_thigh_col, r_shin_col = _limb_col(th), _limb_col(sh)
    l_thigh_col, l_shin_col = _limb_col(th), _limb_col(sh)

    # ── Arms ──────────────────────────────────────────────────────────────
    # Upper arms hang from shoulder joint; forearms hang from elbow joint.
    r_ua_vis = _vis(ua, GRAY, (0, 0, -ua[2]))
    r_fa_vis = _vis(fa, LGRAY, (0, 0, -fa[2]))
    l_ua_vis = _vis(ua, GRAY, (0, 0, -ua[2]))
    l_fa_vis = _vis(fa, LGRAY, (0, 0, -fa[2]))

    # ── Head ──────────────────────────────────────────────────────────────
    # Neck joint sits at the top of the torso; head box rises above it.
    head_vis = _vis(hd, LGRAY, (0, 0, hd[2]))

    ident = [0, 0, 0, 1]
    shoulder_z = TORSO_H[2] - 0.04  # slightly below top of torso

    # linkParentIndices use 1-based link indexing (0 = base).
    robot_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=torso_col,
        baseVisualShapeIndex=torso_vis,
        # Atlas DRC-class mass distribution (kg). Source: atlas_v3.urdf
        # released by Boston Dynamics for the DARPA Robotics Challenge sim.
        # Total = 28 (torso, set via baseMass) + 1.5 (head) + 14 (thighs)
        #       + 7.4 (shins) + 4 (feet) + 6 (upper arms) + 5 (forearms)
        #       ≈ 65.9 kg (hardware Atlas is ~82 kg incl. battery/hands).
        linkMasses=[
            7.0,
            3.7,
            2.0,  # right leg: thigh, shin, foot
            7.0,
            3.7,
            2.0,  # left leg
            3.0,
            2.5,  # right arm: upper, forearm
            3.0,
            2.5,  # left arm
            1.5,  # head
        ],
        linkCollisionShapeIndices=[
            r_thigh_col,
            r_shin_col,
            r_foot_col,  # right leg (thigh/shin only with limb_collision)
            l_thigh_col,
            l_shin_col,
            l_foot_col,  # left leg
            -1,
            -1,  # right arm (no collision)
            -1,
            -1,  # left arm
            -1,  # head
        ],
        linkVisualShapeIndices=[
            r_thigh_vis,
            r_shin_vis,
            r_foot_vis,
            l_thigh_vis,
            l_shin_vis,
            l_foot_vis,
            r_ua_vis,
            r_fa_vis,
            l_ua_vis,
            l_fa_vis,
            head_vis,
        ],
        linkPositions=[
            # ── Legs: hips attach at bottom of torso, separated along ±Y ──
            [0, -leg_dy, -TORSO_H[2]],  # 0 right hip
            [0, 0, -2 * th[2]],  # 1 right knee
            [0, 0, -2 * sh[2]],  # 2 right ankle
            [0, leg_dy, -TORSO_H[2]],  # 3 left hip
            [0, 0, -2 * th[2]],  # 4 left knee
            [0, 0, -2 * sh[2]],  # 5 left ankle
            # ── Arms: shoulders attach at upper sides of torso, ±Y ──
            [0, -arm_dy, shoulder_z],  # 6 right shoulder
            [0, 0, -2 * ua[2]],  # 7 right elbow
            [0, arm_dy, shoulder_z],  # 8 left shoulder
            [0, 0, -2 * ua[2]],  # 9 left elbow
            # ── Head: neck at top of torso ──
            [0, 0, TORSO_H[2]],  # 10 neck/head
        ],
        linkOrientations=[ident] * 11,
        linkInertialFramePositions=[
            [0, 0, -th[2]],
            [0, 0, -sh[2]],
            [0.04, 0, 0],  # right leg
            [0, 0, -th[2]],
            [0, 0, -sh[2]],
            [0.04, 0, 0],  # left leg
            [0, 0, -ua[2]],
            [0, 0, -fa[2]],  # right arm
            [0, 0, -ua[2]],
            [0, 0, -fa[2]],  # left arm
            [0, 0, hd[2]],  # head
        ],
        linkInertialFrameOrientations=[ident] * 11,
        # 0=base, 1=link-0, 2=link-1, …
        linkParentIndices=[0, 1, 2, 0, 4, 5, 0, 7, 0, 9, 0],
        linkJointTypes=[p.JOINT_REVOLUTE] * 10 + [p.JOINT_FIXED],
        # All revolute joints rotate around Y → sagittal-plane forward/back swing.
        # Axis for fixed head joint is arbitrary.
        linkJointAxis=[[0, 1, 0]] * 10 + [[0, 0, 1]],
        physicsClientId=cid,
    )

    # Friction on base torso and both feet.
    for link in FRICTION_LINK_IDS:
        p.changeDynamics(robot_id, link, lateralFriction=friction, physicsClientId=cid)
    # Joint limits + light damping. Without limits, knees would hyperextend
    # backward, ankles would spin freely, etc. — the body becomes nonsensical.
    for j in JOINT_INDICES:
        spec = JOINT_SPECS[j]
        p.changeDynamics(
            robot_id,
            j,
            jointLowerLimit=spec.low,
            jointUpperLimit=spec.high,
            jointDamping=0.5,
            physicsClientId=cid,
        )
    # Under TORQUE_CONTROL the implicit velocity motor must be silenced.
    # Under POSITION_CONTROL (our default) it's harmless but cheap to do.
    dynamics.disable_velocity_motors(robot_id, physicsClientId=cid)
    nominal_masses = {
        link_id: float(p.getDynamicsInfo(robot_id, link_id, physicsClientId=cid)[0])
        for link_id in DYNAMIC_LINK_IDS
    }
    return robot_id, nominal_masses
