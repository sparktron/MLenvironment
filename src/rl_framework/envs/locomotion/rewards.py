from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class WalkerReward:
    """Shaped reward for the bipedal walker.

    Design notes (vs. the original v-shape velocity term):
    - ``alive_bonus`` is deliberately small: velocity tracking is the main
      positive signal, while survival remains a gentle bootstrap incentive.
    - The velocity reward is a **Gaussian around the target**, so standing
      still scores a small positive (≈ alive_bonus + 0.3) instead of being
      penalised harder than falling.
    - A one-time ``fall_penalty`` fires only on the terminal step of a fall
      (not on truncation), giving a clear gradient toward staying upright.
    - ``orientation_penalty_weight`` is scaled down to make the early
      bootstrap easier; raise it later via curriculum.
    """

    alive_bonus: float = 0.25
    forward_velocity_weight: float = 2.0
    target_velocity: float = 1.0
    # Width (1 sigma) of the Gaussian around target_velocity. Larger →
    # broader credit for "close to target" speeds.
    velocity_sigma: float = 0.5
    orientation_penalty_weight: float = 0.3
    torque_penalty_weight: float = 0.01
    # One-time penalty applied on the step a fall is detected.
    fall_penalty: float = 10.0
    # Structural gait terms are opt-in so existing configs and checkpoints
    # retain bit-for-bit reward semantics.
    gait_step_progress_weight: float = 0.0
    gait_step_progress_clip: float = 0.25
    swing_touchdown_progress_weight: float = 0.0
    swing_touchdown_progress_clip: float = 0.25
    stance_slip_penalty_weight: float = 0.0
    # Zero preserves the original unbounded penalty. Positive values cap
    # contact-impulse outliers without changing the measured slip telemetry.
    stance_slip_penalty_clip: float = 0.0

    # ── Velocity reward shape ────────────────────────────────────────────
    # "gaussian" is the historical shape and stays the default so existing
    # configs and checkpoints keep their reward semantics. "clipped_linear"
    # pays ``weight * clip(v, 0, target) / target``, which holds a constant
    # gradient at *every* speed below target. The Gaussian does not: at
    # ``velocity_sigma: 0.10`` with ``target_velocity: 1.0`` the reward is
    # 3.3e-4 at 0.6 m/s and 1.3e-14 at 0.2 m/s, so a policy that is not
    # already near target receives no usable velocity signal at all.
    velocity_mode: str = "gaussian"

    # ── Gait-structure terms (all opt-in, zero-default) ──────────────────
    # Penalises ``||a_t - a_{t-1}||^2``. ``torque_penalty_weight`` charges
    # action *magnitude*, which taxes a sustained stride while leaving
    # high-frequency chatter about zero free; this term is the reverse.
    action_rate_penalty_weight: float = 0.0
    # Dense per-step credit for lifting a swing foot, saturating at
    # ``swing_clearance_target`` metres. Pays the gated quantity directly
    # instead of hoping it emerges from a displacement proxy.
    swing_clearance_weight: float = 0.0
    swing_clearance_target: float = 0.05
    # Penalises touchdowns arriving faster than ``max_cycle_frequency_hz``.
    # One gait cycle is two alternating touchdowns, so the minimum sane
    # inter-touchdown interval is ``1 / (2 * max_cycle_frequency_hz)``.
    touchdown_rate_penalty_weight: float = 0.0
    max_cycle_frequency_hz: float = 2.5

    @property
    def min_touchdown_interval(self) -> float:
        """Shortest inter-touchdown interval that is not penalised (seconds)."""
        return 1.0 / (2.0 * max(self.max_cycle_frequency_hz, 1e-6))

    def compute(
        self,
        lin_vel_x: float,
        pitch_roll_penalty: float,
        action: np.ndarray,
        alive: bool,
        fell: bool = False,
        gait_step_progress: float = 0.0,
        sustained_swing_progress: float = 0.0,
        stance_slip_speed: float = 0.0,
        action_rate_l2: float = 0.0,
        swing_clearance: float = 0.0,
        touchdown_interval: float | None = None,
    ) -> float:
        return float(
            sum(
                self.components(
                    lin_vel_x=lin_vel_x,
                    pitch_roll_penalty=pitch_roll_penalty,
                    action=action,
                    alive=alive,
                    fell=fell,
                    gait_step_progress=gait_step_progress,
                    sustained_swing_progress=sustained_swing_progress,
                    stance_slip_speed=stance_slip_speed,
                    action_rate_l2=action_rate_l2,
                    swing_clearance=swing_clearance,
                    touchdown_interval=touchdown_interval,
                ).values()
            )
        )

    def components(
        self,
        lin_vel_x: float,
        pitch_roll_penalty: float,
        action: np.ndarray,
        alive: bool,
        fell: bool = False,
        gait_step_progress: float = 0.0,
        sustained_swing_progress: float = 0.0,
        stance_slip_speed: float = 0.0,
        action_rate_l2: float = 0.0,
        swing_clearance: float = 0.0,
        touchdown_interval: float | None = None,
    ) -> dict[str, float]:
        """Return individually attributable reward terms.

        Keeping the decomposition in the reward object prevents diagnostics
        from reimplementing—and eventually drifting from—the actual reward.
        The terms sum exactly to :meth:`compute`.
        """
        alive_reward = self.alive_bonus if alive else 0.0
        if self.velocity_mode == "clipped_linear":
            # Constant gradient for every speed in [0, target]; saturates at
            # target so overspeed is neither rewarded nor punished.
            target = max(self.target_velocity, 1e-6)
            tracked = min(max(float(lin_vel_x), 0.0), target) / target
            velocity_reward = self.forward_velocity_weight * tracked
        elif self.velocity_mode == "gaussian":
            # Gaussian velocity reward: 1.0 at v=target, decays smoothly outside.
            diff = (lin_vel_x - self.target_velocity) / max(self.velocity_sigma, 1e-6)
            velocity_reward = self.forward_velocity_weight * math.exp(
                -0.5 * diff * diff
            )
        else:
            raise ValueError(
                f"Unknown velocity_mode {self.velocity_mode!r}; "
                "expected 'gaussian' or 'clipped_linear'"
            )
        orientation_penalty = -self.orientation_penalty_weight * pitch_roll_penalty
        action_penalty = -self.torque_penalty_weight * float(action @ action)
        fall_penalty = -self.fall_penalty if fell else 0.0
        clipped_progress = min(
            max(float(gait_step_progress), 0.0),
            max(self.gait_step_progress_clip, 0.0),
        )
        gait_step_progress_reward = self.gait_step_progress_weight * clipped_progress
        clipped_sustained_swing_progress = min(
            max(float(sustained_swing_progress), 0.0),
            max(self.swing_touchdown_progress_clip, 0.0),
        )
        swing_touchdown_progress_reward = (
            self.swing_touchdown_progress_weight * clipped_sustained_swing_progress
        )
        penalized_slip = max(float(stance_slip_speed), 0.0)
        if self.stance_slip_penalty_clip > 0.0:
            penalized_slip = min(penalized_slip, self.stance_slip_penalty_clip)
        stance_slip_penalty = -self.stance_slip_penalty_weight * penalized_slip
        action_rate_penalty = -self.action_rate_penalty_weight * max(
            float(action_rate_l2), 0.0
        )
        swing_clearance_reward = self.swing_clearance_weight * min(
            max(float(swing_clearance), 0.0), max(self.swing_clearance_target, 0.0)
        )
        # Fires only on a touchdown step, scaled by how far inside the minimum
        # interval the contact arrived (1.0 == instantaneous re-contact).
        touchdown_rate_penalty = 0.0
        if touchdown_interval is not None and self.touchdown_rate_penalty_weight:
            minimum = self.min_touchdown_interval
            shortfall = max(minimum - max(float(touchdown_interval), 0.0), 0.0)
            touchdown_rate_penalty = -self.touchdown_rate_penalty_weight * (
                shortfall / minimum
            )
        return {
            "alive": float(alive_reward),
            "velocity": float(velocity_reward),
            "orientation": float(orientation_penalty),
            "action": float(action_penalty),
            "fall": float(fall_penalty),
            "gait_step_progress": float(gait_step_progress_reward),
            "swing_touchdown_progress": float(swing_touchdown_progress_reward),
            "stance_slip": float(stance_slip_penalty),
            "action_rate": float(action_rate_penalty),
            "swing_clearance": float(swing_clearance_reward),
            "touchdown_rate": float(touchdown_rate_penalty),
        }
