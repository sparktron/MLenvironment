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
    # WARNING: this pays ``swing_clearance``, which the env feeds from
    # ``GaitStep.swing_clearance_now`` -- the MAX over currently-swinging feet.
    # A policy can therefore collect full credit by swinging one leg high and
    # dragging the other, and the reward_v4 screen did exactly that: earned
    # clearance fell only 29% while measured per-foot clearance fell 49%.
    # Retained at zero default so v2-v4 stay reproducible; prefer
    # ``touchdown_clearance_weight`` for new work.
    swing_clearance_weight: float = 0.0
    swing_clearance_target: float = 0.05
    # Per-foot, paid on the step a foot lands, proportional to the clearance
    # that foot actually achieved. Not farmable by one leg, but the payment
    # scales with the NUMBER of touchdowns, so it rewards higher cadence and
    # works against ``touchdown_rate_penalty_weight``. Prefer
    # ``bilateral_clearance_weight`` unless you specifically want that coupling.
    touchdown_clearance_weight: float = 0.0
    touchdown_clearance_target: float = 0.05
    # Dense credit for the LOWER of the two feet's most recent swing
    # clearances. Both feet must lift to earn it, and because it pays per step
    # rather than per landing it is independent of cadence.
    bilateral_clearance_weight: float = 0.0
    bilateral_clearance_target: float = 0.05
    # Penalises touchdowns arriving faster than ``max_cycle_frequency_hz``.
    # One gait cycle is two alternating touchdowns, so the minimum sane
    # inter-touchdown interval is ``1 / (2 * max_cycle_frequency_hz)``.
    touchdown_rate_penalty_weight: float = 0.0
    max_cycle_frequency_hz: float = 2.5
    # Charged on every step where neither foot is on a supporting surface.
    # A walk has no flight phase; a run is defined by having one. Nothing else
    # in the reward distinguishes the two, so a policy paid only for clearance
    # and stride will happily bound.
    flight_penalty_weight: float = 0.0
    # Charged in proportion to the running left/right stance duty imbalance.
    # Without it a policy can satisfy every stride and clearance term while
    # favouring one leg and dragging the other.
    gait_symmetry_penalty_weight: float = 0.0
    # Paid on every step with BOTH feet supported. ``flight_penalty_weight``
    # charges the complement of "grounded", which a policy can satisfy by
    # extending single support -- the v3 screen drove flight 57.6% -> 29.0%
    # while double support stayed at 0.4%. This term prices the target state
    # directly. Keep it well below the velocity weight: standing still scores
    # 100% double support, so an oversized weight makes standing competitive.
    double_support_reward_weight: float = 0.0
    # Paid for matching a periodic reference contact schedule driven by a gait
    # clock in the observation. Every per-step *property* term above constrains
    # a marginal -- how high a foot lifts, how often it lands, how evenly the
    # legs share load -- and four screens showed each can be satisfied by a
    # non-walking gait, most starkly by hopping with both legs in phase. Only a
    # phase-indexed target expresses the anti-phase ORDERING that defines a
    # walk. Input is the fraction of feet whose contact matches the schedule.
    phase_contact_weight: float = 0.0

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
        touchdown_clearance: float | None = None,
        bilateral_clearance: float = 0.0,
        phase_contact_match: float = 0.0,
        in_flight: bool = False,
        in_double_support: bool = False,
        contact_duty_imbalance: float = 0.0,
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
                    touchdown_clearance=touchdown_clearance,
                    bilateral_clearance=bilateral_clearance,
                    phase_contact_match=phase_contact_match,
                    in_flight=in_flight,
                    in_double_support=in_double_support,
                    contact_duty_imbalance=contact_duty_imbalance,
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
        touchdown_clearance: float | None = None,
        bilateral_clearance: float = 0.0,
        phase_contact_match: float = 0.0,
        in_flight: bool = False,
        in_double_support: bool = False,
        contact_duty_imbalance: float = 0.0,
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
        touchdown_clearance_reward = 0.0
        if touchdown_clearance is not None:
            touchdown_clearance_reward = self.touchdown_clearance_weight * min(
                max(float(touchdown_clearance), 0.0),
                max(self.touchdown_clearance_target, 0.0),
            )
        bilateral_clearance_reward = self.bilateral_clearance_weight * min(
            max(float(bilateral_clearance), 0.0),
            max(self.bilateral_clearance_target, 0.0),
        )
        phase_contact_reward = self.phase_contact_weight * min(
            max(float(phase_contact_match), 0.0), 1.0
        )
        flight_penalty = -self.flight_penalty_weight if in_flight else 0.0
        double_support_reward = (
            self.double_support_reward_weight if in_double_support else 0.0
        )
        gait_symmetry_penalty = -self.gait_symmetry_penalty_weight * min(
            max(float(contact_duty_imbalance), 0.0), 1.0
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
            "touchdown_clearance": float(touchdown_clearance_reward),
            "bilateral_clearance": float(bilateral_clearance_reward),
            "phase_contact": float(phase_contact_reward),
            "touchdown_rate": float(touchdown_rate_penalty),
            "flight": float(flight_penalty),
            "double_support": float(double_support_reward),
            "gait_symmetry": float(gait_symmetry_penalty),
        }
