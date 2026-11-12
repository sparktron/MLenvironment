"""Walker training callbacks."""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from rl_framework.training.velocity_target_ramp_callback import VelocityTargetRampCallback

class WalkerMetricsCallback(BaseCallback):
    """Log terminal walker behavior and reward decomposition per rollout."""

    _LOG_NAMES = {
        "episode_length": "episode_length_mean",
        "forward_displacement": "forward_displacement_mean",
        "fall": "fall_rate",
        "mean_abs_tilt": "abs_tilt_mean",
        "mean_action_l2": "action_l2_mean",
        "action_scale": "action_scale",
        "reward_alive_mean": "reward_alive_mean",
        "reward_velocity_mean": "reward_velocity_mean",
        "reward_orientation_mean": "reward_orientation_mean",
        "reward_action_mean": "reward_action_mean",
        "reward_fall_mean": "reward_fall_mean",
        "reward_gait_step_progress_mean": "reward_gait_step_progress_mean",
        "reward_stance_slip_mean": "reward_stance_slip_mean",
        "reward_swing_touchdown_progress_mean": (
            "reward_swing_touchdown_progress_mean"
        ),
        "reward_action_rate_mean": "reward_action_rate_mean",
        "reward_swing_clearance_mean": "reward_swing_clearance_mean",
        "reward_touchdown_clearance_mean": "reward_touchdown_clearance_mean",
        "reward_bilateral_clearance_mean": "reward_bilateral_clearance_mean",
        "reward_phase_contact_mean": "reward_phase_contact_mean",
        "reward_phase_double_support_mean": "reward_phase_double_support_mean",
        "reward_touchdown_rate_mean": "reward_touchdown_rate_mean",
        "reward_flight_mean": "reward_flight_mean",
        "reward_double_support_mean": "reward_double_support_mean",
        "reward_gait_symmetry_mean": "reward_gait_symmetry_mean",
        "gait_right_only_fraction": "gait_right_only_fraction",
        "gait_left_only_fraction": "gait_left_only_fraction",
        "gait_double_support_fraction": "gait_double_support_fraction",
        "gait_flight_fraction": "gait_flight_fraction",
        "gait_contact_duty_imbalance": "gait_contact_duty_imbalance",
        "gait_right_touchdowns": "gait_right_touchdowns_mean",
        "gait_left_touchdowns": "gait_left_touchdowns_mean",
        "gait_alternating_touchdowns": "gait_alternating_touchdowns_mean",
        "gait_alternating_touchdowns_per_100_steps": (
            "gait_alternating_touchdowns_per_100_steps"
        ),
        "gait_progress_per_alternating_touchdown": (
            "gait_progress_per_alternating_touchdown"
        ),
        "gait_stance_slip_speed": "gait_stance_slip_speed",
        "gait_mid_stance_slip_speed": "gait_mid_stance_slip_speed",
        "gait_contact_point_slip_speed": "gait_contact_point_slip_speed",
        "gait_mid_stance_contact_point_slip_speed": (
            "gait_mid_stance_contact_point_slip_speed"
        ),
        "gait_longest_same_foot_sequence": ("gait_longest_same_foot_sequence_mean"),
        "gait_mean_action_delta_l2": "gait_mean_action_delta_l2",
        "gait_stride_length": "gait_stride_length",
        "gait_swing_duration": "gait_swing_duration",
        "gait_foot_clearance": "gait_foot_clearance",
        "gait_cadence_steps_per_min": "gait_cadence_steps_per_min",
        "gait_qualified_touchdowns": "gait_qualified_touchdowns_mean",
    }

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self._values: dict[str, list[float]] = {
            metric: [] for metric in self._LOG_NAMES
        }

    def _on_step(self) -> bool:
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            summary = info.get("walker_episode")
            if not isinstance(summary, dict):
                continue
            for metric in self._LOG_NAMES:
                value = summary.get(metric)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._values[metric].append(float(value))
        return True

    def _on_rollout_end(self) -> None:
        for metric, values in self._values.items():
            if values:
                self.logger.record(
                    f"walker/{self._LOG_NAMES[metric]}", float(np.mean(values))
                )
                values.clear()


def train_callbacks(cfg, paths, slot):
    if slot == "before_curriculum":
        return [WalkerMetricsCallback()]
    if slot != "after_checkpoint":
        return []
    ramp_cfg = cfg["training"].get("velocity_target_ramp")
    first = []
    if ramp_cfg is not None:
        first.append(VelocityTargetRampCallback(
            start=float(ramp_cfg["start"]), end=float(ramp_cfg["end"]),
            ramp_steps=int(ramp_cfg["ramp_steps"]),
            sigma_fraction=None if ramp_cfg.get("sigma_fraction") is None else float(ramp_cfg["sigma_fraction"])))
    return first
