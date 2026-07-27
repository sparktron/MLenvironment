from __future__ import annotations

import pytest

from rl_framework.training.walker_diagnostics import (
    calibrate_walker_slip_thresholds,
    classify_walker_fall_window,
    summarize_walker_fall_windows,
)


def _row(
    step: int,
    *,
    touchdown: bool = False,
    contact: bool = True,
    stance_age: int | None = 0,
    speed: float = 0.1,
    force: float = 100.0,
) -> dict:
    foot = {
        "contact": contact,
        "touchdown": touchdown,
        "stance_age_steps": stance_age,
        "tangential_velocity": [speed, 0.0],
        "tangential_speed": speed,
        "normal_force": force,
    }
    return {
        "step": step,
        "time_to_fall_seconds": (10 - step) * 0.02,
        "roll": 0.1,
        "pitch": 0.2,
        "action_delta_l2": 0.3,
        "target_velocity": 1.0,
        "achieved_velocity": 0.8,
        "velocity_error": 0.2,
        "right": foot,
        "left": {
            **foot,
            "contact": False,
            "touchdown": False,
            "stance_age_steps": None,
        },
    }


def test_slip_thresholds_use_pooled_p90_samples() -> None:
    thresholds = calibrate_walker_slip_thresholds(
        list(range(1, 11)),
        list(range(10, 110, 10)),
    )

    assert thresholds["stance_slip_penalty_clip"] == pytest.approx(9.1)
    assert thresholds["touchdown_normal_force_threshold"] == pytest.approx(91.0)
    assert thresholds["contact_sample_count"] == 10
    assert thresholds["touchdown_sample_count"] == 10


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (
            [_row(step) for step in range(1, 10)]
            + [_row(10, touchdown=True, speed=0.8)],
            "touchdown_overshoot",
        ),
        (
            [_row(step, stance_age=step + 10, speed=0.8) for step in range(1, 11)],
            "late_stance_push_off",
        ),
        (
            [
                _row(
                    step,
                    contact=False,
                    stance_age=None,
                    speed=0.0,
                    force=0.0,
                )
                for step in range(1, 11)
            ],
            "recovery_failure",
        ),
    ],
)
def test_fall_window_classification_is_mutually_exclusive(
    window: list[dict], expected: str
) -> None:
    assert (
        classify_walker_fall_window(
            window,
            control_timestep=0.02,
            slip_threshold=0.5,
            touchdown_force_threshold=500.0,
        )
        == expected
    )


def test_fall_window_summary_preserves_raw_window_and_requested_measurements() -> None:
    window = [_row(step) for step in range(1, 10)]
    window.append(_row(10, touchdown=True, speed=0.8, force=900.0))

    result = summarize_walker_fall_windows(
        [{"episode": 2, "termination_reason": "torso_contact", "window": window}],
        control_timestep=0.02,
        thresholds={
            "stance_slip_penalty_clip": 0.5,
            "touchdown_normal_force_threshold": 500.0,
        },
    )

    assert result["fall_count"] == 1
    assert result["cause_counts"] == {"touchdown_overshoot": 1}
    fall = result["falls"][0]
    assert fall["window"] == window
    assert fall["touchdown_samples"][0]["side"] == "right"
    assert fall["midstance_samples"]
    assert fall["window_summary"]["normal_contact_force"]["max"] == 900.0
    assert fall["window_summary"]["absolute_pitch"]["mean"] == pytest.approx(0.2)
    assert fall["window_summary"]["absolute_velocity_error"]["mean"] == pytest.approx(
        0.2
    )
