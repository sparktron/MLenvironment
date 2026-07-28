from __future__ import annotations

from pathlib import Path

import pytest
from rl_framework.training import quality_study
from rl_framework.training.quality_study import StopOnWallClock, run_quality_study
from rl_framework.training.walker_diagnostics import _verdict


def _diagnostic_result() -> dict:
    metrics = {
        "episode_length_mean": 700.0,
        "episode_length_std": 0.0,
        "return_mean": 10.0,
        "return_std": 0.0,
        "fall_rate": 0.1,
        "peak_z_mean": 0.9,
        "forward_displacement_mean": 2.0,
        "forward_displacement_std": 0.0,
        "pushes_mean": 1.0,
        "push_recovery_rate": 0.8,
        # A plausible walking gait: ~1.2 Hz cycle, 30 cm strides, 4 cm
        # clearance. The pre-correction fixture used 18 touchdowns per 100
        # steps with 4 mm of progress, which is the contact chatter the gait
        # band now rejects.
        "gait_alternating_touchdowns_per_100_steps_mean": 4.0,
        "gait_progress_per_alternating_touchdown_mean": 0.15,
        "gait_stride_length_mean": 0.30,
        "gait_foot_clearance_mean": 0.04,
        "gait_stance_slip_speed_mean": 0.15,
        "gait_contact_point_slip_speed_mean": 0.15,
        "gait_double_support_fraction_mean": 0.15,
        "gait_flight_fraction_mean": 0.05,
        "gait_longest_same_foot_sequence_mean": 3.0,
    }
    return {
        "algorithm": "PPO",
        "model_path": "model.zip",
        "model_timesteps": 64,
        "episodes": 1,
        "zero_action": {**metrics, "episode_length_mean": 100.0},
        "deterministic": metrics,
        "stochastic": metrics,
        "verdict": "partial_learning_deterministic",
    }


def test_quality_study_dry_run_reports_full_plan(tmp_path: Path) -> None:
    result = run_quality_study(
        "all", seeds=[0, 1, 2], output_dir=tmp_path, dry_run=True
    )

    assert result["planned_runs"] == {
        "walker": 12,
        "walker_velocity": 0,
        "walker_gait": 0,
        "walker_velocity_ramp": 0,
        "walker_swing_touchdown": 0,
        "walker_slip_recovery": 0,
        "walker_action_memory": 0,
        "arena": 18,
        "algorithms": 18,
    }
    assert list(tmp_path.iterdir()) == []


def test_walker_quality_study_persists_results_and_is_resumable(
    tmp_path: Path, monkeypatch
) -> None:
    train_calls = []

    def fake_train(cfg, extra_callbacks=None, resume_from=None):
        train_calls.append(cfg)
        model = (
            Path(cfg["output"]["base_dir"])
            / cfg["experiment_name"]
            / f"seed_{cfg['seed']}"
            / "checkpoints"
            / "final_model"
        )
        model.parent.mkdir(parents=True, exist_ok=True)
        model.with_suffix(".zip").write_bytes(b"model")
        return model

    monkeypatch.setattr(quality_study, "train", fake_train)
    monkeypatch.setattr(
        quality_study,
        "evaluate_walker_transfer_suite",
        lambda cfg, path, episodes: {
            terrain: _diagnostic_result()
            for terrain in ("flat", "uneven", "obstacles", "push_recovery")
        },
    )

    first = run_quality_study(
        "walker",
        seeds=[0],
        output_dir=tmp_path,
        step_budget=300_000,
        eval_episodes=1,
    )
    assert first["completed"] is True
    assert len(train_calls) == len(quality_study.WALKER_VARIANTS)
    assert {
        cfg["experiment_name"].removeprefix("quality_walker_") for cfg in train_calls
    } == set(quality_study.WALKER_VARIANTS)
    for cfg in train_calls:
        assert {
            key: cfg["training"][key] for key in quality_study.WALKER_STUDY_TRAINING
        } == quality_study.WALKER_STUDY_TRAINING
        assert cfg["environment"]["sim"]["mass"] == 28.0
        assert 0 < cfg["environment"]["sim"]["action_scale"] <= 1.0
        assert cfg["training"]["log_std_init"] <= -1.0
        assert cfg["curriculum"]["warmup_steps"] == 200_000
    assert first["results"]["walker"]["promotion_ready"] is False
    assert first["results"]["walker"]["evidence_ready"] is False
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()

    resumed = run_quality_study(
        "walker",
        seeds=[0],
        output_dir=tmp_path,
        step_budget=300_000,
        eval_episodes=1,
        resume=True,
    )
    assert resumed["completed"] is True
    assert len(train_calls) == len(quality_study.WALKER_VARIANTS)


def test_walker_velocity_study_resumes_balance_checkpoints(
    tmp_path: Path, monkeypatch
) -> None:
    source_dir = tmp_path / "balance"
    source_model = source_dir / "seed_7" / "checkpoints" / "final_model.zip"
    source_model.parent.mkdir(parents=True)
    source_model.write_bytes(b"balance")
    train_calls = []

    def fake_train(cfg, extra_callbacks=None, resume_from=None):
        train_calls.append((cfg, resume_from))
        model = (
            Path(cfg["output"]["base_dir"])
            / cfg["experiment_name"]
            / f"seed_{cfg['seed']}"
            / "checkpoints"
            / "final_model"
        )
        model.parent.mkdir(parents=True, exist_ok=True)
        model.with_suffix(".zip").write_bytes(b"continued")
        return model

    monkeypatch.setattr(quality_study, "train", fake_train)
    monkeypatch.setattr(
        quality_study,
        "evaluate_walker_checkpoint",
        lambda cfg, path, episodes: _diagnostic_result(),
    )

    result = run_quality_study(
        "walker-velocity",
        seeds=[7],
        source_dir=source_dir,
        output_dir=tmp_path / "study",
        step_budget=100_000,
        eval_episodes=1,
    )

    assert result["completed"] is True
    assert len(train_calls) == len(quality_study.WALKER_VELOCITY_VARIANTS)
    assert all(resume_from == source_model for _, resume_from in train_calls)
    assert {
        cfg["environment"]["reward"]["velocity_sigma"] for cfg, _ in train_calls
    } == {0.10, 0.15}
    assert result["results"]["walker_velocity"]["evidence_ready"] is False
    assert result["results"]["walker_velocity"]["promotion_ready"] is False


def test_walker_gait_study_resumes_seed_matched_continuations(
    tmp_path: Path, monkeypatch
) -> None:
    source_dir = tmp_path / "velocity"
    source_model = source_dir / "seed_7" / "checkpoints" / "final_model.zip"
    source_model.parent.mkdir(parents=True)
    source_model.write_bytes(b"velocity")
    train_calls = []

    def fake_train(cfg, extra_callbacks=None, resume_from=None):
        train_calls.append((cfg, resume_from))
        model = (
            Path(cfg["output"]["base_dir"])
            / cfg["experiment_name"]
            / f"seed_{cfg['seed']}"
            / "checkpoints"
            / "final_model"
        )
        model.parent.mkdir(parents=True, exist_ok=True)
        model.with_suffix(".zip").write_bytes(b"continued")
        return model

    monkeypatch.setattr(quality_study, "train", fake_train)
    monkeypatch.setattr(
        quality_study,
        "evaluate_walker_checkpoint",
        lambda cfg, path, episodes: _diagnostic_result(),
    )

    result = run_quality_study(
        "walker-gait",
        seeds=[7],
        source_dir=source_dir,
        output_dir=tmp_path / "study",
        step_budget=50_000,
        eval_episodes=1,
    )

    assert result["completed"] is True
    assert len(train_calls) == len(quality_study.WALKER_GAIT_VARIANTS)
    assert all(resume_from == source_model for _, resume_from in train_calls)
    assert {
        cfg["environment"]["reward"]["velocity_sigma"] for cfg, _ in train_calls
    } == {0.10}
    assert {
        cfg["environment"]["reward"]["gait_step_progress_weight"]
        for cfg, _ in train_calls
    } == {0.0, 40.0}
    assert result["results"]["walker_gait"]["evidence_ready"] is False
    assert result["results"]["walker_gait"]["promotion_ready"] is False


def test_walker_velocity_ramp_requires_frozen_gates_and_control_improvement() -> None:
    assert quality_study.WALKER_VELOCITY_RAMP_VARIANTS == {
        "control": {},
        "ramp_to_100": {"target_velocity_end": 1.0},
    }
    assert quality_study.WALKER_VELOCITY_RAMP_STEPS == 3_000_000

    control = _diagnostic_result()
    candidate = _diagnostic_result()
    control["stochastic"] = {**control["stochastic"], "forward_displacement_mean": 1.2}
    candidate["stochastic"] = {
        **candidate["stochastic"],
        "forward_displacement_mean": 1.5,
    }

    aggregate = quality_study._aggregate_walker_velocity_ramp_results(
        {"control/seed_21": control, "ramp_to_100/seed_21": candidate}
    )

    ramp = next(row for row in aggregate["rankings"] if row["variant"] == "ramp_to_100")
    assert ramp["frozen_gate_passed"] is True
    assert ramp["displacement_improvement_over_control"] == pytest.approx(0.3)
    assert ramp["gate_passed"] is True
    assert aggregate["promoted_variant"] == "ramp_to_100"


def test_walker_slip_recovery_dry_run_reports_conditional_screen(
    tmp_path: Path,
) -> None:
    result = run_quality_study(
        "walker-slip-recovery",
        seeds=[21, 22, 23],
        output_dir=tmp_path,
        source_dir=tmp_path / "source",
        dry_run=True,
    )

    assert result["planned_runs"]["walker_slip_recovery"] == 3
    assert result["identity"]["source_dir"] == str(tmp_path / "source")
    assert list(tmp_path.iterdir()) == []


def test_walker_slip_recovery_gate_rejects_sliding_and_selects_sole_winner() -> None:
    control = _diagnostic_result()
    control["stochastic"] = {
        **control["stochastic"],
        "fall_rate": 0.4,
        "forward_displacement_mean": 6.0,
        "gait_contact_point_slip_speed_mean": 0.4,
    }
    candidate = _diagnostic_result()
    candidate["stochastic"] = {
        **candidate["stochastic"],
        "fall_rate": 0.2,
        "forward_displacement_mean": 6.0,
        "gait_contact_point_slip_speed_mean": 0.15,
        "peak_z_mean": 0.8,
        "gait_alternating_touchdowns_per_100_steps_mean": 4.0,
    }

    result = quality_study._aggregate_walker_slip_recovery_results(
        {"control": control, "slip_cost_025": candidate}
    )

    assert result["objective_winner"] == "slip_cost_025"
    control_row = next(
        row for row in result["rankings"] if row["variant"] == "control"
    )
    assert control_row["automatic_exploit_flags"]["sliding"] is True
    assert control_row["metrics_gate_passed"] is False


def test_walker_slip_recovery_continuation_keeps_final_target_fixed() -> None:
    cfg = quality_study._slip_recovery_base_cfg(
        Path("src/rl_framework/configs/experiments"),
        1.25,
    )

    assert cfg["environment"]["reward"]["target_velocity"] == 1.0
    assert cfg["environment"]["reward"]["stance_slip_penalty_clip"] == 1.25
    assert "velocity_target_ramp" not in cfg["training"]


def test_walker_action_memory_dry_run_reports_matched_new_lineages(
    tmp_path: Path,
) -> None:
    result = run_quality_study(
        "walker-action-memory",
        seeds=[21, 22, 23],
        output_dir=tmp_path,
        dry_run=True,
    )

    assert result["planned_runs"]["walker_action_memory"] == 2
    assert "source_dir" not in result["identity"]
    assert list(tmp_path.iterdir()) == []


def test_walker_action_memory_base_changes_only_observation_between_arms() -> None:
    base = quality_study._action_memory_base_cfg(
        Path("src/rl_framework/configs/experiments")
    )
    control = quality_study.deepcopy(base)
    candidate = quality_study.deepcopy(base)
    quality_study._apply_overrides(
        control, quality_study.WALKER_ACTION_MEMORY_VARIANTS["control"]
    )
    quality_study._apply_overrides(
        candidate, quality_study.WALKER_ACTION_MEMORY_VARIANTS["previous_action"]
    )

    assert control["environment"]["observation"]["include_previous_action"] is False
    assert candidate["environment"]["observation"]["include_previous_action"] is True
    candidate["environment"]["observation"]["include_previous_action"] = False
    assert candidate == control
    assert base["environment"]["reward"]["stance_slip_penalty_weight"] == 0.5
    assert base["environment"]["reward"]["stance_slip_penalty_clip"] == pytest.approx(
        1.3398472589562194
    )


def test_walker_action_memory_gate_rejects_chatter_and_accepts_real_gait() -> None:
    passing = _diagnostic_result()
    passing["stochastic"].update(
        {
            "forward_displacement_mean": 8.0,
            "fall_rate": 0.2,
            "peak_z_mean": 0.7,
            "gait_contact_point_slip_speed_mean": 0.17,
        }
    )
    chattering = quality_study.deepcopy(passing)
    chattering["stochastic"][
        "gait_alternating_touchdowns_per_100_steps_mean"
    ] = 20.0

    assert quality_study._walker_action_memory_gate_result(passing)[
        "metrics_gate_passed"
    ]
    rejected = quality_study._walker_action_memory_gate_result(chattering)
    assert not rejected["metrics_gate_passed"]
    assert rejected["automatic_exploit_flags"]["contact_chatter"]

    aggregate = quality_study._aggregate_walker_action_memory_results(
        {"control": chattering, "previous_action": passing}
    )
    assert aggregate["objective_winner"] == "previous_action"


def test_walker_action_memory_smoke_stays_in_phase1_below_evidence_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_calls: list[dict] = []

    def fake_train(cfg, extra_callbacks=None, resume_from=None):
        assert resume_from is None
        train_calls.append(cfg)
        model = (
            Path(cfg["output"]["base_dir"])
            / cfg["experiment_name"]
            / f"seed_{cfg['seed']}"
            / "checkpoints"
            / "final_model"
        )
        model.parent.mkdir(parents=True, exist_ok=True)
        model.with_suffix(".zip").write_bytes(b"model")
        return model

    diagnostics = _diagnostic_result()
    diagnostics["stochastic"].update(
        {
            "forward_displacement_mean": 8.0,
            "fall_rate": 0.2,
            "peak_z_mean": 0.7,
            "gait_contact_point_slip_speed_mean": 0.17,
        }
    )
    monkeypatch.setattr(quality_study, "train", fake_train)
    monkeypatch.setattr(
        quality_study,
        "evaluate_walker_checkpoint",
        lambda cfg, path, episodes: quality_study.deepcopy(diagnostics),
    )
    monkeypatch.setattr(
        quality_study,
        "collect_walker_fall_recovery_telemetry",
        lambda cfg, path, episodes, render_dir: {"falls": []},
    )

    result = run_quality_study(
        "walker-action-memory",
        seeds=[21],
        output_dir=tmp_path,
        step_budget=100,
        eval_episodes=1,
    )

    assert result["completed"] is True
    assert len(train_calls) == 2
    assert {
        cfg["environment"]["observation"]["include_previous_action"]
        for cfg in train_calls
    } == {False, True}
    phase1 = result["results"]["walker_action_memory"]["phase1"]
    assert phase1["objective_winner"] == "previous_action"
    assert phase1["screening_ready"] is False
    assert result["results"]["walker_action_memory"]["phase2"]["status"] == "not_ready"


def test_walker_action_memory_confirms_three_seeds_before_transfer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_calls: list[dict] = []

    def fake_train(cfg, extra_callbacks=None, resume_from=None):
        train_calls.append(cfg)
        model = (
            Path(cfg["output"]["base_dir"])
            / cfg["experiment_name"]
            / f"seed_{cfg['seed']}"
            / "checkpoints"
            / "final_model"
        )
        model.parent.mkdir(parents=True, exist_ok=True)
        model.with_suffix(".zip").write_bytes(b"model")
        return model

    passing = _diagnostic_result()
    passing["stochastic"].update(
        {
            "forward_displacement_mean": 8.0,
            "fall_rate": 0.2,
            "peak_z_mean": 0.7,
            "gait_contact_point_slip_speed_mean": 0.17,
        }
    )
    monkeypatch.setattr(quality_study, "WALKER_ACTION_MEMORY_SCREEN_STEPS", 100)
    monkeypatch.setattr(quality_study, "train", fake_train)
    monkeypatch.setattr(
        quality_study,
        "evaluate_walker_checkpoint",
        lambda cfg, path, episodes: quality_study.deepcopy(passing),
    )
    monkeypatch.setattr(
        quality_study,
        "collect_walker_fall_recovery_telemetry",
        lambda cfg, path, episodes, render_dir: {"falls": []},
    )
    monkeypatch.setattr(
        quality_study,
        "evaluate_walker_transfer_suite",
        lambda cfg, path, episodes: {"flat": quality_study.deepcopy(passing)},
    )

    result = run_quality_study(
        "walker-action-memory",
        seeds=[21, 22, 23],
        output_dir=tmp_path,
        step_budget=100,
        eval_episodes=20,
        approved_variant="previous_action",
    )

    study = result["results"]["walker_action_memory"]
    assert len(train_calls) == 4
    assert study["phase1"]["visual_approval"]["accepted"] is True
    assert study["phase2"]["status"] == "transfer_completed"
    assert study["phase2"]["per_seed_gate"] == {
        "seed_21": True,
        "seed_22": True,
        "seed_23": True,
    }
    assert set(study["phase2"]["transfer"]) == {"seed_21", "seed_22", "seed_23"}


def test_wall_clock_callback_stops_after_budget(monkeypatch) -> None:
    times = iter([10.0, 10.5, 11.1])
    monkeypatch.setattr(quality_study.time, "perf_counter", lambda: next(times))
    callback = StopOnWallClock(1.0)

    callback._on_training_start()

    assert callback._on_step() is True
    assert callback._on_step() is False


def test_walker_verdict_detects_reward_hack() -> None:
    baseline = {"episode_length_mean": 100.0}
    deterministic = {
        "episode_length_mean": 800.0,
        "return_mean": 10.0,
        "peak_z_mean": 1.45,
        "forward_displacement_mean": 0.0,
    }
    stochastic = {**deterministic, "return_mean": 5.0}

    assert (
        _verdict(baseline, deterministic, stochastic, 800) == "reward_hack_high_peak_z"
    )


def test_walker_verdict_does_not_call_standing_drift_walking() -> None:
    baseline = {"episode_length_mean": 100.0}
    deterministic = {
        "episode_length_mean": 800.0,
        "return_mean": 100.0,
        "peak_z_mean": 0.7,
        "forward_displacement_mean": 0.1,
    }
    stochastic = {
        "episode_length_mean": 400.0,
        "return_mean": 50.0,
        "peak_z_mean": 0.7,
        "forward_displacement_mean": 0.5,
    }

    assert (
        _verdict(baseline, deterministic, stochastic, 800)
        == "partial_learning_deterministic"
    )


def test_arena_measurements_aggregate_resource_signals() -> None:
    measurements = quality_study._arena_measurements(
        [
            {
                "matches": [
                    {
                        "competitor": "baseline",
                        "opponent": "scarce",
                        "timeout_rate": 0.25,
                        "competitor_episode_metrics": {"food_pickups": 2.0},
                        "opponent_episode_metrics": {"food_pickups": 1.0},
                    }
                ]
            }
        ]
    )

    assert measurements["timeout_rate_mean"] == 0.25
    assert measurements["per_variant_timeout_rate"] == {
        "baseline": 0.25,
        "scarce": 0.25,
    }
    assert (
        measurements["per_variant_episode_metrics"]["baseline"]["food_pickups"] == 2.0
    )


def test_walker_behavior_gate_requires_robust_transfer() -> None:
    passing = _diagnostic_result()
    for mode in ("deterministic", "stochastic"):
        passing[mode] = {
            **passing[mode],
            "episode_length_mean": 800.0,
            "forward_displacement_mean": 4.0,
            "fall_rate": 0.0,
            "peak_z_mean": 0.8,
        }

    aggregate = quality_study._aggregate_walker_results(
        {
            "candidate/seed_0": {
                terrain: passing
                for terrain in ("flat", "uneven", "obstacles", "push_recovery")
            }
        }
    )

    assert aggregate["rankings"][0]["behavioral_gate_passed"] is True
    assert aggregate["promoted_variant"] == "candidate"


def test_walker_balance_gate_requires_stochastic_stability_on_flat() -> None:
    passing = _diagnostic_result()
    passing["deterministic"] = {
        **passing["deterministic"],
        "episode_length_mean": 100.0,
        "fall_rate": 1.0,
    }
    passing["stochastic"] = {
        **passing["stochastic"],
        "episode_length_mean": 650.0,
        "forward_displacement_mean": 0.2,
        "fall_rate": 0.2,
        "peak_z_mean": 0.8,
    }

    aggregate = quality_study._aggregate_walker_results(
        {"candidate/seed_0": {"flat": passing}}
    )

    assert aggregate["rankings"][0]["balance_gate_passed"] is True
    assert aggregate["balance_candidate"] == "candidate"
    assert aggregate["promoted_variant"] is None


def test_walker_balance_gate_rejects_deterministic_only_balance() -> None:
    failing = _diagnostic_result()
    failing["deterministic"] = {
        **failing["deterministic"],
        "episode_length_mean": 800.0,
        "forward_displacement_mean": 0.0,
        "fall_rate": 0.0,
    }
    failing["stochastic"] = {
        **failing["stochastic"],
        "episode_length_mean": 500.0,
        "forward_displacement_mean": 0.0,
        "fall_rate": 0.5,
    }

    aggregate = quality_study._aggregate_walker_results(
        {"candidate/seed_0": {"flat": failing}}
    )

    assert aggregate["rankings"][0]["balance_gate_passed"] is False
    assert aggregate["balance_candidate"] is None


def test_walker_velocity_gate_requires_every_seed_to_pass() -> None:
    passing = _diagnostic_result()
    passing["stochastic"] = {
        **passing["stochastic"],
        "episode_length_mean": 700.0,
        "forward_displacement_mean": 1.2,
        "fall_rate": 0.2,
        "peak_z_mean": 0.8,
    }
    failing = {
        **passing,
        "stochastic": {
            **passing["stochastic"],
            "forward_displacement_mean": 0.8,
        },
    }

    aggregate = quality_study._aggregate_walker_velocity_results(
        {
            "candidate/seed_0": passing,
            "candidate/seed_1": failing,
        }
    )

    assert aggregate["rankings"][0]["per_seed_passed"] == {
        "seed_0": True,
        "seed_1": False,
    }
    assert aggregate["rankings"][0]["gate_passed"] is False
    assert aggregate["promoted_variant"] is None


def test_walker_gait_gate_rejects_behavior_without_gait_structure() -> None:
    passing = _diagnostic_result()
    not_stepping = {
        **passing,
        "stochastic": {
            **passing["stochastic"],
            "gait_alternating_touchdowns_per_100_steps_mean": 0.5,
        },
    }

    aggregate = quality_study._aggregate_walker_gait_results(
        {
            "candidate/seed_0": passing,
            "candidate/seed_1": not_stepping,
        }
    )

    assert aggregate["rankings"][0]["per_seed_passed"] == {
        "seed_0": True,
        "seed_1": False,
    }
    assert aggregate["rankings"][0]["gate_passed"] is False
    assert aggregate["promoted_variant"] is None


def test_walker_gait_gate_v2_support_and_slip_contract_is_frozen() -> None:
    assert quality_study.WALKER_GAIT_GATE["gate_version"] == 2
    assert quality_study.WALKER_GAIT_GATE["slip_metric"] == (
        "gait_contact_point_slip_speed_mean"
    )
    assert quality_study.WALKER_GAIT_GATE["max_slip_speed"] == 0.18
    assert quality_study.WALKER_GAIT_GATE["min_double_support_fraction"] == 0.10
    assert quality_study.WALKER_GAIT_GATE["max_flight_fraction"] == 0.10


def test_walker_gait_gate_rejects_contact_chatter() -> None:
    """The regime the pre-correction gate demanded must now fail.

    Twenty alternating touchdowns per 100 steps is ~11 Hz stepping with
    millimetre strides — the shuffle that passed every structural ablation
    while displacement stalled near 1-2 m.
    """
    chatter = _diagnostic_result()
    chatter["stochastic"] = {
        **chatter["stochastic"],
        "gait_alternating_touchdowns_per_100_steps_mean": 20.0,
        "gait_progress_per_alternating_touchdown_mean": 0.005,
        "gait_stride_length_mean": 0.011,
        "gait_foot_clearance_mean": 0.0007,
    }

    aggregate = quality_study._aggregate_walker_gait_results(
        {"candidate/seed_0": chatter}
    )

    assert aggregate["rankings"][0]["per_seed_passed"] == {"seed_0": False}
    assert aggregate["promoted_variant"] is None


def test_walker_gait_gate_rejects_short_strides_at_valid_cadence() -> None:
    """Correct cadence alone is not evidence of stepping."""
    mincing = _diagnostic_result()
    mincing["stochastic"] = {
        **mincing["stochastic"],
        "gait_stride_length_mean": 0.02,
        "gait_foot_clearance_mean": 0.001,
    }

    aggregate = quality_study._aggregate_walker_gait_results(
        {"candidate/seed_0": mincing}
    )

    assert aggregate["rankings"][0]["per_seed_passed"] == {"seed_0": False}


def test_walker_gait_gate_rejects_bound_support_occupancy() -> None:
    """A low-slip bound must not be certified as walking."""
    bound = _diagnostic_result()
    bound["stochastic"] = {
        **bound["stochastic"],
        "gait_contact_point_slip_speed_mean": 0.10,
        "gait_double_support_fraction_mean": 0.0,
        "gait_flight_fraction_mean": 0.55,
    }

    aggregate = quality_study._aggregate_walker_gait_results(
        {"candidate/seed_0": bound}
    )

    assert aggregate["rankings"][0]["per_seed_passed"] == {"seed_0": False}


def test_walker_gait_gate_uses_contact_point_slip() -> None:
    """Heel/toe pivoting at a planted contact must not count as sliding."""
    pivoting = _diagnostic_result()
    pivoting["stochastic"] = {
        **pivoting["stochastic"],
        "gait_stance_slip_speed_mean": 0.50,
        "gait_contact_point_slip_speed_mean": 0.17,
    }
    sliding = _diagnostic_result()
    sliding["stochastic"] = {
        **sliding["stochastic"],
        "gait_stance_slip_speed_mean": 0.10,
        "gait_contact_point_slip_speed_mean": 0.19,
    }

    aggregate = quality_study._aggregate_walker_gait_results(
        {
            "pivoting/seed_0": pivoting,
            "sliding/seed_0": sliding,
        }
    )
    by_variant = {row["variant"]: row for row in aggregate["rankings"]}

    assert by_variant["pivoting"]["gate_passed"] is True
    assert by_variant["sliding"]["gate_passed"] is False


def test_walker_gait_score_does_not_reward_raw_cadence() -> None:
    """A faster shuffle must not out-rank longer strides on score alone."""
    long_strides = _diagnostic_result()
    fast_shuffle = _diagnostic_result()
    fast_shuffle["stochastic"] = {
        **fast_shuffle["stochastic"],
        "gait_alternating_touchdowns_per_100_steps_mean": 8.0,
        "gait_stride_length_mean": 0.05,
        "gait_foot_clearance_mean": 0.005,
    }

    aggregate = quality_study._aggregate_walker_gait_results(
        {
            "strides/seed_0": long_strides,
            "shuffle/seed_0": fast_shuffle,
        }
    )

    by_variant = {row["variant"]: row for row in aggregate["rankings"]}
    assert by_variant["strides"]["score"] > by_variant["shuffle"]["score"]
    assert aggregate["rankings"][0]["variant"] == "strides"


def test_arena_behavior_gate_requires_hits_damage_and_fewer_timeouts() -> None:
    measurements = {
        "per_variant_timeout_rate": {"candidate": 0.5, "timeout": 1.0},
        "per_variant_episode_metrics": {
            "candidate": {"attack_hits": 2.0, "damage_dealt": 0.1},
            "timeout": {"attack_hits": 0.0, "damage_dealt": 0.0},
        },
    }

    results = quality_study._arena_behavior_results(
        measurements, {"candidate": {}, "timeout": {}}
    )

    assert results["candidate"]["passed"] is True
    assert results["timeout"]["passed"] is False


def test_arena_native_tournaments_use_each_variant_environment(monkeypatch) -> None:
    seen = []

    def fake_tournament(paths, cfg, **kwargs):
        placement = cfg["environment"]["resources"]["food_placement"]
        seen.append((paths, placement, cfg["seed"], kwargs))
        return {
            "competitors": [
                {"label": "final_model", "path": paths[0]},
                {"label": "random", "path": "random"},
            ],
            "standings": [
                {"competitor": "final_model"},
                {"competitor": "random"},
            ],
            "matches": [{"competitor": "final_model", "opponent": "random"}],
            "ratings": {"final_model": 1500.0, "random": 1500.0},
            "win_rate_matrix": {
                "final_model": {"final_model": None, "random": 0.5},
                "random": {"final_model": 0.5, "random": None},
            },
        }

    monkeypatch.setattr(quality_study, "run_tournament", fake_tournament)
    base = {
        "seed": 0,
        "environment": {
            "seed": 0,
            "battle_rules": {},
            "resources": {"food_placement": "uniform"},
        },
    }

    tournaments = quality_study._arena_native_tournaments(
        {
            "baseline": {},
            "contested": {"environment.resources.food_placement": "center"},
        },
        {"baseline": {3: "base.zip"}, "contested": {3: "center.zip"}},
        [3],
        base,
        4,
    )

    assert [(row[1], row[2]) for row in seen] == [("uniform", 3), ("center", 3)]
    assert [row["environment_variant"] for row in tournaments] == [
        "baseline",
        "contested",
    ]
