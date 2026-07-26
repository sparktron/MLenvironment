from __future__ import annotations

from pathlib import Path

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
        "gait_alternating_touchdowns_per_100_steps_mean": 18.0,
        "gait_progress_per_alternating_touchdown_mean": 0.004,
        "gait_stance_slip_speed_mean": 0.15,
        "gait_flight_fraction_mean": 0.18,
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
        cfg["experiment_name"].removeprefix("quality_walker_")
        for cfg in train_calls
    } == set(quality_study.WALKER_VARIANTS)
    for cfg in train_calls:
        assert {
            key: cfg["training"][key]
            for key in quality_study.WALKER_STUDY_TRAINING
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

    assert _verdict(baseline, deterministic, stochastic, 800) == "reward_hack_high_peak_z"


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
    assert measurements["per_variant_episode_metrics"]["baseline"]["food_pickups"] == 2.0


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
    weak_gait = {
        **passing,
        "stochastic": {
            **passing["stochastic"],
            "gait_alternating_touchdowns_per_100_steps_mean": 5.0,
        },
    }

    aggregate = quality_study._aggregate_walker_gait_results(
        {
            "candidate/seed_0": passing,
            "candidate/seed_1": weak_gait,
        }
    )

    assert aggregate["rankings"][0]["per_seed_passed"] == {
        "seed_0": True,
        "seed_1": False,
    }
    assert aggregate["rankings"][0]["gate_passed"] is False
    assert aggregate["promoted_variant"] is None


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
            "matches": [
                {"competitor": "final_model", "opponent": "random"}
            ],
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
        {"baseline": {}, "contested": {"environment.resources.food_placement": "center"}},
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
