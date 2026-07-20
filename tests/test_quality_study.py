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
        "walker": 15,
        "arena": 18,
        "algorithms": 18,
    }
    assert list(tmp_path.iterdir()) == []


def test_walker_quality_study_persists_results_and_is_resumable(
    tmp_path: Path, monkeypatch
) -> None:
    train_calls = []

    def fake_train(cfg, extra_callbacks=None):
        train_calls.append(cfg["experiment_name"])
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
        step_budget=64,
        eval_episodes=1,
    )
    assert first["completed"] is True
    assert len(train_calls) == len(quality_study.WALKER_VARIANTS)
    assert first["results"]["walker"]["promotion_ready"] is False
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()

    resumed = run_quality_study(
        "walker",
        seeds=[0],
        output_dir=tmp_path,
        step_budget=64,
        eval_episodes=1,
        resume=True,
    )
    assert resumed["completed"] is True
    assert len(train_calls) == len(quality_study.WALKER_VARIANTS)


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
    assert measurements["per_variant_episode_metrics"]["baseline"]["food_pickups"] == 2.0


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
