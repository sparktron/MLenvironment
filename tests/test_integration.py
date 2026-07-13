"""End-to-end integration tests: minimal real training + evaluation.

These tests spin up a short training run (≤300 timesteps) to verify the
full pipeline — env creation, SB3 PPO loop, checkpoint saving,
VecNormalize serialisation, and eval_metrics.csv writing — without mocking
the training stack.

They are slower than unit tests (~5-15 s each) but still fast enough for CI.
"""

from __future__ import annotations

from pathlib import Path


def _walker_cfg(
    tmp_path: Path, timesteps: int = 256, checkpoint_every: int = 128
) -> dict:
    return {
        "experiment_name": "integ_walker",
        "seed": 0,
        "output": {"base_dir": str(tmp_path)},
        "environment": {
            "type": "walker_bullet",
            "sim": {
                "gravity": -9.81,
                "mass": 3.0,
                "friction": 0.8,
                "max_force": 30.0,
                "arena_half_extent": 5.0,
            },
            "reward": {
                "alive_bonus": 1.0,
                "forward_velocity_weight": 1.0,
                "target_velocity": 0.5,
                "orientation_penalty_weight": 0.1,
                "torque_penalty_weight": 0.01,
            },
            "termination": {
                "min_height": -0.5,
                "max_tilt_radians": 1.5,
                "max_steps": 50,
            },
            "domain_randomization": {
                "mass_scale_range": [1.0, 1.0],
                "friction_range": [0.8, 0.8],
                "sensor_noise_std": 0.0,
                "action_latency_steps": 0,
            },
        },
        "training": {
            "total_timesteps": timesteps,
            "learning_rate": 3e-4,
            "n_steps": 64,
            "batch_size": 32,
            "num_envs": 1,
            "device": "cpu",
            "checkpoint_every": checkpoint_every,
            "normalize_observations": True,
        },
        "evaluation": {"episodes": 1},
    }


def test_walker_train_produces_model_and_vecnorm(tmp_path: Path) -> None:
    """train() writes a .zip checkpoint and vecnormalize.pkl to disk."""
    from rl_framework.training.sb3_runner import train

    cfg = _walker_cfg(tmp_path)
    model_path = train(cfg)

    zip_path = (
        Path(str(model_path) + ".zip")
        if not str(model_path).endswith(".zip")
        else Path(model_path)
    )
    assert zip_path.exists(), f"Model zip not found: {zip_path}"

    vecnorm_path = zip_path.with_name("vecnormalize.pkl")
    assert vecnorm_path.exists(), f"vecnormalize.pkl not found alongside {zip_path}"


def test_walker_best_model_writes_matching_vecnorm_sidecar(tmp_path: Path) -> None:
    """Best-model eval must save a model-specific normalizer at the same point
    as the selected checkpoint, so replay and resume never pair it with later
    running statistics."""
    from rl_framework.training.sb3_runner import train

    cfg = _walker_cfg(tmp_path, timesteps=256, checkpoint_every=128)
    cfg["evaluation"]["best_model"] = {
        "enabled": True,
        "eval_every": 64,
        "episodes": 1,
    }
    train(cfg)

    checkpoints = (
        Path(tmp_path)
        / cfg["experiment_name"]
        / f"seed_{cfg['seed']}"
        / "checkpoints"
    )
    assert (checkpoints / "best_model.zip").exists()
    assert (checkpoints / "best_model_vecnormalize.pkl").exists()


def _arena_cfg(tmp_path: Path, timesteps: int = 256) -> dict:
    return {
        "experiment_name": "integ_arena",
        "seed": 0,
        "output": {"base_dir": str(tmp_path)},
        "environment": {
            "type": "organism_arena_parallel",
            "sim": {"arena_half_extent": 1.0},
            "battle_rules": {
                "max_steps": 20,
                "damage": 0.2,
                "attack_range": 0.5,
                "cooldown_steps": 0,
            },
        },
        "training": {
            "total_timesteps": timesteps,
            "n_steps": 64,
            "batch_size": 64,
            "num_envs": 1,
            "device": "cpu",
            "normalize_observations": True,
        },
    }


def test_arena_n_agent_shared_policy_train(tmp_path: Path) -> None:
    """A 3-agent free-for-all trains end to end via the shared-policy SuperSuit
    path (constant agent set thanks to inert spectators)."""
    from rl_framework.training.sb3_runner import train

    cfg = _arena_cfg(tmp_path, timesteps=256)
    cfg["environment"]["num_agents"] = 3
    model_path = train(cfg)
    assert Path(str(model_path) + ".zip").exists()


def test_arena_train_runs_and_logs_metrics(tmp_path: Path) -> None:
    """Arena PPO training starts (regression for the SuperSuit seed bug) and the
    ArenaMetricsCallback writes arena/* scalars to TensorBoard."""
    import glob

    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    from rl_framework.training.sb3_runner import train

    cfg = _arena_cfg(tmp_path)
    model_path = train(cfg)

    zip_path = Path(str(model_path) + ".zip")
    assert zip_path.exists(), f"Arena model zip not found: {zip_path}"

    events = glob.glob(str(tmp_path / "**" / "events.out.tfevents.*"), recursive=True)
    assert events, "no TensorBoard event file written"
    acc = EventAccumulator(events[0])
    acc.Reload()
    arena_tags = [t for t in acc.Tags()["scalars"] if t.startswith("arena/")]
    assert "arena/episode_outcomes" in arena_tags, (
        f"arena metrics missing from TensorBoard; got {arena_tags}"
    )


def test_arena_selfplay_train_writes_league(tmp_path: Path) -> None:
    """Self-play arena training runs end to end and writes league snapshots that
    the env's LeagueSampler can consume (regression for the single-agent
    VecNormalize/uint8-dones crash and the cloudpickle-cloned-callback no-op)."""
    from rl_framework.training.sb3_runner import train

    cfg = _arena_cfg(tmp_path, timesteps=320)
    cfg["self_play"] = {"enabled": True, "snapshot_freq": 128, "max_league_size": 5}
    model_path = train(cfg)

    assert Path(str(model_path) + ".zip").exists()
    league = list(
        (
            Path(tmp_path)
            / cfg["experiment_name"]
            / f"seed_{cfg['seed']}"
            / "checkpoints"
            / "league"
        ).glob("selfplay_*.zip")
    )
    assert league, "self-play training wrote no league snapshots"


def test_arena_selfplay_monitor_csv_records_episode_outcome(tmp_path: Path) -> None:
    """The self-play Monitor CSV must gain an episode_outcome column — the
    comment above the Monitor() construction claimed info_keywords already
    did this, but the call was missing the argument, so the column never
    appeared."""
    import csv

    from rl_framework.training.sb3_runner import train

    cfg = _arena_cfg(tmp_path, timesteps=256)
    cfg["environment"]["battle_rules"]["max_steps"] = 15  # force multiple episodes
    cfg["self_play"] = {"enabled": True, "snapshot_freq": 128, "max_league_size": 5}
    train(cfg)

    logs_dir = Path(tmp_path) / cfg["experiment_name"] / f"seed_{cfg['seed']}" / "logs"
    # SB3's Monitor appends its own ".monitor.csv" suffix to any filename that
    # doesn't already end in "monitor.csv", so the file on disk is
    # "monitor_env0.csv.monitor.csv", not the literal "monitor_env0.csv".
    candidates = list(logs_dir.glob("monitor_env0.csv*"))
    assert candidates, f"no monitor csv found in {logs_dir}"
    monitor_csv = candidates[0]
    with monitor_csv.open(encoding="utf-8") as fh:
        next(fh)  # skip the '#{"t_start": ...}' header comment line
        rows = list(csv.DictReader(fh))
    assert rows, "no episodes recorded in the monitor csv"
    assert "episode_outcome" in rows[0], (
        "episode_outcome column missing from the Monitor CSV header"
    )
    assert any(row["episode_outcome"] for row in rows), (
        "episode_outcome column is present but empty on every row"
    )


def test_arena_selfplay_parallel_envs_train_and_propagate(tmp_path: Path) -> None:
    """Self-play arena training runs with num_envs > 1 (R2): it bypasses
    SuperSuit, uses SB3's native SubprocVecEnv, writes one Monitor CSV per
    worker, and reward annealing propagates to workers via env_method without
    crashing. Regression for the single-process arena cap."""
    from rl_framework.training.sb3_runner import train

    cfg = _arena_cfg(tmp_path, timesteps=512)
    cfg["training"]["num_envs"] = 2
    cfg["training"]["n_steps"] = 128
    cfg["self_play"] = {"enabled": True, "snapshot_freq": 256, "max_league_size": 5}
    cfg["reward_annealing"] = {"enabled": True, "anneal_steps": 512}

    model_path = train(cfg)
    assert Path(str(model_path) + ".zip").exists()

    base = Path(tmp_path) / cfg["experiment_name"] / f"seed_{cfg['seed']}"
    league = list((base / "checkpoints" / "league").glob("selfplay_*.zip"))
    assert league, "parallel self-play training wrote no league snapshots"
    # One Monitor CSV per worker proves the native vec-env path (not SuperSuit).
    monitors = list((base / "logs").glob("monitor_env*.csv"))
    assert len(monitors) == 2, f"expected one monitor csv per worker, got {monitors}"


def test_arena_render_replay_headless(tmp_path: Path) -> None:
    """Arena render-replay produces a GIF headlessly, both as a shared-policy
    replay and against an explicit opponent (R3c)."""
    from rl_framework.cli.main import _render_replay
    from rl_framework.training.sb3_runner import train

    cfg = _arena_cfg(tmp_path, timesteps=256)
    model_path = train(cfg)
    checkpoint = str(model_path) + ".zip"

    # Shared-policy replay (opponent mirrors the main policy).
    shared = _render_replay(cfg, checkpoint)
    assert shared["frames"] > 0
    assert shared["opponent"] == "self"
    assert Path(shared["saved_replay"]).exists()
    assert shared["saved_replay"].endswith("replay.gif")

    # Matchup vs a random opponent.
    versus = _render_replay(cfg, checkpoint, opponent_path="random")
    assert versus["frames"] > 0
    assert versus["opponent"] == "random"
    assert Path(versus["saved_replay"]).exists()


def test_arena_eval_on_trained_checkpoint(tmp_path: Path) -> None:
    """Train a real arena checkpoint, then run head-to-head eval vs random.

    Exercises load_frozen_policy on a real PPO checkpoint with sibling
    vecnormalize.pkl discovery and the full episode-driving loop."""
    from rl_framework.training.arena_eval import run_arena_eval
    from rl_framework.training.sb3_runner import train

    cfg = _arena_cfg(tmp_path, timesteps=256)
    model_path = train(cfg)
    checkpoint = str(model_path) + ".zip"

    result = run_arena_eval(checkpoint, "random", cfg, n_episodes=5, swap_roles=True)
    assert result["n_episodes"] == 10
    total = (
        result["policy_win_rate"]
        + result["opponent_win_rate"]
        + result["draw_rate"]
        + result["timeout_rate"]
    )
    assert abs(total - 1.0) < 1e-6


def test_arena_tournament_on_trained_checkpoint(tmp_path: Path) -> None:
    """Train one arena checkpoint, then run a round-robin tournament of it vs a
    random baseline through the real run_arena_eval path. Exercises competitor
    resolution, the Bradley-Terry rating, and JSON/markdown output end to end."""
    from rl_framework.training.arena_tournament import run_tournament
    from rl_framework.training.sb3_runner import train

    cfg = _arena_cfg(tmp_path, timesteps=256)
    model_path = train(cfg)
    checkpoint = str(model_path) + ".zip"

    json_out = tmp_path / "tourney.json"
    md_out = tmp_path / "tourney.md"
    result = run_tournament(
        [checkpoint],
        cfg,
        n_episodes=3,
        include_random=True,
        output_path=str(json_out),
        markdown_path=str(md_out),
    )
    assert len(result["competitors"]) == 2
    assert set(result["ratings"]) == {"final_model", "random"}
    assert [s["rank"] for s in result["standings"]] == [1, 2]
    assert json_out.exists() and md_out.exists()
    # Ratings are centred on the Elo zero-point.
    assert all(1000 < e < 2000 for e in result["ratings"].values())


def test_walker_eval_writes_metrics_csv(tmp_path: Path) -> None:
    """evaluate() appends a row to eval_metrics.csv and returns a metrics dict."""
    from rl_framework.training.eval_runner import evaluate
    from rl_framework.training.sb3_runner import train

    cfg = _walker_cfg(tmp_path)
    model_path = train(cfg)
    zip_path = (
        str(model_path) + ".zip"
        if not str(model_path).endswith(".zip")
        else str(model_path)
    )

    metrics = evaluate(cfg, zip_path)

    assert "mean_return" in metrics
    assert "std_return" in metrics
    assert isinstance(metrics["mean_return"], float)

    csv_path = (
        Path(tmp_path)
        / cfg["experiment_name"]
        / f"seed_{cfg['seed']}"
        / "logs"
        / "eval_metrics.csv"
    )
    assert csv_path.exists(), f"eval_metrics.csv not written: {csv_path}"


def test_walker_checkpoint_saved_at_interval(tmp_path: Path) -> None:
    """Intermediate checkpoints include model-specific VecNormalize sidecars."""
    from rl_framework.training.sb3_runner import train

    cfg = _walker_cfg(tmp_path, timesteps=256, checkpoint_every=64)
    model_path = train(cfg)

    ckpt_dir = (
        Path(tmp_path) / cfg["experiment_name"] / f"seed_{cfg['seed']}" / "checkpoints"
    )
    checkpoints = list(ckpt_dir.glob("*.zip"))
    assert len(checkpoints) >= 1, (
        f"Expected at least one intermediate checkpoint in {ckpt_dir}, found none.\n"
        f"Final model: {model_path}"
    )
    periodic = [path for path in checkpoints if path.name.startswith("ppo_model_")]
    assert periodic, f"Expected periodic checkpoints in {ckpt_dir}, found {checkpoints}"
    for checkpoint in periodic:
        sidecar = checkpoint.with_name(checkpoint.stem + "_vecnormalize.pkl")
        assert sidecar.exists(), f"VecNormalize sidecar missing for {checkpoint}"
