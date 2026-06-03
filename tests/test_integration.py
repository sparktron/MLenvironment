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
                "body_half_extents": [0.15, 0.15, 0.15],
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
        result["policy_win_rate"] + result["opponent_win_rate"] + result["timeout_rate"]
    )
    assert abs(total - 1.0) < 1e-6


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
    """At least one intermediate checkpoint is written when checkpoint_every fires."""
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
