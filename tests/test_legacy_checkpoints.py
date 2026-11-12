"""Load the model and normalizer artifacts produced before the restructure."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import PPO

from rl_framework.core.registry import make_env
from rl_framework.training.eval_runner import evaluate
from rl_framework.training.sb3_runner import train
from rl_framework.training.self_play_env_wrapper import (
    LeagueSampler,
    load_frozen_policy,
    load_obs_normalizer,
)
from rl_framework.utils.checkpoint import vecnormalize_path_for_model

FIXTURES = Path(__file__).with_name("fixtures") / "legacy_checkpoints"
WALKER_RUN = FIXTURES / "walker"
WALKER_MODEL = WALKER_RUN / "checkpoints" / "final_model.zip"


def _walker_config(tmp_path):
    manifest = json.loads(
        (WALKER_RUN / "run_metadata.json").read_text(encoding="utf-8")
    )
    cfg = manifest["config"]
    cfg["output"]["base_dir"] = str(tmp_path)
    cfg["reproducibility"]["strict"] = True
    cfg["training"]["total_timesteps"] = 256
    return cfg


def test_evaluate_legacy_walker_checkpoint(tmp_path):
    cfg = _walker_config(tmp_path)
    metrics = evaluate(cfg, str(WALKER_MODEL))

    assert np.isfinite(metrics["mean_return"])
    assert metrics["mean_episode_length"] > 0
    assert metrics["terminated_rate"] + metrics["truncated_rate"] == pytest.approx(1)


def test_resume_legacy_walker_checkpoint(tmp_path):
    cfg = _walker_config(tmp_path)
    mismatched = copy.deepcopy(cfg)
    mismatched["training"]["gamma"] = 0.5
    with pytest.raises(
        RuntimeError, match=r"Resume provenance check failed.*\n.*training.gamma"
    ):
        train(mismatched, resume_from=WALKER_MODEL)

    resumed_path = train(cfg, resume_from=WALKER_MODEL)
    original = PPO.load(WALKER_MODEL, device="cpu")
    resumed = PPO.load(resumed_path, device="cpu")
    assert (
        resumed.num_timesteps
        == original.num_timesteps + cfg["training"]["total_timesteps"]
    )
    assert resumed.learning_rate(0.0) == pytest.approx(
        cfg["training"]["learning_rate_end"]
    )
    assert resumed.learning_rate(1.0) == pytest.approx(cfg["training"]["learning_rate"])
    original_stats = load_obs_normalizer(vecnormalize_path_for_model(WALKER_MODEL))
    resumed_stats = load_obs_normalizer(vecnormalize_path_for_model(resumed_path))
    assert original_stats is not None and resumed_stats is not None
    assert resumed_stats.obs_rms.count >= (
        original_stats.obs_rms.count + cfg["training"]["total_timesteps"]
    )


def test_load_legacy_arena_league_snapshot():
    league = FIXTURES / "arena"
    snapshot = league / "selfplay_1024.zip"
    env_cfg = {"type": "organism_arena_parallel"}
    arena = make_env(env_cfg["type"], env_cfg)
    try:
        obs, _ = arena.reset(seed=11)
        action_space = arena.action_space("agent_1")
        direct = load_frozen_policy(snapshot, action_space)
        sampled = LeagueSampler(league, seed=11).sample()
        assert sampled is not None

        raw_obs = obs["agent_1"]
        action, _ = direct.predict(raw_obs, deterministic=True)
        league_action, _ = sampled.predict(raw_obs, deterministic=True)
        raw_model = PPO.load(snapshot, device="cpu")
        unnormalized_action, _ = raw_model.predict(raw_obs, deterministic=True)

        assert action_space.contains(action)
        np.testing.assert_allclose(league_action, action, rtol=0, atol=1e-6)
        # Both public loaders must use the shipped per-snapshot sidecar.
        assert not np.allclose(action, unnormalized_action, rtol=0, atol=1e-6)
    finally:
        arena.close()
