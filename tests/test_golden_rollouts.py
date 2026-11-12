"""Replay the pre-restructure environment traces without generating new actions."""

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from rl_framework.core.registry import make_env
from rl_framework.training.self_play_env_wrapper import (
    LeagueSampler,
    SelfPlayEnvWrapper,
    SingleAgentArenaEnv,
)
from rl_framework.utils.config import load_config

CASES = {
    "walker_v9": "walker_reward_v9_flight_cost",
    "walker_v12": "walker_reward_v12_mixed_terrain",
    "walker_v1": "robot_walk_basic",
    "arena_parallel": "organisms_fight_arena",
    "arena_self_play": "organisms_fight_arena",
}
FIXTURES = Path(__file__).with_name("fixtures") / "golden"
CONFIG_DIR = Path("src/rl_framework/configs/experiments")
ATOL = 1e-6


def _make_case(case, seed, empty_league):
    config = load_config(CASES[case], CONFIG_DIR)
    env_cfg = OmegaConf.to_container(config.environment, resolve=True)
    arena_or_walker = make_env(env_cfg["type"], env_cfg)
    if case != "arena_self_play":
        return arena_or_walker

    # Arena action_space() creates a new Box each time. Give the empty-league
    # fallback one seeded opponent Box so it still executes its real sample().
    original_action_space = arena_or_walker.action_space
    opponent_space = original_action_space("agent_1")
    opponent_space.seed(seed)

    def action_space(agent):
        return opponent_space if agent == "agent_1" else original_action_space(agent)

    arena_or_walker.action_space = action_space
    sampler = LeagueSampler(empty_league, seed=seed)
    return SingleAgentArenaEnv(SelfPlayEnvWrapper(arena_or_walker, sampler))


def _record_obs(obs, infos, parallel):
    if not parallel:
        obs, infos = {"agent_0": obs}, {"agent_0": infos}
    scalar_infos = {}
    for agent, info in infos.items():
        scalar_infos[agent] = {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in info.items()
            if isinstance(value, (int, float, np.bool_, np.integer, np.floating))
            or key == "terrain_preset"
        }
    return {
        "observations": {agent: value.tolist() for agent, value in obs.items()},
        "infos": scalar_infos,
    }


def _rollout(env, actions, seed, *, parallel):
    obs, infos = env.reset(seed=seed)
    resets = {"0": _record_obs(obs, infos, parallel)}
    steps = []
    for index, action in enumerate(actions, start=1):
        if parallel:
            action = {
                agent: np.asarray(value, dtype=np.float32)
                for agent, value in action.items()
            }
        else:
            action = np.asarray(action, dtype=np.float32)
        obs, rewards, terminated, truncated, infos = env.step(action)
        step = _record_obs(obs, infos, parallel)
        step["rewards"] = rewards if parallel else {"agent_0": rewards}
        step["terminated"] = terminated if parallel else {"agent_0": terminated}
        step["truncated"] = truncated if parallel else {"agent_0": truncated}
        steps.append(step)
        done = not env.agents if parallel else terminated or truncated
        if done:
            # Let the seeded RNG advance across episodes so v12 exercises
            # terrain sampling and rebuilding with the same sequence.
            obs, infos = env.reset()
            resets[str(index)] = _record_obs(obs, infos, parallel)
    return steps, resets


def _assert_record(actual, expected):
    assert actual.keys() == expected.keys()
    for field in ("observations", "rewards"):
        if field in expected:
            assert actual[field].keys() == expected[field].keys()
            for agent, value in expected[field].items():
                np.testing.assert_allclose(
                    actual[field][agent],
                    value,
                    rtol=0,
                    atol=ATOL,
                    err_msg=f"{field}.{agent}",
                )
    for field in ("terminated", "truncated"):
        if field in expected:
            assert actual[field] == expected[field]
    assert actual["infos"].keys() == expected["infos"].keys()
    for agent, info in expected["infos"].items():
        actual_info = actual["infos"][agent]
        assert actual_info.keys() == info.keys()
        for key, value in info.items():
            if isinstance(value, float):
                np.testing.assert_allclose(
                    actual_info[key],
                    value,
                    rtol=0,
                    atol=ATOL,
                    err_msg=f"info.{agent}.{key}",
                )
            else:
                assert actual_info[key] == value, f"info.{agent}.{key}"


@pytest.mark.parametrize("case", CASES)
def test_golden_rollout(case, tmp_path):
    with gzip.open(FIXTURES / f"{case}.json.gz", "rt", encoding="utf-8") as fixture:
        golden = json.load(fixture)

    env = _make_case(case, golden["seed"], tmp_path)
    try:
        steps, resets = _rollout(
            env, golden["actions"], golden["seed"], parallel=case == "arena_parallel"
        )
    finally:
        env.close()

    assert len(steps) == len(golden["steps"])
    assert resets.keys() == golden["resets"].keys()
    assert len(resets) >= 3
    if case == "walker_v12":
        episode_terrains = {
            steps[int(start)]["infos"]["agent_0"]["terrain_preset"]
            for start in resets
            if int(start) < len(steps)
        }
        assert episode_terrains == {"flat", "uneven", "obstacles", "push_recovery"}
    for index, (actual, expected) in enumerate(
        zip(steps, golden["steps"], strict=True), 1
    ):
        try:
            _assert_record(actual, expected)
        except AssertionError as exc:
            raise AssertionError(f"{case} step {index}: {exc}") from exc
    for index, expected in golden["resets"].items():
        try:
            _assert_record(resets[index], expected)
        except AssertionError as exc:
            raise AssertionError(f"{case} reset after {index} steps: {exc}") from exc
