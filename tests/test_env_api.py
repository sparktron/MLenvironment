import numpy as np
import pybullet as p

from rl_framework.envs.registry import make_env


def test_walker_env_api() -> None:
    cfg = {
        "type": "walker_bullet",
        "seed": 1,
        "sim": {"gravity": -9.81, "mass": 3.0, "friction": 0.8, "max_force": 30.0},
    }
    env = make_env("walker_bullet", cfg)
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()


def test_walker_domain_randomization_preserves_link_mass_ratios() -> None:
    """Mass randomization scales each link from its own nominal mass."""
    cfg = {
        "type": "walker_bullet",
        "seed": 1,
        "sim": {"gravity": -9.81, "mass": 28.0, "friction": 0.8, "max_force": 30.0},
        "domain_randomization": {
            "mass_scale_range": [1.0, 1.0],
            "friction_range": [1.0, 1.0],
        },
    }
    env = make_env("walker_bullet", cfg)
    try:
        obs, _ = env.reset(seed=1)
        assert obs[33] == 1.0
        masses = {
            link_id: p.getDynamicsInfo(
                env.robot_id, link_id, physicsClientId=env._connection
            )[0]
            for link_id in (-1, 0, 2, 5, 10)
        }
        assert masses[-1] == 28.0
        assert masses[0] == 7.0
        assert masses[2] == 2.0
        assert masses[5] == 2.0
        assert masses[10] == 1.5
    finally:
        env.close()


def test_organism_env_api() -> None:
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 1,
        "sim": {"arena_half_extent": 1.0},
    }
    env = make_env("organism_arena_parallel", cfg)
    observations, infos = env.reset(seed=1)
    assert "agent_0" in observations and "agent_1" in observations
    actions = {agent: env.action_space(agent).sample() for agent in observations.keys()}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    assert set(rewards.keys()) == {"agent_0", "agent_1"}
    assert set(terminations.keys()) == {"agent_0", "agent_1"}
    assert set(truncations.keys()) == {"agent_0", "agent_1"}


def test_organism_env_render_rgb_array() -> None:
    """Smoke test: rgb_array render returns HxWx3 uint8 frames."""
    cfg = {"type": "organism_arena_parallel", "seed": 1, "render_mode": "rgb_array"}
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=1)
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
    env.close()


def test_organism_env_obs_shape() -> None:
    cfg = {"type": "organism_arena_parallel", "seed": 1}
    env = make_env("organism_arena_parallel", cfg)
    observations, _ = env.reset(seed=1)
    for agent, obs in observations.items():
        assert obs.shape == env.observation_space(agent).shape, (
            f"{agent} obs shape mismatch"
        )


def test_organism_env_terminates_when_health_depleted() -> None:
    """An agent whose health reaches 0 should be marked terminated."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {
            "damage": 10.0,
            "attack_range": 5.0,
            "cooldown_steps": 0,
            "max_steps": 400,
        },
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    # Force agent_0's health to near zero; only agent_1 attacks.
    env.state["agent_0"]["health"] = 0.001
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    no_attack = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    attack = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    _, rewards, terminations, _, _ = env.step({"agent_0": no_attack, "agent_1": attack})
    assert terminations["agent_0"], (
        "agent_0 should be terminated after health reaches 0"
    )
    assert rewards["agent_1"] > 0, "winner (agent_1) should receive a positive reward"
    assert env.agents == [], "agents list should be cleared after termination"


def test_organism_env_truncates_at_max_steps() -> None:
    """Episode should truncate (not terminate) when max_steps is reached."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"damage": 0.0, "max_steps": 3},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    noop = np.zeros(3, dtype=np.float32)
    for step in range(3):
        _, _, terminations, truncations, _ = env.step(
            {"agent_0": noop, "agent_1": noop}
        )
    # On the final step both agents should be truncated, not terminated.
    assert all(truncations.values()), "all agents should be truncated at max_steps"
    assert not any(terminations.values()), "no agent should be terminated on truncation"
    assert env.agents == [], "agents list should be cleared after truncation"


def test_organism_env_agents_cleared_after_game_over() -> None:
    """self.agents must be empty after any terminal/truncated step."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"max_steps": 1},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    noop = np.zeros(3, dtype=np.float32)
    env.step({"agent_0": noop, "agent_1": noop})
    assert env.agents == []


def test_organism_env_obs_reports_self_velocity() -> None:
    """Feature 4: the first two obs components are displacement since last step."""
    cfg = {"type": "organism_arena_parallel", "seed": 0}
    env = make_env("organism_arena_parallel", cfg)
    observations, _ = env.reset(seed=0)
    # Velocity reads zero on the first observation after reset.
    assert observations["agent_0"][0] == 0.0 and observations["agent_0"][1] == 0.0
    # Move agent_0 at full speed: velocity obs is in units of move_speed.
    move = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    observations, _, _, _, _ = env.step({"agent_0": move, "agent_1": noop})
    assert abs(observations["agent_0"][0] - 1.0) < 1e-5, (
        "vel_x should be +1.0 (full speed in move_speed units)"
    )
    assert abs(observations["agent_0"][1]) < 1e-6, "vel_y should be 0"


def test_organism_env_obs_hides_opponent_beyond_sensing_radius() -> None:
    """Feature 4: opponent relative pos/health zeroed past sensing_radius."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"sensing_radius": 0.5},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array([5.0, 0.0], dtype=np.float32)
    obs = env._obs("agent_0")
    # rel_opp_x, rel_opp_y, opp_health are obs[3:6]; obs[7] is the visibility flag.
    assert obs[3] == 0.0 and obs[4] == 0.0 and obs[5] == 0.0
    assert obs[7] == 0.0, "visibility flag should be 0 when out of sensing range"


def test_organism_env_attack_damage_scales_with_distance() -> None:
    """Feature 6: linear falloff deals ~50% damage at half attack_range."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"damage": 0.1, "attack_range": 0.4, "cooldown_steps": 0},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array([0.2, 0.0], dtype=np.float32)  # half range
    env.state["agent_0"]["cooldown"] = 0
    attack = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    _, rewards, _, _, _ = env.step({"agent_0": attack, "agent_1": noop})
    expected = 0.1 * 0.5  # linear falloff at half range, size == 1.0
    assert abs(rewards["agent_0"] - expected) < 0.01


def test_organism_env_attack_zero_damage_beyond_range() -> None:
    """Feature 6: no damage when the defender sits past attack_range."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"damage": 0.1, "attack_range": 0.2, "cooldown_steps": 0},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array([0.3, 0.0], dtype=np.float32)
    env.state["agent_0"]["cooldown"] = 0
    attack = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    _, rewards, _, _, _ = env.step({"agent_0": attack, "agent_1": noop})
    assert rewards["agent_0"] == 0.0


def test_organism_env_attack_binary_mode_is_a_cliff() -> None:
    """Feature 6: binary mode deals full damage inside range, none outside."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {
            "damage": 0.1,
            "attack_range": 0.2,
            "cooldown_steps": 0,
            "attack_falloff": "binary",
        },
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array([0.19, 0.0], dtype=np.float32)  # inside
    env.state["agent_0"]["cooldown"] = 0
    attack = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    _, rewards, _, _, _ = env.step({"agent_0": attack, "agent_1": noop})
    assert abs(rewards["agent_0"] - 0.1) < 1e-6, "full damage inside range"


def test_organism_env_infos_episode_outcome_on_ko() -> None:
    """Feature 2: terminal step annotates infos with a 'ko' episode_outcome."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"damage": 10.0, "attack_range": 5.0, "cooldown_steps": 0},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    env.state["agent_1"]["health"] = 0.001
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    attack = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    _, _, terminations, _, infos = env.step({"agent_0": attack, "agent_1": noop})
    assert any(terminations.values()), "episode should terminate on KO"
    for info in infos.values():
        assert "episode_outcome" in info
        assert info["episode_outcome"]["outcome"] == "ko"
        assert info["episode_outcome"]["winner"] == "agent_0"
        assert info["episode_outcome"]["loser"] == "agent_1"


def test_organism_env_infos_episode_outcome_on_timeout() -> None:
    """Feature 2: truncation annotates infos with a 'timeout' episode_outcome."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"damage": 0.0, "max_steps": 1},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    _, _, _, truncations, infos = env.step({"agent_0": noop, "agent_1": noop})
    assert all(truncations.values())
    for info in infos.values():
        assert info["episode_outcome"]["outcome"] == "timeout"
        assert info["episode_outcome"]["winner"] is None


def test_arena_metrics_callback_computes_win_rates() -> None:
    """ArenaMetricsCallback aggregates outcomes and records win rates."""
    from rl_framework.training.sb3_runner import ArenaMetricsCallback

    class _StubLogger:
        def __init__(self) -> None:
            self.records: dict[str, float] = {}

        def record(self, key: str, value: float) -> None:
            self.records[key] = value

    from types import SimpleNamespace

    cb = ArenaMetricsCallback()
    cb.model = SimpleNamespace(logger=_StubLogger())
    # Three episodes: agent_0 KO win, agent_1 KO win, a timeout.
    cb.locals = {
        "infos": [
            {"episode_outcome": {"outcome": "ko", "winner": "agent_0"}},
            {"episode_outcome": {"outcome": "ko", "winner": "agent_1"}},
            {"episode_outcome": {"outcome": "timeout", "winner": None}},
            {"step": 5},  # non-terminal info — ignored
        ]
    }
    cb._on_step()
    cb._on_rollout_end()
    recs = cb.logger.records
    assert abs(recs["arena/agent_0_win_rate"] - 1 / 3) < 1e-9
    assert abs(recs["arena/agent_1_win_rate"] - 1 / 3) < 1e-9
    assert abs(recs["arena/timeout_rate"] - 1 / 3) < 1e-9
    assert recs["arena/episode_outcomes"] == 3


def test_update_live_params_anneals_dense_reward_but_keeps_damage() -> None:
    """Feature 5A: damage_scale=0 zeroes the dense reward but combat still resolves."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"damage": 0.1, "attack_range": 0.4, "cooldown_steps": 0},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    env.update_live_params({"reward.damage_scale": 0.0})
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_0"]["cooldown"] = 0
    health_before = env.state["agent_1"]["health"]
    attack = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    _, rewards, _, _, _ = env.step({"agent_0": attack, "agent_1": noop})
    assert rewards["agent_0"] == 0.0, "dense reward should be zero at damage_scale=0"
    assert env.state["agent_1"]["health"] < health_before, "health must still drop"


def test_update_live_params_updates_battle_rules() -> None:
    """Feature 5B: curriculum overrides mutate BattleRules in place, type-coerced."""
    cfg = {"type": "organism_arena_parallel", "seed": 0}
    env = make_env("organism_arena_parallel", cfg)
    original = env.rules.cooldown_steps
    env.update_live_params({"battle_rules.cooldown_steps": original + 2})
    assert env.rules.cooldown_steps == original + 2
    assert isinstance(env.rules.cooldown_steps, int)


def test_arena_live_params_update_morphology_and_bounds() -> None:
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "sim": {"arena_half_extent": 1.0},
        "morphology": {"base_size": 1.0},
    }
    env = make_env("organism_arena_parallel", cfg)

    env.update_live_params({"morphology.base_size": 1.4, "sim.arena_half_extent": 2.5})

    assert env.morphology["base_size"] == 1.4
    assert env.cfg["morphology"]["base_size"] == 1.4
    assert env.bounds == 2.5
    assert env.cfg["sim"]["arena_half_extent"] == 2.5


def test_update_live_params_ignores_unknown_keys() -> None:
    cfg = {"type": "organism_arena_parallel", "seed": 0}
    env = make_env("organism_arena_parallel", cfg)
    # Unknown keys (e.g. from a shared walker curriculum) must be silently ignored.
    env.update_live_params({"reward.target_velocity": 2.0, "battle_rules.bogus": 1})
    assert not hasattr(env.rules, "bogus")


def test_unknown_battle_rules_key_warns() -> None:
    import pytest

    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"dammage": 0.1, "max_steps": 10},
    }
    with pytest.warns(UserWarning, match="dammage"):
        env = make_env("organism_arena_parallel", cfg)
    # Valid keys still apply despite the typo'd sibling.
    assert env.rules.max_steps == 10


def test_arena_step_after_episode_end_is_inert() -> None:
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "battle_rules": {"max_steps": 2},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    idle = {a: np.zeros(3, dtype=np.float32) for a in env.possible_agents}
    env.step(idle)
    env.step(idle)  # hits max_steps -> truncation, agents emptied
    assert env.agents == []
    obs, rewards, terms, truncs, infos = env.step({})
    # No fabricated timeout outcome, no phantom agents.
    assert obs == {} and rewards == {} and terms == {} and truncs == {} and infos == {}


def test_arena_observe_matches_observation_contract() -> None:
    cfg = {"type": "organism_arena_parallel", "seed": 0}
    env = make_env("organism_arena_parallel", cfg)
    reset_obs, _ = env.reset(seed=0)
    obs = env.observe("agent_0")
    assert obs.shape == env.observation_space("agent_0").shape
    assert np.array_equal(obs, reset_obs["agent_0"])


def test_arena_spawn_jitter_differs_across_seeds() -> None:
    """Different reset seeds must yield different spawn positions (B1)."""
    cfg = {"type": "organism_arena_parallel", "seed": 0}
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    pos_a = {a: env.state[a]["pos"].copy() for a in env.possible_agents}
    env.reset(seed=1)
    pos_b = {a: env.state[a]["pos"].copy() for a in env.possible_agents}
    assert any(not np.array_equal(pos_a[a], pos_b[a]) for a in env.possible_agents), (
        "spawn positions should differ across seeds"
    )
    # Same seed reproduces the same spawn.
    env.reset(seed=1)
    pos_c = {a: env.state[a]["pos"].copy() for a in env.possible_agents}
    assert all(np.array_equal(pos_b[a], pos_c[a]) for a in env.possible_agents)


def test_arena_spawn_jitter_zero_restores_fixed_spawn() -> None:
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "sim": {"spawn_jitter": 0.0},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    assert np.allclose(env.state["agent_0"]["pos"], [-0.6, 0.0])
    assert np.allclose(env.state["agent_1"]["pos"], [0.6, 0.0])


def test_arena_diagonal_movement_not_faster() -> None:
    """Movement is clamped by norm: diagonal speed equals axis-aligned (B3)."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "sim": {"spawn_jitter": 0.0},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    start = env.state["agent_0"]["pos"].copy()
    diagonal = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    noop = np.zeros(3, dtype=np.float32)
    env.step({"agent_0": diagonal, "agent_1": noop})
    displacement = float(np.linalg.norm(env.state["agent_0"]["pos"] - start))
    assert abs(displacement - env.move_speed) < 1e-6, (
        "diagonal displacement should equal move_speed, not move_speed * sqrt(2)"
    )


def test_arena_obs_normalized_and_visibility_disambiguates() -> None:
    """Obs components are fraction-scaled, and the visibility flag separates
    'out of range' from 'visible with near-zero health' (B4/B10)."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "morphology": {"health": 3.0},  # raw health != 1 to catch raw leakage
        "battle_rules": {"sensing_radius": 0.5, "cooldown_steps": 4},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    env.state["agent_0"]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
    env.state["agent_1"]["pos"] = np.array([0.1, 0.0], dtype=np.float32)
    env.state["agent_1"]["health"] = 0.0
    env.state["agent_0"]["cooldown"] = 2

    obs = env.observe("agent_0")
    assert obs[2] == 1.0, "own health is a fraction of max, not raw health"
    assert obs[5] == 0.0 and obs[7] == 1.0, (
        "visible near-dead opponent: health 0 but visibility flag 1"
    )
    assert abs(obs[6] - 0.5) < 1e-6, "cooldown is a fraction of cooldown_steps"

    env.state["agent_1"]["pos"] = np.array([5.0, 0.0], dtype=np.float32)
    obs = env.observe("agent_0")
    assert obs[5] == 0.0 and obs[7] == 0.0, (
        "out-of-range opponent: zero block plus visibility flag 0"
    )
