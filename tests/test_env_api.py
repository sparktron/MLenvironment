import numpy as np
import pybullet as p
import pytest

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


@pytest.mark.parametrize(
    ("observation", "shape"),
    [({"version": "v2"}, (37,)), ({"version": "v2", "coordinate_free": True}, (35,))],
)
def test_walker_observation_v2_adds_foot_contacts(observation, shape) -> None:
    env = make_env("walker_bullet", {"type": "walker_bullet", "seed": 1, "observation": observation})
    try:
        obs, _ = env.reset(seed=1)
        assert obs.shape == env.observation_space.shape == shape
        assert set(obs[-2:]) <= {0.0, 1.0}
    finally:
        env.close()


@pytest.mark.parametrize("preset,expected_bodies", [("flat", 0), ("uneven", 5), ("obstacles", 3)])
def test_walker_terrain_presets_build_static_geometry(preset: str, expected_bodies: int) -> None:
    env = make_env("walker_bullet", {"type": "walker_bullet", "seed": 1, "terrain": {"preset": preset}})
    try:
        env.reset(seed=1)
        assert len(env._terrain_body_ids) == expected_bodies
    finally:
        env.close()


def test_walker_push_recovery_publishes_push_event() -> None:
    env = make_env(
        "walker_bullet",
        {"type": "walker_bullet", "seed": 1, "terrain": {"preset": "push_recovery", "push_recovery": {"interval_steps": 1, "force": 100.0}}},
    )
    try:
        env.reset(seed=1)
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert info["push_applied"] is True
    finally:
        env.close()


def test_walker_construction_tolerates_null_sections() -> None:
    """The GUI wizard has historically written `section: null` for empty
    nested groups. Construction must not crash on it."""
    cfg = {
        "type": "walker_bullet",
        "seed": 1,
        "sim": None,
        "reward": None,
        "termination": None,
        "domain_randomization": None,
        "reset_randomization": None,
    }
    env = make_env("walker_bullet", cfg)
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    env.close()


def test_walker_update_live_params_tolerates_null_sections() -> None:
    """update_live_params must not crash when the target section exists in
    cfg as an explicit null (setdefault leaves an existing None untouched,
    so a naive fix would still crash on the subsequent item assignment)."""
    cfg = {
        "type": "walker_bullet",
        "seed": 1,
        "reward": None,
        "termination": None,
        "domain_randomization": None,
        "sim": None,
    }
    env = make_env("walker_bullet", cfg)
    try:
        env.reset(seed=1)
        env.update_live_params(
            {
                "reward.alive_bonus": 2.0,
                "termination.max_steps": 500,
                "domain_randomization.sensor_noise_std": 0.05,
                "sim.gravity": -10.0,
            }
        )
        assert env.reward_fn.alive_bonus == 2.0
        assert env.termination.max_steps == 500
        assert env.cfg["domain_randomization"]["sensor_noise_std"] == 0.05
        assert env.cfg["sim"]["gravity"] == -10.0
    finally:
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


@pytest.mark.parametrize(
    "termination_cfg",
    [
        {"min_height": 999.0, "max_height": 1000.0, "max_steps": 100},
        {"min_height": 0.0, "max_height": 0.1, "max_steps": 100},
    ],
)
def test_walker_terminal_fallbacks_penalize_clipped_action(termination_cfg) -> None:
    """Low-height and max-height terminal paths should count as falls."""
    cfg = {
        "type": "walker_bullet",
        "seed": 1,
        "sim": {"gravity": -9.81, "mass": 3.0, "friction": 0.8, "max_force": 30.0},
        "reward": {
            "alive_bonus": 0.0,
            "forward_velocity_weight": 0.0,
            "orientation_penalty_weight": 0.0,
            "torque_penalty_weight": 1.0,
            "fall_penalty": 7.0,
        },
        "termination": termination_cfg,
    }
    env = make_env("walker_bullet", cfg)
    captured = {}
    original_compute = env.reward_fn.compute

    def _capture_compute(**kwargs):
        captured.update(kwargs)
        return original_compute(**kwargs)

    env.reward_fn.compute = _capture_compute
    try:
        env.reset(seed=1)
        action = np.full(env.action_space.shape, 2.0, dtype=np.float32)
        _, reward, terminated, _, _ = env.step(action)
        assert terminated
        assert captured["fell"] is True
        np.testing.assert_allclose(captured["action"], np.ones_like(action))
        assert reward == -17.0
    finally:
        env.close()


def test_walker_diverged_physics_terminates_with_finite_reward() -> None:
    """A NaN/Inf PyBullet velocity read must not leak into reward/termination.

    Regression test for the NaN-action-mean bug: previously step() used the
    raw (un-sanitized) lin_vel/pos/quat for reward and termination while only
    the observation had a nan_to_num guard, so a solver-divergence frame
    produced a NaN reward that survived (uncaught) until max_steps
    truncation, poisoning GAE for that episode and NaN-ing the next PPO
    gradient update.
    """
    cfg = {
        "type": "walker_bullet",
        "seed": 1,
        "sim": {"gravity": -9.81, "mass": 3.0, "friction": 0.8, "max_force": 30.0},
    }
    env = make_env("walker_bullet", cfg)
    try:
        env.reset(seed=1)
        real_get_base_velocity = p.getBaseVelocity

        def _diverged_velocity(*args, **kwargs):
            return ([float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0])

        p.getBaseVelocity = _diverged_velocity
        try:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
        finally:
            p.getBaseVelocity = real_get_base_velocity

        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert terminated is True
        assert info["torso_contact"] is True
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


def test_arena_collision_separates_overlapping_organisms() -> None:
    env = make_env(
        "organism_arena_parallel",
        {"type": "organism_arena_parallel", "seed": 0, "sim": {"collision_radius": 0.1}},
    )
    try:
        env.reset(seed=0)
        for agent in env.agents:
            env.state[agent]["pos"] = np.zeros(2, dtype=np.float32)
        noop = np.zeros(3, dtype=np.float32)
        env.step({agent: noop for agent in env.agents})
        distance = np.linalg.norm(env.state["agent_0"]["pos"] - env.state["agent_1"]["pos"])
        assert distance >= 0.2 - 1e-6
    finally:
        env.close()


def test_arena_food_restores_energy_and_respawns() -> None:
    env = make_env(
        "organism_arena_parallel",
        {"type": "organism_arena_parallel", "seed": 0, "resources": {"food_count": 1, "food_energy": 0.4, "food_respawn_steps": 1}},
    )
    try:
        env.reset(seed=0)
        env.state["agent_0"]["energy"] = 0.2
        env._food[0]["pos"] = env.state["agent_0"]["pos"].copy()
        noop = np.zeros(3, dtype=np.float32)
        env.step({agent: noop for agent in env.agents})
        assert env.state["agent_0"]["energy"] == pytest.approx(0.6)
        assert env._food[0]["respawn_at"] == 1 + env.resources.food_respawn_steps
        env.step({agent: noop for agent in env.agents})
        assert env._food[0]["respawn_at"] is None
    finally:
        env.close()


def test_arena_larger_size_moves_more_slowly() -> None:
    env = make_env(
        "organism_arena_parallel",
        {"type": "organism_arena_parallel", "seed": 0, "morphology": {"base_size": 2.0}},
    )
    try:
        env.reset(seed=0)
        start = env.state["agent_0"]["pos"].copy()
        env.step({"agent_0": np.array([1.0, 0.0, 0.0], dtype=np.float32), "agent_1": np.zeros(3, dtype=np.float32)})
        assert np.linalg.norm(env.state["agent_0"]["pos"] - start) == pytest.approx(env.move_speed / 2.0)
    finally:
        env.close()


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
    # Opponent block starts after energy and size at indices 5:8; index 9 is visibility.
    assert obs[5] == 0.0 and obs[6] == 0.0 and obs[7] == 0.0
    assert obs[9] == 0.0, "visibility flag should be 0 when out of sensing range"


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
    # Four episodes: agent_0 KO win, agent_1 KO win, a timeout, and a
    # learner elimination in an N-agent self-play free-for-all.
    cb.locals = {
        "infos": [
            {"episode_outcome": {"outcome": "ko", "winner": "agent_0"}},
            {"episode_outcome": {"outcome": "ko", "winner": "agent_1"}},
            {"episode_outcome": {"outcome": "timeout", "winner": None}},
            {"episode_outcome": {"outcome": "eliminated", "winner": None}},
            {"step": 5},  # non-terminal info — ignored
        ]
    }
    cb._on_step()
    cb._on_rollout_end()
    recs = cb.logger.records
    assert abs(recs["arena/agent_0_win_rate"] - 1 / 4) < 1e-9
    assert abs(recs["arena/agent_1_win_rate"] - 1 / 4) < 1e-9
    assert abs(recs["arena/timeout_rate"] - 1 / 4) < 1e-9
    assert abs(recs["arena/elimination_rate"] - 1 / 4) < 1e-9
    assert recs["arena/episode_outcomes"] == 4


def test_arena_metrics_callback_generalizes_beyond_two_agents() -> None:
    """N-agent free-for-alls (num_agents up to 8) must get their win rate
    logged too, not just agent_0/agent_1 — the callback used to hardcode a
    2-entry wins dict, so agent_2+ wins were silently dropped."""
    from types import SimpleNamespace

    from rl_framework.training.sb3_runner import ArenaMetricsCallback

    class _StubLogger:
        def __init__(self) -> None:
            self.records: dict[str, float] = {}

        def record(self, key: str, value: float) -> None:
            self.records[key] = value

    cb = ArenaMetricsCallback()
    cb.model = SimpleNamespace(logger=_StubLogger())
    # 4-agent free-for-all: agent_2 wins twice, agent_0 once, one draw.
    cb.locals = {
        "infos": [
            {"episode_outcome": {"outcome": "ko", "winner": "agent_2"}},
            {"episode_outcome": {"outcome": "ko", "winner": "agent_2"}},
            {"episode_outcome": {"outcome": "ko", "winner": "agent_0"}},
            {"episode_outcome": {"outcome": "draw", "winner": None}},
        ]
    }
    cb._on_step()
    cb._on_rollout_end()
    recs = cb.logger.records
    assert abs(recs["arena/agent_2_win_rate"] - 0.5) < 1e-9, (
        "agent_2 (not hardcoded into the original 2-entry wins dict) must "
        "have its win rate logged"
    )
    assert abs(recs["arena/agent_0_win_rate"] - 0.25) < 1e-9
    assert abs(recs["arena/agent_1_win_rate"] - 0.0) < 1e-9, (
        "agent_1 must still be logged at 0.0, not omitted"
    )
    assert abs(recs["arena/draw_rate"] - 0.25) < 1e-9
    assert recs["arena/episode_outcomes"] == 4

    # A later rollout with no agent_2 wins must still log its rate as 0.0,
    # not drop the key now that it has been discovered.
    cb.locals = {"infos": [{"episode_outcome": {"outcome": "ko", "winner": "agent_0"}}]}
    cb._on_step()
    cb._on_rollout_end()
    assert cb.logger.records["arena/agent_2_win_rate"] == 0.0


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


def test_arena_construction_tolerates_null_sections() -> None:
    """The GUI wizard has historically written `section: null` for empty
    nested groups. cfg.get(key, {}) does not help there (a present key with
    value None is returned as None, not the default), so construction used to
    raise AttributeError on the first `.get(...)` call against a null
    section."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "sim": None,
        "battle_rules": None,
        "morphology": None,
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    assert env.morphology == {}


def test_arena_update_live_params_tolerates_null_sections() -> None:
    """update_live_params must not crash (or silently no-op forever) when the
    target section exists in cfg as an explicit null."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "reward": None,
        "battle_rules": None,
        "morphology": None,
        "sim": None,
    }
    env = make_env("organism_arena_parallel", cfg)
    env.update_live_params(
        {
            "reward.damage_scale": 0.5,
            "battle_rules.cooldown_steps": 4,
            "morphology.base_size": 1.3,
            "sim.arena_half_extent": 2.0,
        }
    )
    assert env.cfg["reward"]["damage_scale"] == 0.5
    assert env.cfg["battle_rules"]["cooldown_steps"] == 4
    assert env.cfg["morphology"]["base_size"] == 1.3
    assert env.cfg["sim"]["arena_half_extent"] == 2.0


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
    # Near-dead but still alive (a truly 0-health opponent is gone, not sensed).
    env.state["agent_1"]["health"] = 0.003
    env.state["agent_0"]["cooldown"] = 2

    obs = env.observe("agent_0")
    assert obs[2] == 1.0, "own health is a fraction of max, not raw health"
    assert obs[7] > 0.0 and obs[7] < 0.01 and obs[9] == 1.0, (
        "visible near-dead opponent: tiny health fraction but visibility flag 1"
    )
    assert abs(obs[8] - 0.5) < 1e-6, "cooldown is a fraction of cooldown_steps"

    env.state["agent_1"]["pos"] = np.array([5.0, 0.0], dtype=np.float32)
    obs = env.observe("agent_0")
    assert obs[7] == 0.0 and obs[9] == 0.0, (
        "out-of-range opponent: zero block plus visibility flag 0"
    )


def test_arena_growth_scales_max_health_and_preserves_fraction() -> None:
    """B6: episode growth raises max_health in lockstep with size, and current
    health is rescaled so the health fraction is preserved across growth."""
    growth = 0.01
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "sim": {"spawn_jitter": 0.0},
        "morphology": {"base_size": 1.0, "episode_growth_scale": growth, "health": 2.0},
        # No damage: isolate growth from combat so the only health change is
        # the growth rescale.
        "battle_rules": {"damage": 0.0, "max_steps": 50},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    a = env.state["agent_0"]
    assert a["size"] == 1.0
    assert a["max_health"] == 2.0 and a["health"] == 2.0

    noop = np.zeros(3, dtype=np.float32)
    # Take a step at full health: size grows, max_health and health track it.
    env.step({"agent_0": noop, "agent_1": noop})
    a = env.state["agent_0"]
    expected_size = 1.0 + growth * env.step_count
    assert abs(a["size"] - expected_size) < 1e-6
    assert abs(a["max_health"] - 2.0 * expected_size) < 1e-6
    # Full health before growth -> still full (fraction 1.0) after.
    assert abs(a["health"] - a["max_health"]) < 1e-6


def test_arena_growth_preserves_partial_health_fraction() -> None:
    """A wounded organism keeps its health *fraction* (not absolute hp) on growth."""
    growth = 0.05
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "sim": {"spawn_jitter": 0.0},
        "morphology": {"base_size": 1.0, "episode_growth_scale": growth, "health": 1.0},
        "battle_rules": {"damage": 0.0, "max_steps": 50},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    # Wound agent_0 to 50% before any growth step.
    env.state["agent_0"]["health"] = 0.5 * env.state["agent_0"]["max_health"]
    noop = np.zeros(3, dtype=np.float32)
    env.step({"agent_0": noop, "agent_1": noop})
    a = env.state["agent_0"]
    assert abs(a["health"] / a["max_health"] - 0.5) < 1e-6, (
        "growth should preserve the health fraction, not the absolute hp"
    )


def test_arena_no_growth_keeps_max_health_fixed() -> None:
    """With episode_growth_scale=0, max_health stays at its spawn value."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "sim": {"spawn_jitter": 0.0},
        "morphology": {"base_size": 1.0, "episode_growth_scale": 0.0, "health": 1.5},
        "battle_rules": {"damage": 0.0, "max_steps": 10},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    noop = np.zeros(3, dtype=np.float32)
    for _ in range(5):
        env.step({"agent_0": noop, "agent_1": noop})
    a = env.state["agent_0"]
    assert a["size"] == 1.0
    assert abs(a["max_health"] - 1.5) < 1e-6 and abs(a["health"] - 1.5) < 1e-6


# ----- N-agent arenas (Phase 4) -----


def test_arena_rejects_fewer_than_two_agents() -> None:
    import pytest

    with pytest.raises(ValueError, match="num_agents must be >= 2"):
        make_env(
            "organism_arena_parallel",
            {"type": "organism_arena_parallel", "num_agents": 1},
        )


def test_arena_n_agents_reset_and_obs_shape() -> None:
    cfg = {"type": "organism_arena_parallel", "seed": 0, "num_agents": 4}
    env = make_env("organism_arena_parallel", cfg)
    obs, infos = env.reset(seed=0)
    assert set(obs) == {"agent_0", "agent_1", "agent_2", "agent_3"}
    for a in obs:
        # Obs stays a fixed 13-D (nearest opponent + nearest food) regardless of N.
        assert obs[a].shape == env.observation_space(a).shape == (13,)
    # Spawns are distinct (circle layout + jitter).
    positions = [tuple(env.state[a]["pos"]) for a in env.possible_agents]
    assert len(set(positions)) == 4


def test_arena_two_agent_spawn_unchanged_under_circle() -> None:
    """The N-agent circle layout must reduce exactly to the legacy 2-agent spawn."""
    cfg = {
        "type": "organism_arena_parallel",
        "seed": 0,
        "num_agents": 2,
        "sim": {"spawn_jitter": 0.0},
    }
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    assert np.allclose(env.state["agent_0"]["pos"], [-0.6, 0.0], atol=1e-6)
    assert np.allclose(env.state["agent_1"]["pos"], [0.6, 0.0], atol=1e-6)


def test_arena_n_agent_last_standing_winner() -> None:
    """With all but one knocked out, the episode ends with a single ko winner."""
    cfg = {"type": "organism_arena_parallel", "seed": 0, "num_agents": 3}
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    # Knock out agent_1 and agent_2 directly; agent_0 survives.
    env.state["agent_1"]["health"] = 0.0
    env.state["agent_2"]["health"] = 0.0
    noop = np.zeros(3, dtype=np.float32)
    obs, rewards, terms, truncs, infos = env.step({a: noop for a in env.agents})
    assert all(terms.values()), "elimination terminates every agent together"
    assert rewards["agent_0"] == 1.0, "lone survivor gets +1"
    assert rewards["agent_1"] == -1.0 and rewards["agent_2"] == -1.0
    assert infos["agent_0"]["episode_outcome"] == {
        "winner": "agent_0",
        "outcome": "ko",
        "step": 1,
    }
    assert env.agents == []


def test_arena_n_agent_death_is_inert_spectator_until_end() -> None:
    """A death with >1 survivor does not end the episode; the dead agent lingers
    as an inert, zero-reward spectator with a constant agent set."""
    cfg = {"type": "organism_arena_parallel", "seed": 0, "num_agents": 3}
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    env.state["agent_2"]["health"] = 0.0  # only agent_2 dies
    noop = np.zeros(3, dtype=np.float32)
    obs, rewards, terms, truncs, infos = env.step({a: noop for a in env.agents})
    assert not any(terms.values()), "episode continues with 2 alive"
    assert rewards["agent_2"] == -1.0, "death is penalised once"
    assert "episode_outcome" not in infos["agent_0"]
    assert len(env.agents) == 3, "agent set stays constant (spectator lingers)"

    # Next step: the spectator earns no further reward and cannot act/be hit.
    _, rewards2, _, _, _ = env.step({a: noop for a in env.agents})
    assert rewards2["agent_2"] == 0.0


def test_arena_n_agent_simultaneous_wipeout_is_draw() -> None:
    cfg = {"type": "organism_arena_parallel", "seed": 0, "num_agents": 3}
    env = make_env("organism_arena_parallel", cfg)
    env.reset(seed=0)
    for a in ("agent_0", "agent_1", "agent_2"):
        env.state[a]["health"] = 0.0
    noop = np.zeros(3, dtype=np.float32)
    _, rewards, terms, _, infos = env.step({a: noop for a in env.agents})
    assert all(terms.values())
    assert infos["agent_0"]["episode_outcome"]["outcome"] == "draw"
    assert all(rewards[a] == -1.0 for a in ("agent_0", "agent_1", "agent_2"))
