"""Tests for the round-robin arena tournament and its Bradley-Terry ratings."""

from __future__ import annotations

import json

import pytest

from rl_framework.training import arena_tournament
from rl_framework.training.arena_tournament import (
    _bradley_terry_elo,
    _unique_labels,
    resolve_competitors,
    run_tournament,
)


def _fake_eval_by_index(policy, opponent, cfg, n_episodes, swap_roles):
    """Deterministic stand-in for run_arena_eval: higher path index wins 80%."""
    pi, oi = int(policy[-1]), int(opponent[-1])
    total = n_episodes * (2 if swap_roles else 1)
    if pi > oi:
        p_wr, o_wr = 0.8, 0.2
    elif pi < oi:
        p_wr, o_wr = 0.2, 0.8
    else:
        p_wr, o_wr = 0.5, 0.5
    return {
        "policy_win_rate": p_wr,
        "opponent_win_rate": o_wr,
        "draw_rate": 0.0,
        "timeout_rate": 0.0,
        "n_episodes": total,
    }


# -- Bradley-Terry / Elo -----------------------------------------------------


def test_bradley_terry_orders_by_strength() -> None:
    names = ["A", "B", "C"]
    # A beats B and C 80%; B beats C 80%.
    wins = {a: {b: 0.0 for b in names} for a in names}
    games = {a: {b: 0 for b in names} for a in names}
    pairs = {("A", "B"): 0.8, ("A", "C"): 0.8, ("B", "C"): 0.8}
    for (x, y), wr in pairs.items():
        wins[x][y] = wr * 10
        wins[y][x] = (1 - wr) * 10
        games[x][y] = games[y][x] = 10
    elo = _bradley_terry_elo(names, wins, games)
    assert elo["A"] > elo["B"] > elo["C"]


def test_bradley_terry_symmetric_field_is_flat() -> None:
    names = ["A", "B"]
    wins = {"A": {"A": 0.0, "B": 5.0}, "B": {"A": 5.0, "B": 0.0}}
    games = {"A": {"A": 0, "B": 10}, "B": {"A": 10, "B": 0}}
    elo = _bradley_terry_elo(names, wins, games)
    assert abs(elo["A"] - elo["B"]) < 1e-6
    assert abs(elo["A"] - 1500.0) < 1e-6  # geomean-normalized zero point


# -- labels / competitor resolution ------------------------------------------


def test_unique_labels_uses_stem_then_parent() -> None:
    assert _unique_labels(["a/x.zip", "b/y.zip"]) == ["x", "y"]
    assert _unique_labels(["run1/final_model.zip", "run2/final_model.zip"]) == [
        "run1/final_model",
        "run2/final_model",
    ]


def test_resolve_competitors_expands_dir_and_random(tmp_path) -> None:
    (tmp_path / "a.zip").write_text("x", encoding="utf-8")
    (tmp_path / "b.zip").write_text("x", encoding="utf-8")
    competitors = resolve_competitors([str(tmp_path)], include_random=True)
    labels = [c[0] for c in competitors]
    assert labels == ["a", "b", "random"]
    assert competitors[-1] == ("random", "random")


# -- run_tournament ----------------------------------------------------------


def test_run_tournament_ranks_field(monkeypatch) -> None:
    monkeypatch.setattr(arena_tournament, "run_arena_eval", _fake_eval_by_index)
    result = run_tournament(["p0", "p1", "p2"], {}, n_episodes=5)

    standings = result["standings"]
    assert [s["competitor"] for s in standings] == ["p2", "p1", "p0"]
    assert [s["rank"] for s in standings] == [1, 2, 3]
    # 3 competitors -> 3 unique pairings.
    assert len(result["matches"]) == 3
    # Win-rate matrix reflects the stub: p2 beats p0 80%.
    assert result["win_rate_matrix"]["p2"]["p0"] == 0.8
    assert result["win_rate_matrix"]["p0"]["p2"] == 0.2
    assert result["win_rate_matrix"]["p1"]["p1"] is None


def test_run_tournament_writes_json_and_markdown(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(arena_tournament, "run_arena_eval", _fake_eval_by_index)
    json_out = tmp_path / "t.json"
    md_out = tmp_path / "t.md"
    run_tournament(
        ["p0", "p1"],
        {},
        n_episodes=3,
        output_path=str(json_out),
        markdown_path=str(md_out),
    )
    payload = json.loads(json_out.read_text())
    assert {"standings", "ratings", "win_rate_matrix", "matches"} <= set(payload)
    md = md_out.read_text()
    assert "## Standings" in md and "Win-rate matrix" in md
    assert "p0" in md and "p1" in md


def test_run_tournament_requires_two_competitors() -> None:
    with pytest.raises(ValueError, match="at least 2 competitors"):
        run_tournament(["only_one"], {}, n_episodes=1)


def test_run_tournament_counts_draws_as_half(monkeypatch) -> None:
    """All-draw matchups should leave the field rated equally."""

    def all_draws(policy, opponent, cfg, n_episodes, swap_roles):
        total = n_episodes * (2 if swap_roles else 1)
        return {
            "policy_win_rate": 0.0,
            "opponent_win_rate": 0.0,
            "draw_rate": 0.5,
            "timeout_rate": 0.5,
            "n_episodes": total,
        }

    monkeypatch.setattr(arena_tournament, "run_arena_eval", all_draws)
    result = run_tournament(["p0", "p1", "p2"], {}, n_episodes=4)
    elos = list(result["ratings"].values())
    assert max(elos) - min(elos) < 1e-6
    # Every game is a draw -> zero wins/losses, all draws.
    for s in result["standings"]:
        assert s["wins"] == 0 and s["losses"] == 0
        assert s["draws"] == s["games"]


def test_run_tournament_uses_n_agent_semantics(monkeypatch) -> None:
    def fake_n_agent(paths, cfg, n_episodes):
        return {
            "agent_mean_scores": {"agent_0": 1.0, "agent_1": 0.0, "agent_2": 0.0},
            "agent_win_rates": {"agent_0": 1.0, "agent_1": 0.0, "agent_2": 0.0},
            "draw_rate": 0.0, "timeout_rate": 0.0, "n_episodes": n_episodes,
        }
    monkeypatch.setattr(arena_tournament, "run_n_agent_eval", fake_n_agent)
    result = run_tournament(["p0", "p1", "p2"], {"environment": {"num_agents": 3}}, n_episodes=2)
    assert result["n_agents"] == 3
    assert len(result["matches"]) == 3
