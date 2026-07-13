"""Tests for morphology search orchestration.

Uses monkeypatching to replace ``train`` and ``evaluate`` with deterministic
stubs, so this exercises the search loop without pulling SB3/torch in.
"""

from __future__ import annotations

import pytest


def _base_cfg() -> dict:
    return {
        "experiment_name": "exp",
        "seed": 0,
        "output": {"base_dir": "outputs"},
        "environment": {
            "type": "organism_arena_parallel",
            "morphology": {"base_size": 1.0, "health": 1.0},
        },
        "training": {"total_timesteps": 1000},
    }


def test_morphology_search_picks_highest_score(monkeypatch):
    from rl_framework.training import morphology_search as ms

    # Score = base_size so trial with biggest base_size wins.
    scores: list[float] = []
    model_paths_seen: list[str] = []

    def fake_train(cfg, **kw):
        return f"/tmp/{cfg['experiment_name']}_{cfg['output']['run_id']}.zip"

    def fake_eval(cfg, model_path):
        model_paths_seen.append(model_path)
        score = float(cfg["environment"]["morphology"]["base_size"])
        scores.append(score)
        return {"mean_return": score}

    from rl_framework.training import eval_runner, sb3_runner

    monkeypatch.setattr(sb3_runner, "train", fake_train)
    monkeypatch.setattr(eval_runner, "evaluate", fake_eval)

    result = ms.run_morphology_search(_base_cfg(), trials=4, seed=7)

    assert len(result["results"]) == 4
    assert result["best_trial"] in range(4)
    assert result["best_score"] == max(scores)
    assert (
        result["best_params"] == result["results"][result["best_trial"]]["morphology"]
    )
    # experiment_name stays fixed; trials are separated by run_id so outputs
    # don't collide (outputs/<experiment_name>/runs/<run_id>/).
    assert {r["experiment_name"] for r in result["results"]} == {"exp"}
    run_ids = [r["run_id"] for r in result["results"]]
    assert run_ids == ["morph_000", "morph_001", "morph_002", "morph_003"]
    assert all(path.endswith(".zip") for path in model_paths_seen)


def test_as_model_zip_path_keeps_existing_zip_suffix() -> None:
    from rl_framework.training.morphology_search import _as_model_zip_path

    assert _as_model_zip_path("/tmp/model.zip") == "/tmp/model.zip"
    assert _as_model_zip_path("/tmp/model") == "/tmp/model.zip"


def test_morphology_search_rejects_non_arena_env(monkeypatch):
    from rl_framework.training import morphology_search as ms

    cfg = _base_cfg()
    cfg["environment"]["type"] = "walker_bullet"
    with pytest.raises(ValueError, match="organism_arena_parallel"):
        ms.run_morphology_search(cfg, trials=1)


def test_morphology_search_requires_positive_trials():
    from rl_framework.training import morphology_search as ms

    with pytest.raises(ValueError, match="trials"):
        ms.run_morphology_search(_base_cfg(), trials=0)


def test_morphology_search_can_score_by_tournament_elo(monkeypatch):
    from rl_framework.training import morphology_search as ms
    from rl_framework.training import sb3_runner
    from rl_framework.training import arena_tournament

    monkeypatch.setattr(sb3_runner, "train", lambda cfg: f"/tmp/{cfg['output']['run_id']}.zip")
    monkeypatch.setattr(
        arena_tournament,
        "run_tournament",
        lambda paths, *_args, **_kwargs: {"competitors": [{"label": "candidate", "path": paths[0]}], "ratings": {"candidate": 1600.0}},
    )
    cfg = _base_cfg()
    cfg["morphology_search"] = {"scoring": "tournament_elo", "tournament_episodes": 1}
    result = ms.run_morphology_search(cfg, trials=1)
    assert result["best_score"] == 1600.0
