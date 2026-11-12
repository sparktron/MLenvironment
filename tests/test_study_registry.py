from types import SimpleNamespace

import pytest

from rl_framework.core import studies
from rl_framework.training import quality_study


def test_task_study_contribution_drives_plan_and_dispatch(tmp_path, monkeypatch):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return {"result": "done"}

    spec = studies.StudySpec(
        "custom", "custom_result", run, 2, 123, 0, include_in_all=True
    )
    monkeypatch.setattr(studies, "registered_tasks", lambda: ("custom_task",))
    monkeypatch.setattr(studies, "get_task", lambda name: SimpleNamespace(studies=lambda: (spec,)))

    plan = quality_study.run_quality_study("all", seeds=[1, 2], dry_run=True)
    assert plan["identity"]["studies"] == ["custom"]
    assert plan["planned_runs"] == {"custom_result": 4}

    report = quality_study.run_quality_study("all", seeds=[1, 2], output_dir=tmp_path)
    assert report["results"] == {"custom_result": {"result": "done"}}
    assert calls[0]["step_budget"] == 123
    assert calls[0]["seeds"] == [1, 2]


def test_duplicate_study_names_are_rejected(monkeypatch):
    spec = studies.StudySpec("same", "result", lambda **kwargs: {}, 1, 10, 0)
    monkeypatch.setattr(studies, "registered_tasks", lambda: ("one", "two"))
    monkeypatch.setattr(studies, "get_task", lambda name: SimpleNamespace(studies=lambda: (spec,)))
    with pytest.raises(ValueError, match="Duplicate quality study name"):
        studies.registered_studies()
