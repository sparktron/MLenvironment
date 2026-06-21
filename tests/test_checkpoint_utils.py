"""Tests for rl_framework.utils.checkpoint path helpers."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from rl_framework.utils.checkpoint import (
    find_vecnormalize_path_for_model,
    legacy_vecnormalize_path_for_model,
    model_zip_path,
    validate_resume_path,
    vecnormalize_path_for_model,
)


def test_model_zip_path_adds_extension():
    p = model_zip_path("/some/dir/model")
    assert str(p).endswith(".zip")


def test_model_zip_path_no_double_extension():
    p = model_zip_path("/some/dir/model.zip")
    assert str(p) == "/some/dir/model.zip"


def test_vecnormalize_path_for_model(tmp_path):
    model = tmp_path / "ppo_100.zip"
    sidecar = vecnormalize_path_for_model(model)
    assert sidecar.name == "ppo_100_vecnormalize.pkl"
    assert sidecar.parent == tmp_path


def test_legacy_vecnormalize_path_for_model(tmp_path):
    model = tmp_path / "model.zip"
    assert legacy_vecnormalize_path_for_model(model).name == "vecnormalize.pkl"


def test_find_vecnormalize_prefers_specific_sidecar(tmp_path):
    model = tmp_path / "model.zip"
    specific = vecnormalize_path_for_model(model)
    legacy = legacy_vecnormalize_path_for_model(model)
    specific.write_bytes(b"s")
    legacy.write_bytes(b"l")
    assert find_vecnormalize_path_for_model(model) == specific


def test_find_vecnormalize_falls_back_to_legacy(tmp_path):
    model = tmp_path / "model.zip"
    legacy = legacy_vecnormalize_path_for_model(model)
    legacy.write_bytes(b"l")
    assert find_vecnormalize_path_for_model(model) == legacy


def test_find_vecnormalize_returns_none_when_absent(tmp_path):
    model = tmp_path / "model.zip"
    assert find_vecnormalize_path_for_model(model) is None


def _make_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("placeholder", "fake")


def test_validate_resume_path_missing_zip(tmp_path):
    with pytest.raises(FileNotFoundError, match="resume_from model not found"):
        validate_resume_path(tmp_path / "ghost.zip", normalize=False)


def test_validate_resume_path_corrupt_zip(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="appears corrupt"):
        validate_resume_path(bad, normalize=False)


def test_validate_resume_path_missing_vecnorm(tmp_path):
    model = tmp_path / "model.zip"
    _make_zip(model)
    with pytest.raises(FileNotFoundError, match="VecNormalize sidecar not found"):
        validate_resume_path(model, normalize=True)


def test_validate_resume_path_passes_with_legacy_vecnorm(tmp_path):
    model = tmp_path / "model.zip"
    _make_zip(model)
    (tmp_path / "vecnormalize.pkl").write_bytes(b"v")
    validate_resume_path(model, normalize=True)  # should not raise


def test_validate_resume_path_passes_with_specific_vecnorm(tmp_path):
    model = tmp_path / "ppo_64.zip"
    _make_zip(model)
    vecnormalize_path_for_model(model).write_bytes(b"v")
    validate_resume_path(model, normalize=True)  # should not raise


def test_sb3_runner_aliases_still_work():
    """The private aliases in sb3_runner resolve to the same functions."""
    from rl_framework.training.sb3_runner import (
        _find_vecnormalize_path_for_model,
        _validate_resume_path,
        _vecnormalize_path_for_model,
    )

    assert _vecnormalize_path_for_model is vecnormalize_path_for_model
    assert _find_vecnormalize_path_for_model is find_vecnormalize_path_for_model
    assert _validate_resume_path is validate_resume_path


def test_morphology_search_uses_shared_helper():
    """morphology_search delegates path normalisation to the shared helper."""
    from rl_framework.training.morphology_search import _as_model_zip_path

    assert _as_model_zip_path("/tmp/model") == str(model_zip_path("/tmp/model"))
    assert _as_model_zip_path("/tmp/model.zip") == "/tmp/model.zip"
