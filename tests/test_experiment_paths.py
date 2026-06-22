"""Tests for the experiment output layout helper.

Covers the run_id nesting introduced to stop sweep/morph orchestration from
mutating experiment_name to encode variants into the directory name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rl_framework.utils.logging_utils import (
    create_experiment_paths,
    sanitize_run_id,
)


def test_paths_without_run_id_use_flat_seed_layout(tmp_path: Path) -> None:
    paths = create_experiment_paths(str(tmp_path), "exp", 3)
    assert paths.run_dir == tmp_path / "exp" / "seed_3"
    assert paths.checkpoints_dir == paths.run_dir / "checkpoints"
    assert paths.logs_dir == paths.run_dir / "logs"
    assert paths.videos_dir == paths.run_dir / "videos"
    for p in (paths.run_dir, paths.checkpoints_dir, paths.logs_dir, paths.videos_dir):
        assert p.is_dir()


def test_paths_with_run_id_nest_under_runs(tmp_path: Path) -> None:
    paths = create_experiment_paths(
        str(tmp_path), "exp", 0, run_id="lr_0.001__gamma_0.99"
    )
    assert (
        paths.run_dir == tmp_path / "exp" / "runs" / "lr_0.001__gamma_0.99" / "seed_0"
    )
    assert paths.run_dir.is_dir()


def test_sanitize_run_id_replaces_unsafe_characters() -> None:
    # Path separators and other unsafe characters collapse to underscores,
    # keeping the whole variant on a single directory level.
    assert sanitize_run_id("a/b c:d") == "a_b_c_d"
    # The conservative allowlist preserves the characters sweep ids actually use.
    assert sanitize_run_id("lr_0.001__gamma_0.99") == "lr_0.001__gamma_0.99"


@pytest.mark.parametrize("bad", ["", ".", "..", "___", "."])
def test_sanitize_run_id_rejects_dotty_or_empty_ids(bad: str) -> None:
    with pytest.raises(ValueError, match="unsafe path segment"):
        sanitize_run_id(bad)


def test_run_id_path_traversal_is_neutralised(tmp_path: Path) -> None:
    paths = create_experiment_paths(str(tmp_path), "exp", 0, run_id="../escape")
    # The traversal is sanitised to a literal child of runs/, not an escape.
    assert paths.run_dir.parent.parent == tmp_path / "exp" / "runs"
    assert tmp_path in paths.run_dir.parents
