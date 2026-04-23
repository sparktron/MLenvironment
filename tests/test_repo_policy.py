from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_repo_policy.py"
SPEC = importlib.util.spec_from_file_location("check_repo_policy", MODULE_PATH)
check_repo_policy = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(check_repo_policy)


def test_generated_file_detector_flags_expected_paths() -> None:
    tracked = [
        "src/rl_framework/train.py",
        "src/rl_framework/__pycache__/train.cpython-311.pyc",
        ".venv/bin/python",
        "src/rl_experiment_framework.egg-info/PKG-INFO",
    ]
    assert check_repo_policy._check_no_generated_files(tracked) == [
        "src/rl_framework/__pycache__/train.cpython-311.pyc",
        ".venv/bin/python",
        "src/rl_experiment_framework.egg-info/PKG-INFO",
    ]


def test_lockfile_parser_finds_direct_dependency_names(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "\n".join(
            [
                "[project]",
                "dependencies = [",
                '  "numpy>=1.0",',
                '  "torch==2.2.0",',
                "]",
            ]
        ),
        encoding="utf-8",
    )
    lockfile = tmp_path / "requirements-lock.txt"
    lockfile.write_text("numpy==1.2.0\n", encoding="utf-8")

    direct = check_repo_policy._direct_dependencies_from_pyproject(pyproject)
    locked = check_repo_policy._dependency_names_from_lockfile(lockfile)
    missing = sorted(dep for dep in direct if dep not in locked)
    assert missing == ["torch"]
