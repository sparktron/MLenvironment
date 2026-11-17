from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

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


@pytest.mark.parametrize(("filename", "source"), [
    ("core/x.py", "import rl_framework.physics.locomotion"),
    ("physics/x.py", "from rl_framework.games import arena"),
    ("games/x.py", "from rl_framework.physics import locomotion"),
    ("plugins.py", "import rl_framework.core.registry"),
    ("core/x.py", "def f():\n    import rl_framework.games.arena"),
    ("physics/x.py", "from ..games import arena"),
    ("core/x.py", "value = 'walker_bullet'"),
    ("core/x.py", "value = 'organism_arena_parallel'"),
    ("core/x.py", "value = 'rl_framework.physics.foo'"),
    ("core/x.py", "value = 'rl_framework.games.foo'"),
    ("core/x.py", "from rl_framework.plugins import PLUGIN_MODULES"),
])
def test_import_policy_rejects_violations(tmp_path: Path, filename: str, source: str) -> None:
    path = tmp_path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source + "\n")
    violations = check_repo_policy._check_import_directions(tmp_path)
    assert len(violations) == 1
    assert f"{path}:" in violations[0]


def test_import_policy_allows_clean_tree_and_registry_exception(tmp_path: Path) -> None:
    for filename, source in {
        "core/registry.py": "from rl_framework.plugins import PLUGIN_MODULES",
        "core/x.py": '"""walker_bullet"""\nfrom . import registry',
        "physics/x.py": "from rl_framework.core import registry",
        "games/x.py": "from rl_framework.games import arena",
        "plugins.py": "PLUGIN_MODULES = ('rl_framework.physics',)",
    }.items():
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source + "\n")
    assert check_repo_policy._check_import_directions(tmp_path) == []


def test_real_tree_obeys_import_policy() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "rl_framework"
    assert check_repo_policy._check_import_directions(root) == []


def test_root_shim_check_flags_root_package_folder(tmp_path: Path) -> None:
    (tmp_path / "rl_framework").mkdir()
    (tmp_path / "rl_framework" / "__init__.py").write_text("", encoding="utf-8")
    assert check_repo_policy._check_no_root_shim(tmp_path)


def test_root_shim_check_allows_src_only_layout(tmp_path: Path) -> None:
    (tmp_path / "src" / "rl_framework").mkdir(parents=True)
    assert check_repo_policy._check_no_root_shim(tmp_path) == []
