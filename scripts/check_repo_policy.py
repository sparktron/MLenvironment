from __future__ import annotations

import subprocess
import os
from pathlib import Path


def _tracked_files() -> list[str]:
    output = subprocess.check_output(["git", "ls-files"], text=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _check_no_generated_files(tracked_files: list[str]) -> list[str]:
    violations: list[str] = []
    for path in tracked_files:
        if "__pycache__/" in path or path.endswith(".pyc") or ".egg-info/" in path or path.startswith(".venv/"):
            violations.append(path)
    return violations


def _direct_dependencies_from_pyproject(pyproject_path: Path) -> set[str]:
    text = pyproject_path.read_text(encoding="utf-8")
    in_dependencies = False
    names: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "dependencies = [":
            in_dependencies = True
            continue
        if in_dependencies and stripped == "]":
            break
        if in_dependencies and stripped.startswith('"'):
            package = stripped.strip('",')
            name = package.split(">=")[0].split("==")[0].strip().lower()
            if name:
                names.add(name)
    return names


def _dependency_names_from_lockfile(lockfile_path: Path) -> set[str]:
    names: set[str] = set()
    for line in lockfile_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "==" not in stripped:
            continue
        names.add(stripped.split("==")[0].strip().lower())
    return names


def main() -> int:
    if os.environ.get("STRICT_REPO_CLEAN", "0") == "1":
        generated = _check_no_generated_files(_tracked_files())
        if generated:
            print("Generated files must not be tracked:")
            for path in generated:
                print(f" - {path}")
            return 1

    pyproject_path = Path("pyproject.toml")
    lockfile_path = Path("requirements-lock.txt")
    if not lockfile_path.exists():
        print("Missing requirements-lock.txt")
        return 1

    direct_deps = _direct_dependencies_from_pyproject(pyproject_path)
    locked_deps = _dependency_names_from_lockfile(lockfile_path)
    missing = sorted(dep for dep in direct_deps if dep not in locked_deps)
    if missing:
        print("Lockfile is missing direct dependencies:")
        for dep in missing:
            print(f" - {dep}")
        return 1

    print("Repository policy checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
