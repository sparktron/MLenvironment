from __future__ import annotations

import ast
import importlib.util
import subprocess
from pathlib import Path


def _tracked_files() -> list[str]:
    output = subprocess.check_output(["git", "ls-files"], text=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _check_no_generated_files(tracked_files: list[str]) -> list[str]:
    violations: list[str] = []
    for path in tracked_files:
        if (
            "__pycache__/" in path
            or path.endswith(".pyc")
            or ".egg-info/" in path
            or path.startswith(".venv/")
        ):
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


def _check_import_directions(package: Path) -> list[str]:
    """Check every import and non-docstring literal in the framework tree."""
    violations: list[str] = []
    allowed = {"core": {"core"}, "physics": {"core", "physics"},
               "games": {"core", "games"}}
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package)
        layer = relative.parts[0]
        if layer not in allowed and relative.as_posix() != "plugins.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parts = list(relative.with_suffix("").parts)
        package_parts = parts[:-1] if parts[-1] != "__init__" else parts[:-1]
        current_package = ".".join(["rl_framework", *package_parts])
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    try:
                        module = importlib.util.resolve_name("." * node.level + module, current_package)
                    except ImportError:
                        violations.append(f"{path}:{node.lineno}: invalid relative import: {ast.unparse(node)}")
                        continue
                targets = [module]
                if module == "rl_framework":
                    targets = [f"{module}.{alias.name}" for alias in node.names]
            for target in targets:
                if target != "rl_framework" and not target.startswith("rl_framework."):
                    continue
                exception = (relative.as_posix() == "core/registry.py"
                             and isinstance(node, ast.ImportFrom) and node.level == 0
                             and node.module == "rl_framework.plugins"
                             and len(node.names) == 1 and node.names[0].name == "PLUGIN_MODULES")
                destination = target.removeprefix("rl_framework.").split(".")[0]
                if not exception and (layer == "plugins.py" or destination not in allowed[layer]):
                    violations.append(f"{path}:{node.lineno}: {ast.unparse(node)}: {layer} import rule")
            if (layer == "core" and isinstance(node, ast.Constant)
                    and isinstance(node.value, str) and id(node) not in docstrings
                    and (node.value in {"walker_bullet", "organism_arena_parallel"}
                         or node.value.startswith(("rl_framework.physics", "rl_framework.games")))):
                violations.append(f"{path}:{node.lineno}: {node.value!r}: core task literal rule")
    return violations


def main() -> int:
    imports = _check_import_directions(Path("src/rl_framework"))
    if imports:
        for violation in imports:
            print(violation)
        return 1
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
