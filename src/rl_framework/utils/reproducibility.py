from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_run_metadata(
    run_dir: Path,
    cfg: dict[str, Any],
    *,
    strict: bool = False,
    resume_from: str | Path | None = None,
    lockfile_path: str | Path = "requirements-lock.txt",
) -> Path:
    """Write reproducibility metadata for a run.

    In strict mode, required provenance fields must be available or this raises.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = _build_metadata(
        cfg,
        strict=strict,
        resume_from=resume_from,
        lockfile_path=Path(lockfile_path),
    )
    out_path = run_dir / "run_metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def _build_metadata(
    cfg: dict[str, Any],
    *,
    strict: bool,
    resume_from: str | Path | None,
    lockfile_path: Path,
) -> dict[str, Any]:
    warnings: list[str] = []
    config_hash = hashlib.sha256(_stable_config_json(cfg).encode("utf-8")).hexdigest()
    git_info = _git_info()
    lockfile_hash = _sha256_file(lockfile_path)
    if lockfile_hash is None:
        warnings.append(f"missing_lockfile:{lockfile_path}")
    if git_info["commit"] is None:
        warnings.append("missing_git_commit")

    if strict:
        _require(config_hash, "Missing config hash in strict reproducibility mode.")
        _require(git_info["commit"], "Missing git commit in strict reproducibility mode.")
        _require(lockfile_hash, f"Missing lockfile hash for {lockfile_path} in strict reproducibility mode.")

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strict_reproducibility": bool(strict),
        "experiment_name": cfg.get("experiment_name"),
        "seed": cfg.get("seed"),
        "resume_from": str(resume_from) if resume_from is not None else None,
        "config_hash_sha256": config_hash,
        "config": cfg,
        "git": git_info,
        "lockfile": {
            "path": str(lockfile_path),
            "sha256": lockfile_hash,
        },
        "runtime": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "hostname": socket.gethostname(),
            "cwd": os.getcwd(),
        },
        "warnings": warnings,
    }


def _stable_config_json(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _git_info() -> dict[str, Any]:
    return {
        "commit": _git_cmd(["rev-parse", "HEAD"]),
        "branch": _git_cmd(["rev-parse", "--abbrev-ref", "HEAD"]),
        "is_dirty": _git_dirty(),
    }


def _git_cmd(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip() or None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        subprocess.check_output(["git", "diff", "--quiet"], stderr=subprocess.STDOUT)
        return False
    except subprocess.CalledProcessError:
        return True
    except Exception:
        return None


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _require(value: Any, message: str) -> None:
    if value is None or value == "":
        raise RuntimeError(message)
