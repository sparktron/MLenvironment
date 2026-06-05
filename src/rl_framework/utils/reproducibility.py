from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


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
    out_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
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
        if not strict:
            _log.warning(
                "Reproducibility: git commit unavailable (not in a git repo or git not installed). "
                "run_metadata.json will be incomplete."
            )

    if strict:
        _require(config_hash, "Missing config hash in strict reproducibility mode.")
        _require(
            git_info["commit"], "Missing git commit in strict reproducibility mode."
        )
        _require(
            lockfile_hash,
            f"Missing lockfile hash for {lockfile_path} in strict reproducibility mode.",
        )

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
            "cwd": _sanitize_cwd(os.getcwd()),
        },
        "warnings": warnings,
    }


def _stable_config_json(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def check_resume_provenance(
    resume_from: str | Path,
    cfg: dict[str, Any],
    *,
    strict: bool,
) -> list[str]:
    """Verify a resuming run is compatible with the checkpoint it continues.

    Stable-Baselines3 already rejects a resume whose observation/action spaces
    differ (``PPO.load(env=...)`` calls ``check_for_correct_spaces``). This
    catches the *silent* drift it misses: a changed ``environment`` block,
    flipped observation normalization, or a different policy architecture /
    discount, any of which would make the loaded weights or ``vecnormalize.pkl``
    statistics wrong for the new run without raising on its own.

    The comparison is against the source checkpoint's ``run_metadata.json``
    (written by :func:`write_run_metadata`). Fields that legitimately vary across
    a resume — ``total_timesteps``, learning rate, seed, output paths — are
    excluded.

    Parameters
    ----------
    resume_from:
        Path to the checkpoint being resumed (``.zip`` or extension-less).
    cfg:
        The resolved config for the resuming run.
    strict:
        When True, any drift — or an unverifiable provenance chain (missing
        source manifest) — raises :class:`RuntimeError`. When False, drift is
        logged as a warning and returned.

    Returns
    -------
    The list of human-readable drift messages (empty when fully compatible).
    """
    model_path = _as_zip_path(Path(resume_from))
    meta_path = _find_source_manifest(model_path)
    if meta_path is None:
        return _report(
            [
                f"cannot verify resume provenance: no run_metadata.json found near "
                f"{model_path} (pre-manifest checkpoint?)"
            ],
            strict=strict,
        )

    try:
        source_cfg = json.loads(meta_path.read_text(encoding="utf-8")).get("config")
    except (OSError, ValueError) as exc:
        return _report(
            [f"cannot verify resume provenance: failed to read {meta_path}: {exc}"],
            strict=strict,
        )
    if not isinstance(source_cfg, dict):
        return _report(
            [f"cannot verify resume provenance: {meta_path} has no 'config' block"],
            strict=strict,
        )

    drift = _fingerprint_diffs(
        _resume_fingerprint(source_cfg), _resume_fingerprint(cfg)
    )
    return _report(drift, strict=strict)


def _report(messages: list[str], *, strict: bool) -> list[str]:
    if messages and strict:
        raise RuntimeError(
            "Resume provenance check failed (strict reproducibility):\n  - "
            + "\n  - ".join(messages)
        )
    for msg in messages:
        _log.warning("Resume provenance: %s", msg)
    return messages


def _as_zip_path(resume_from: Path) -> Path:
    return (
        resume_from
        if str(resume_from).endswith(".zip")
        else Path(str(resume_from) + ".zip")
    )


def _find_source_manifest(model_path: Path) -> Path | None:
    """Locate the run_metadata.json for a checkpoint.

    The standard layout is ``<run_dir>/checkpoints/<name>.zip`` with the
    manifest at ``<run_dir>/run_metadata.json``, so the run dir is two levels
    up. Fall back to the checkpoint's own directory for non-standard layouts.
    """
    for run_dir in (model_path.parent.parent, model_path.parent):
        candidate = run_dir / "run_metadata.json"
        if candidate.is_file():
            return candidate
    return None


def _resume_fingerprint(cfg: dict[str, Any]) -> dict[str, Any]:
    """Project a config down to the fields that must match across a resume.

    The whole ``environment`` block is included (env type, physics, reward,
    termination, morphology) since changing any of it alters the task or the
    observation distribution the loaded normalization stats describe. The seed
    is dropped — like the top-level seed, it is a legitimate per-run knob.
    """
    env = {k: v for k, v in cfg.get("environment", {}).items() if k != "seed"}
    training = cfg.get("training", {})
    return {
        "environment": env,
        "training.policy": training.get("policy", "MlpPolicy"),
        "training.policy_kwargs": training.get("policy_kwargs"),
        "training.normalize_observations": training.get("normalize_observations", True),
        "training.gamma": training.get("gamma", 0.99),
    }


def _fingerprint_diffs(source: Any, current: Any, prefix: str = "") -> list[str]:
    """Return dotted-path messages for every leaf that differs between two trees."""
    if isinstance(source, dict) and isinstance(current, dict):
        diffs: list[str] = []
        for key in sorted(set(source) | set(current)):
            sub = f"{prefix}.{key}" if prefix else str(key)
            diffs.extend(_fingerprint_diffs(source.get(key), current.get(key), sub))
        return diffs
    if source != current:
        return [f"{prefix or '<root>'}: checkpoint={source!r} -> resume={current!r}"]
    return []


def _git_info() -> dict[str, Any]:
    return {
        "commit": _git_cmd(["rev-parse", "HEAD"]),
        "branch": _git_cmd(["rev-parse", "--abbrev-ref", "HEAD"]),
        "is_dirty": _git_dirty(),
    }


def _git_cmd(args: list[str]) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", *args], text=True, stderr=subprocess.STDOUT
            ).strip()
            or None
        )
    except Exception as exc:
        _log.debug("git %s failed: %s", " ".join(args), exc)
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


def _sanitize_cwd(cwd: str) -> str:
    """Return CWD with home directory replaced by '~' to avoid leaking absolute paths."""
    try:
        return "~/" + str(Path(cwd).relative_to(Path.home()))
    except ValueError:
        return cwd
