"""Durable registry for training runs and GUI control events.

The registry intentionally uses SQLite from the standard library: it is local,
transactional, and safe for the GUI thread plus training subprocesses without
adding a service dependency.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


class RunRegistry:
    """Store immutable run identity, provenance, events, and artifacts."""

    def __init__(self, base_dir: str | Path) -> None:
        self.path = Path(base_dir) / "run_registry.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    run_dir TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    parent_run_id TEXT,
                    resume_from TEXT,
                    status TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT '',
                    model_path TEXT NOT NULL DEFAULT '',
                    started_at REAL NOT NULL,
                    finished_at REAL,
                    latest_metrics_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS run_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tuning_commands (
                    command_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    params_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    applied_at REAL
                );
                CREATE TABLE IF NOT EXISTS run_artifacts (
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (run_id, kind, path)
                );
                """
            )
            schema = db.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'runs'"
            ).fetchone()["sql"]
            if "run_dir TEXT NOT NULL UNIQUE" in schema:
                # Version 1 keyed runs by output directory. Preserve its rows
                # while allowing later executions to reuse an output location.
                db.executescript(
                    """
                    ALTER TABLE runs RENAME TO runs_v1;
                    CREATE TABLE runs (
                        run_id TEXT PRIMARY KEY,
                        experiment_name TEXT NOT NULL,
                        seed INTEGER NOT NULL,
                        run_dir TEXT NOT NULL,
                        config_json TEXT NOT NULL,
                        parent_run_id TEXT,
                        resume_from TEXT,
                        status TEXT NOT NULL,
                        error TEXT NOT NULL DEFAULT '',
                        model_path TEXT NOT NULL DEFAULT '',
                        started_at REAL NOT NULL,
                        finished_at REAL,
                        latest_metrics_json TEXT NOT NULL DEFAULT '{}'
                    );
                    INSERT INTO runs SELECT * FROM runs_v1;
                    DROP TABLE runs_v1;
                    """
                )

    def register_run(
        self, run_id: str, cfg: dict[str, Any], run_dir: Path, *, resume_from: str | Path | None = None
    ) -> str:
        """Create a run record once; subsequent calls preserve its identity/config."""
        now = time.time()
        parent = self.find_run_for_artifact(resume_from) if resume_from else None
        with self._connect() as db:
            existing = db.execute("SELECT run_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if existing is not None:
                return str(existing["run_id"])
            db.execute(
                """INSERT INTO runs (run_id, experiment_name, seed, run_dir, config_json, parent_run_id,
                   resume_from, status, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, cfg["experiment_name"], int(cfg["seed"]), str(run_dir),
                 json.dumps(cfg, sort_keys=True), parent, str(resume_from) if resume_from else None,
                 "running", now),
            )
        self.record_event(run_id, "run_started", {"run_dir": str(run_dir)})
        return run_id

    def find_run_for_artifact(self, artifact: str | Path | None) -> str | None:
        if artifact is None:
            return None
        path = Path(artifact).resolve()
        with self._connect() as db:
            row = db.execute(
                "SELECT run_id FROM runs WHERE ? LIKE run_dir || '%' "
                "ORDER BY length(run_dir) DESC, started_at DESC LIMIT 1",
                (str(path),),
            ).fetchone()
        return str(row["run_id"]) if row else None

    def update_run(self, run_id: str, *, status: str | None = None, error: str | None = None,
                   model_path: str | Path | None = None, metrics: dict[str, Any] | None = None) -> None:
        fields: list[str] = []
        values: list[Any] = []
        if status is not None:
            fields.append("status = ?")
            values.append(status)
            if status in {"completed", "failed", "stopped"}:
                fields.append("finished_at = ?")
                values.append(time.time())
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        if model_path is not None:
            fields.append("model_path = ?")
            values.append(str(model_path))
        if metrics is not None:
            fields.append("latest_metrics_json = ?")
            values.append(json.dumps(metrics, sort_keys=True))
        if not fields:
            return
        values.append(run_id)
        with self._connect() as db:
            db.execute(f"UPDATE runs SET {', '.join(fields)} WHERE run_id = ?", values)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["metrics"] = json.loads(data.pop("latest_metrics_json"))
        data["config"] = json.loads(data.pop("config_json"))
        return data

    def list_runs(self) -> list[dict[str, Any]]:
        """Return completed and active runs newest first, with their artifacts."""
        with self._connect() as db:
            rows = db.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
            artifacts = db.execute(
                "SELECT run_id, kind, path FROM run_artifacts ORDER BY created_at"
            ).fetchall()
        by_run: dict[str, list[dict[str, str]]] = {}
        for artifact in artifacts:
            by_run.setdefault(str(artifact["run_id"]), []).append(
                {"kind": str(artifact["kind"]), "path": str(artifact["path"])}
            )
        result = []
        for row in rows:
            data = dict(row)
            data["metrics"] = json.loads(data.pop("latest_metrics_json"))
            config = json.loads(data.pop("config_json"))
            data["algorithm"] = config.get("training", {}).get("algorithm", "PPO")
            data["environment_type"] = config.get("environment", {}).get("type")
            data["artifacts"] = by_run.get(data["run_id"], [])
            result.append(data)
        return result

    def record_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO run_events (run_id, created_at, event_type, payload_json) VALUES (?, ?, ?, ?)",
                       (run_id, time.time(), event_type, json.dumps(payload, sort_keys=True)))

    def enqueue_tuning(self, run_id: str, params: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO tuning_commands (run_id, created_at, params_json) VALUES (?, ?, ?)",
                       (run_id, time.time(), json.dumps(params, sort_keys=True)))
        self.record_event(run_id, "tuning_requested", params)

    def claim_tuning(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            rows = db.execute("SELECT command_id, params_json FROM tuning_commands WHERE run_id = ? AND status = 'pending' ORDER BY command_id", (run_id,)).fetchall()
            if not rows:
                return None
            ids = [row["command_id"] for row in rows]
            db.executemany("UPDATE tuning_commands SET status = 'applied', applied_at = ? WHERE command_id = ?", [(time.time(), command_id) for command_id in ids])
        merged: dict[str, Any] = {}
        for row in rows:
            merged.update(json.loads(row["params_json"]))
        self.record_event(run_id, "tuning_applied", merged)
        return merged

    def record_artifact(self, run_id: str, kind: str, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO run_artifacts (run_id, kind, path, created_at) VALUES (?, ?, ?, ?)",
                       (run_id, kind, str(path), time.time()))


def registry_for_config(cfg: dict[str, Any]) -> RunRegistry:
    return RunRegistry(cfg.get("output", {}).get("base_dir", "outputs"))


def new_run_id() -> str:
    """Return an opaque immutable identity for a non-orchestrated CLI run."""
    return f"run_{uuid.uuid4().hex}"
