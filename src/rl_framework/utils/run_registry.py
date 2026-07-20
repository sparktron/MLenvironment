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
                CREATE TABLE IF NOT EXISTS analysis_jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    params_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    finished_at REAL
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
        self,
        run_id: str,
        cfg: dict[str, Any],
        run_dir: Path,
        *,
        resume_from: str | Path | None = None,
    ) -> str:
        """Create a run record once; subsequent calls preserve its identity/config."""
        now = time.time()
        parent = self.find_run_for_artifact(resume_from) if resume_from else None
        with self._connect() as db:
            existing = db.execute(
                "SELECT run_id FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if existing is not None:
                return str(existing["run_id"])
            db.execute(
                """INSERT INTO runs (run_id, experiment_name, seed, run_dir, config_json, parent_run_id,
                   resume_from, status, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    cfg["experiment_name"],
                    int(cfg["seed"]),
                    str(run_dir),
                    json.dumps(cfg, sort_keys=True),
                    parent,
                    str(resume_from) if resume_from else None,
                    "running",
                    now,
                ),
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

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        error: str | None = None,
        model_path: str | Path | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
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
            row = db.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
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

    def record_event(
        self, run_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO run_events (run_id, created_at, event_type, payload_json) VALUES (?, ?, ?, ?)",
                (run_id, time.time(), event_type, json.dumps(payload, sort_keys=True)),
            )

    def enqueue_tuning(self, run_id: str, params: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO tuning_commands (run_id, created_at, params_json) VALUES (?, ?, ?)",
                (run_id, time.time(), json.dumps(params, sort_keys=True)),
            )
        self.record_event(run_id, "tuning_requested", params)

    def claim_tuning(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            rows = db.execute(
                "SELECT command_id, params_json FROM tuning_commands WHERE run_id = ? AND status = 'pending' ORDER BY command_id",
                (run_id,),
            ).fetchall()
            if not rows:
                return None
            ids = [row["command_id"] for row in rows]
            db.executemany(
                "UPDATE tuning_commands SET status = 'applied', applied_at = ? WHERE command_id = ?",
                [(time.time(), command_id) for command_id in ids],
            )
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
            db.execute(
                "INSERT OR IGNORE INTO run_artifacts (run_id, kind, path, created_at) VALUES (?, ?, ?, ?)",
                (run_id, kind, str(path), time.time()),
            )

    # ---- analysis jobs ------------------------------------------------
    # Replay and league-rating jobs run as GUI threads; their lifecycle is
    # persisted here so a restarted GUI can list past results and mark jobs
    # orphaned by a dead process instead of losing them silently.

    def create_analysis_job(
        self, job_id: str, kind: str, params: dict[str, Any] | None = None
    ) -> str:
        with self._connect() as db:
            db.execute(
                "INSERT INTO analysis_jobs (job_id, kind, params_json, status, created_at) VALUES (?, ?, ?, 'running', ?)",
                (job_id, kind, json.dumps(params or {}, sort_keys=True), time.time()),
            )
        return job_id

    def finish_analysis_job(
        self,
        job_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str = "",
    ) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE analysis_jobs SET status = ?, result_json = ?, error = ?, finished_at = ? WHERE job_id = ?",
                (
                    status,
                    json.dumps(result, sort_keys=True) if result is not None else None,
                    error,
                    time.time(),
                    job_id,
                ),
            )

    def get_analysis_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM analysis_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._analysis_job_row(row) if row else None

    def list_analysis_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM analysis_jobs ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._analysis_job_row(row) for row in rows]

    def recover_interrupted_analysis_jobs(self) -> int:
        """Mark jobs left 'running' by a previous process as interrupted.

        Call once at GUI startup: worker threads are process-local, so any
        job still marked running at that point can never complete.
        """
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE analysis_jobs SET status = 'interrupted', "
                "error = 'GUI restarted before the job finished', finished_at = ? "
                "WHERE status = 'running'",
                (time.time(),),
            )
        return cursor.rowcount

    @staticmethod
    def _analysis_job_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        params = data.pop("params_json")
        result = data.pop("result_json")
        data["params"] = json.loads(params) if params else {}
        data["result"] = json.loads(result) if result else None
        return data

    # ---- maintenance -------------------------------------------------
    # Inspect / export / prune helpers backing the `registry` CLI command.

    def summary(self) -> dict[str, Any]:
        """Return counts of runs and analysis jobs by status plus row totals."""
        with self._connect() as db:
            run_rows = db.execute(
                "SELECT status, COUNT(*) AS c FROM runs GROUP BY status"
            ).fetchall()
            job_rows = db.execute(
                "SELECT status, COUNT(*) AS c FROM analysis_jobs GROUP BY status"
            ).fetchall()
            artifact_rows = db.execute("SELECT path FROM run_artifacts").fetchall()

            def _count(table: str) -> int:
                return int(
                    db.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
                )

            return {
                "path": str(self.path),
                "runs_total": sum(int(r["c"]) for r in run_rows),
                "runs_by_status": {r["status"]: int(r["c"]) for r in run_rows},
                "analysis_jobs_total": sum(int(r["c"]) for r in job_rows),
                "analysis_jobs_by_status": {r["status"]: int(r["c"]) for r in job_rows},
                "run_events_total": _count("run_events"),
                "tuning_commands_total": _count("tuning_commands"),
                "run_artifacts_total": _count("run_artifacts"),
                "missing_artifacts_total": sum(
                    not Path(row["path"]).exists() for row in artifact_rows
                ),
            }

    def export(self) -> dict[str, Any]:
        """Return every registry table as lists of rows for backup/inspection."""
        tables = (
            "runs",
            "run_events",
            "tuning_commands",
            "run_artifacts",
            "analysis_jobs",
        )
        with self._connect() as db:
            return {
                table: [dict(row) for row in db.execute(f"SELECT * FROM {table}")]
                for table in tables
            }

    @staticmethod
    def _prune_where(
        age_column: str,
        statuses: list[str] | None,
        older_than: float | None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if statuses:
            clauses.append(f"status IN ({','.join('?' * len(statuses))})")
            params.extend(statuses)
        if older_than is not None:
            clauses.append(f"{age_column} < ?")
            params.append(older_than)
        return (f" WHERE {' AND '.join(clauses)}" if clauses else "", params)

    def _prune(
        self,
        table: str,
        id_column: str,
        age_column: str,
        *,
        statuses: list[str] | None,
        older_than: float | None,
        dry_run: bool,
    ) -> list[str]:
        where, params = self._prune_where(age_column, statuses, older_than)
        with self._connect() as db:
            ids = [
                str(row[id_column])
                for row in db.execute(
                    f"SELECT {id_column} FROM {table}{where} ORDER BY {id_column}",
                    params,
                ).fetchall()
            ]
            if ids and not dry_run:
                db.execute(f"DELETE FROM {table}{where}", params)
        return ids

    def prune_analysis_jobs(
        self,
        *,
        statuses: list[str] | None = None,
        older_than: float | None = None,
        dry_run: bool = False,
    ) -> list[str]:
        """Delete analysis jobs matching *statuses* and/or age.

        *older_than* is a unix timestamp compared against each job's
        ``finished_at`` (falling back to ``created_at`` for jobs that never
        finished). Returns the matched job ids; with ``dry_run`` nothing is
        deleted but the ids are still returned for preview.
        """
        return self._prune(
            "analysis_jobs",
            "job_id",
            "COALESCE(finished_at, created_at)",
            statuses=statuses,
            older_than=older_than,
            dry_run=dry_run,
        )

    def prune_runs(
        self,
        *,
        statuses: list[str] | None = None,
        older_than: float | None = None,
        dry_run: bool = False,
    ) -> list[str]:
        """Delete run records (and their events/tuning/artifact rows) by
        status and/or age.

        *older_than* compares against ``finished_at`` (falling back to
        ``started_at`` for still-running rows). This removes only registry
        rows; artifact files on disk are left untouched. Returns the matched
        run ids.
        """
        where, params = self._prune_where(
            "COALESCE(finished_at, started_at)",
            statuses,
            older_than,
        )
        with self._connect() as db:
            run_ids = [
                str(row["run_id"])
                for row in db.execute(
                    f"SELECT run_id FROM runs{where} ORDER BY run_id", params
                ).fetchall()
            ]
            if run_ids and not dry_run:
                placeholders = ",".join("?" * len(run_ids))
                for table in ("run_events", "tuning_commands", "run_artifacts"):
                    db.execute(
                        f"DELETE FROM {table} WHERE run_id IN ({placeholders})",
                        run_ids,
                    )
                db.execute(
                    f"DELETE FROM runs WHERE run_id IN ({placeholders})", run_ids
                )
        return run_ids

    def prune_artifacts(
        self,
        *,
        older_than: float | None = None,
        missing_only: bool = False,
        dry_run: bool = False,
    ) -> list[str]:
        """Delete matching artifact index rows, never the artifact files.

        ``missing_only`` limits the operation to paths that no longer exist.
        ``older_than`` is a unix timestamp compared with ``created_at``. The
        returned identifiers include the run and kind because artifact paths
        alone are not guaranteed to be unique across registry records.
        """
        with self._connect() as db:
            if older_than is None:
                rows = db.execute(
                    "SELECT run_id, kind, path FROM run_artifacts "
                    "ORDER BY run_id, kind, path"
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT run_id, kind, path FROM run_artifacts "
                    "WHERE created_at < ? ORDER BY run_id, kind, path",
                    (older_than,),
                ).fetchall()
            matched = [
                row
                for row in rows
                if not missing_only or not Path(row["path"]).exists()
            ]
            if matched and not dry_run:
                db.executemany(
                    "DELETE FROM run_artifacts "
                    "WHERE run_id = ? AND kind = ? AND path = ?",
                    [tuple(row) for row in matched],
                )
        return [
            f"{row['run_id']}:{row['kind']}:{row['path']}" for row in matched
        ]

    def metric_history(
        self, run_id: str, keys: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return this run's persisted metric snapshots oldest-first.

        Each training metrics push is stored as a ``metrics`` run event; this
        replays them as ``{"t": <created_at>, <metric>: <value>, ...}`` points.
        When *keys* is given, each point is narrowed to those metric keys (plus
        the timestamp), so callers can chart a specific series cheaply.
        """
        with self._connect() as db:
            rows = db.execute(
                "SELECT created_at, payload_json FROM run_events "
                "WHERE run_id = ? AND event_type = 'metrics' ORDER BY event_id",
                (run_id,),
            ).fetchall()
        history: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            if keys is not None:
                payload = {k: payload[k] for k in keys if k in payload}
            history.append({"t": float(row["created_at"]), **payload})
        return history


def registry_for_config(cfg: dict[str, Any]) -> RunRegistry:
    return RunRegistry(cfg.get("output", {}).get("base_dir", "outputs"))


def new_run_id() -> str:
    """Return an opaque immutable identity for a non-orchestrated CLI run."""
    return f"run_{uuid.uuid4().hex}"
