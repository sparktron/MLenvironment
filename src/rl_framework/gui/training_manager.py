"""Background training manager for the GUI.

Runs training in a separate thread, exposes status via JSON files, and accepts
live parameter changes from the GUI through a shared tuning file.
"""
from __future__ import annotations

import copy
import json
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rl_framework.utils.config import validate_experiment_config


@dataclass
class _RunState:
    run_id: str
    cfg: dict[str, Any]
    status: str = "pending"  # pending | running | stopping | completed | failed
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    model_path: str = ""
    tuning_file: str = ""
    status_file: str = ""
    stop_event: threading.Event = field(default_factory=threading.Event)


class TrainingManager:
    """Manages background training runs, one at a time."""

    def __init__(self) -> None:
        self._runs: dict[str, _RunState] = {}
        self._lock = threading.Lock()
        self._active_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(self, run_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
        """Validate config and start training in a background thread."""
        validate_experiment_config(cfg)

        with self._lock:
            if any(r.status == "running" for r in self._runs.values()):
                return {"error": "A training run is already active. Stop it first."}

            state = _RunState(run_id=run_id, cfg=cfg)
            self._runs[run_id] = state

        thread = threading.Thread(target=self._train_worker, args=(state,), daemon=True)
        self._active_thread = thread
        thread.start()
        return {"run_id": run_id, "status": "started"}

    def stop_run(self, run_id: str) -> dict[str, Any]:
        """Request a run to stop.

        Sets a threading.Event consumed by :class:`StopOnEvent` inside SB3's
        training loop.  Training halts at the end of the current rollout.
        """
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                return {"error": f"Unknown run_id: {run_id}"}
            if state.status != "running":
                return {"error": f"Run is not active (status={state.status})"}
            state.status = "stopping"
            state.stop_event.set()
        return {"run_id": run_id, "status": "stopping"}

    def get_status(self, run_id: str) -> dict[str, Any]:
        """Return the status of a run, including live metrics from the status file."""
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                return {"error": f"Unknown run_id: {run_id}"}
            result: dict[str, Any] = {
                "run_id": state.run_id,
                "status": state.status,
                "error": state.error,
                "model_path": state.model_path,
                "started_at": state.started_at,
                "finished_at": state.finished_at,
            }

        # Read live metrics from the status file written by LiveTuningCallback.
        if state.status_file:
            try:
                raw = Path(state.status_file).read_text(encoding="utf-8").strip()
                if raw:
                    result["metrics"] = json.loads(raw)
            except (OSError, json.JSONDecodeError):
                pass

        return result

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"run_id": s.run_id, "status": s.status, "experiment": s.cfg.get("experiment_name", "")}
                for s in self._runs.values()
            ]

    def apply_tuning(self, run_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Write live parameter changes for a running experiment."""
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                return {"error": f"Unknown run_id: {run_id}"}
            if state.status != "running":
                return {"error": "Run is not active"}
            tuning_file = state.tuning_file
        if not tuning_file:
            return {"error": "Tuning file not configured for this run"}
        try:
            Path(tuning_file).write_text(json.dumps(params), encoding="utf-8")
        except OSError as exc:
            return {"error": str(exc)}
        return {"applied": True, "params": params}

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _train_worker(self, state: _RunState) -> None:
        """Run training in a background thread."""
        # Lazy imports so the heavy ML stack is only loaded when a run actually starts.
        from rl_framework.training.live_tuning_callback import LiveTuningCallback
        from rl_framework.training.sb3_runner import train

        cfg = copy.deepcopy(state.cfg)
        base_dir = Path(cfg["output"]["base_dir"]) / cfg["experiment_name"] / f"seed_{cfg['seed']}"
        base_dir.mkdir(parents=True, exist_ok=True)

        tuning_file = base_dir / "live_tuning.json"
        status_file = base_dir / "live_status.json"
        tuning_file.write_text("", encoding="utf-8")
        status_file.write_text("", encoding="utf-8")

        with self._lock:
            state.tuning_file = str(tuning_file)
            state.status_file = str(status_file)
            state.status = "running"
            state.started_at = time.time()

        try:
            live_cb = LiveTuningCallback(
                tuning_file=tuning_file,
                env_cfg=cfg["environment"],
                status_file=status_file,
                verbose=1,
            )
            model_path = train(cfg, extra_callbacks=[live_cb], stop_event=state.stop_event)
            with self._lock:
                state.status = "completed"
                state.model_path = str(model_path)
                state.finished_at = time.time()
        except Exception:
            with self._lock:
                state.status = "failed"
                state.error = traceback.format_exc()
                state.finished_at = time.time()
