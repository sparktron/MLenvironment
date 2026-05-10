"""Background training manager for the GUI."""
from __future__ import annotations

import atexit
import copy
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from rl_framework.utils.config import validate_experiment_config

if TYPE_CHECKING:
    from rl_framework.training.frame_capture_callback import FrameCaptureCallback


@dataclass
class _RunState:
    run_id: str
    cfg: dict[str, Any]
    status: str = "pending"  # pending | running | stopping | completed | failed
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    model_path: str = ""
    pending_tuning_events: list[dict[str, Any]] = field(default_factory=list)
    latest_metrics: dict[str, Any] = field(default_factory=dict)
    stop_event: threading.Event = field(default_factory=threading.Event)
    frame_capture_callback: Optional["FrameCaptureCallback"] = None


class TrainingManager:
    """Manages background training runs, one at a time."""

    def __init__(self) -> None:
        self._runs: dict[str, _RunState] = {}
        self._lock = threading.Lock()
        self._active_thread: threading.Thread | None = None
        atexit.register(self._shutdown)

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

        thread = threading.Thread(target=self._train_worker, args=(state,), daemon=False)
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
        """Return the status of a run, including latest streamed metrics."""
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                return {"error": f"Unknown run_id: {run_id}"}
            return {
                "run_id": state.run_id,
                "status": state.status,
                "error": state.error,
                "model_path": state.model_path,
                "started_at": state.started_at,
                "finished_at": state.finished_at,
                "metrics": copy.deepcopy(state.latest_metrics),
            }

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"run_id": s.run_id, "status": s.status, "experiment": s.cfg.get("experiment_name", "")}
                for s in self._runs.values()
            ]

    def apply_tuning(self, run_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Queue live parameter changes for the callback to apply atomically."""
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                return {"error": f"Unknown run_id: {run_id}"}
            if state.status != "running":
                return {"error": "Run is not active"}
            state.pending_tuning_events.append(copy.deepcopy(params))
        return {"applied": True, "params": params}

    def get_frames(self, run_id: str, since: int = 0) -> dict[str, Any]:
        """Return captured frames for a run with frame_index >= since.

        Callers pass the last seen frame_index + 1 so each response only
        contains frames they haven't processed yet, keeping payload size small.
        """
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                return {"error": f"Unknown run_id: {run_id}"}
            if state.frame_capture_callback is None:
                return {"frames": []}
        # get_frames acquires its own lock; release ours first to avoid nesting.
        frames = state.frame_capture_callback.get_frames(since=since)
        return {"frames": frames}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        """Signal any active run to stop and wait up to 30 s for the thread to finish.

        Registered via ``atexit`` so mid-checkpoint writes complete before the
        interpreter exits, avoiding partial checkpoint files.
        """
        with self._lock:
            for state in self._runs.values():
                if state.status == "running":
                    state.status = "stopping"
                    state.stop_event.set()
        thread = self._active_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=30.0)

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _train_worker(self, state: _RunState) -> None:
        """Run training in a background thread."""
        # Lazy imports so the heavy ML stack is only loaded when a run actually starts.
        from rl_framework.training.frame_capture_callback import FrameCaptureCallback
        from rl_framework.training.live_tuning_callback import LiveTuningCallback
        from rl_framework.training.sb3_runner import train

        cfg = copy.deepcopy(state.cfg)
        # Enable rendering for frame capture
        cfg.setdefault("environment", {})["render_mode"] = "rgb_array"

        with self._lock:
            state.status = "running"
            state.started_at = time.time()
            state.latest_metrics = {}
            state.pending_tuning_events = []

        try:
            live_cb = LiveTuningCallback(
                env_cfg=cfg["environment"],
                pop_tuning_event=lambda: self._pop_tuning_event(state.run_id),
                publish_status=lambda payload: self._publish_status(state.run_id, payload),
                verbose=1,
            )
            frame_cb = FrameCaptureCallback(capture_interval=50, max_frames=200, verbose=1)

            with self._lock:
                state.frame_capture_callback = frame_cb

            model_path = train(cfg, extra_callbacks=[live_cb, frame_cb], stop_event=state.stop_event)
            with self._lock:
                state.status = "completed"
                state.model_path = str(model_path)
                state.finished_at = time.time()
        except Exception:
            with self._lock:
                state.status = "failed"
                state.error = traceback.format_exc()
                state.finished_at = time.time()

    def _pop_tuning_event(self, run_id: str) -> dict[str, Any] | None:
        """Atomically consume pending tuning events for *run_id*.

        Multiple queued updates are merged so later values win for the same key.
        """
        with self._lock:
            state = self._runs.get(run_id)
            if state is None or not state.pending_tuning_events:
                return None
            merged: dict[str, Any] = {}
            for event in state.pending_tuning_events:
                if isinstance(event, dict):
                    merged.update(event)
            state.pending_tuning_events.clear()
            return merged or None

    def _publish_status(self, run_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            state = self._runs.get(run_id)
            if state is None:
                return
            state.latest_metrics = copy.deepcopy(payload)
