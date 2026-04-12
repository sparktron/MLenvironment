"""Background training manager for the GUI.

Runs training in a separate thread, exposes status via JSON files, and accepts
live parameter changes from the GUI through a shared tuning file.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class _StopOnEvent(BaseCallback):
    """SB3 callback that stops training when a threading.Event is set."""

    def __init__(self, stop_event: threading.Event) -> None:
        super().__init__(verbose=0)
        self._stop_event = stop_event

    def _on_step(self) -> bool:
        return not self._stop_event.is_set()

from rl_framework.training.sb3_runner import train as _train_core
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
        """Request a run to stop (sets status so the GUI knows; actual SB3 interruption
        requires the thread to finish its current rollout)."""
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
        import copy

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
            # Inject LiveTuningCallback into the config so the train function picks it up.
            cfg["_live_tuning"] = {
                "tuning_file": str(tuning_file),
                "status_file": str(status_file),
            }
            model_path = _train_with_live_tuning(cfg, state.stop_event)
            with self._lock:
                state.status = "completed"
                state.model_path = str(model_path)
                state.finished_at = time.time()
        except Exception:
            with self._lock:
                state.status = "failed"
                state.error = traceback.format_exc()
                state.finished_at = time.time()


def _train_with_live_tuning(cfg: dict[str, Any], stop_event: threading.Event | None = None) -> Path:
    """Extended train() that injects the LiveTuningCallback."""
    import supersuit as ss
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    from rl_framework.envs.registry import make_env
    from rl_framework.training.curriculum_callback import CurriculumCallback
    from rl_framework.training.live_tuning_callback import LiveTuningCallback
    from rl_framework.training.self_play_callback import SelfPlayCallback
    from rl_framework.utils.logging_utils import create_experiment_paths

    paths = create_experiment_paths(cfg["output"]["base_dir"], cfg["experiment_name"], cfg["seed"])
    env_cfg = cfg["environment"]
    num_envs = int(cfg["training"].get("num_envs", 1))

    if env_cfg["type"] == "organism_arena_parallel":
        par_env = make_env(env_cfg["type"], env_cfg)
        vec_env = ss.pettingzoo_env_to_vec_env_v1(par_env)
        vec_env = ss.concat_vec_envs_v1(
            vec_env, max(num_envs, 1), num_cpus=max(num_envs, 1), base_class="stable_baselines3",
        )
    else:
        env_fns = [lambda: make_env(env_cfg["type"], env_cfg) for _ in range(max(num_envs, 1))]
        if num_envs > 1:
            vec_env = SubprocVecEnv(env_fns)
        else:
            vec_env = DummyVecEnv(env_fns)

    try:
        if cfg["training"].get("normalize_observations", True):
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

        model = PPO(
            policy=cfg["training"].get("policy", "MlpPolicy"),
            env=vec_env,
            learning_rate=cfg["training"].get("learning_rate", 3e-4),
            n_steps=cfg["training"].get("n_steps", 1024),
            batch_size=cfg["training"].get("batch_size", 256),
            tensorboard_log=str(paths.logs_dir),
            seed=cfg["seed"],
            verbose=1,
        )

        checkpoint_cb = CheckpointCallback(
            save_freq=cfg["training"].get("checkpoint_every", 10000),
            save_path=str(paths.checkpoints_dir),
            name_prefix="ppo_model",
        )
        callbacks = [checkpoint_cb]

        # Stop-on-request callback (set by stop_run()).
        if stop_event is not None:
            callbacks.append(_StopOnEvent(stop_event))

        # Live tuning callback.
        live_cfg = cfg.get("_live_tuning", {})
        if live_cfg:
            callbacks.append(LiveTuningCallback(
                tuning_file=live_cfg["tuning_file"],
                env_cfg=env_cfg,
                status_file=live_cfg.get("status_file"),
                verbose=1,
            ))

        # Curriculum learning.
        curriculum_cfg = cfg.get("curriculum", {})
        if curriculum_cfg.get("enabled", False):
            callbacks.append(CurriculumCallback(curriculum_cfg, env_cfg, verbose=1))

        # Self-play.
        self_play_cfg = cfg.get("self_play", {})
        if self_play_cfg.get("enabled", False) and env_cfg["type"] == "organism_arena_parallel":
            callbacks.append(SelfPlayCallback(
                snapshot_dir=paths.checkpoints_dir / "league",
                snapshot_freq=int(self_play_cfg.get("snapshot_freq", 5000)),
                max_league_size=int(self_play_cfg.get("max_league_size", 10)),
                sampling_mode=str(self_play_cfg.get("sampling_mode", "uniform")),
                recent_bias_alpha=float(self_play_cfg.get("recent_bias_alpha", 1.0)),
                verbose=1,
            ))

        model.learn(total_timesteps=cfg["training"]["total_timesteps"], callback=callbacks)
        final_path = paths.checkpoints_dir / "final_model"
        model.save(str(final_path))
        if isinstance(vec_env, VecNormalize):
            vec_env.save(str(paths.checkpoints_dir / "vecnormalize.pkl"))
        return final_path
    finally:
        vec_env.close()
