"""Flask web application for the RL Experiment GUI."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import yaml
from flask import Flask, abort, jsonify, render_template, request

from rl_framework.gui.training_manager import TrainingManager
from rl_framework.utils.config import validate_experiment_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "experiments"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)


def _is_safe_config_name(name: str) -> bool:
    """Reject config names that could escape CONFIGS_DIR via path traversal."""
    return bool(name) and ".." not in name and "/" not in name and "\\" not in name


app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = (
    5 * 1024 * 1024
)  # 5 MB — guard against oversized JSON payloads
manager = TrainingManager()


# ------------------------------------------------------------------
# CSRF protection — lightweight Origin/Referer check (no new deps).
# Rejects state-changing requests (non-GET/HEAD) whose Origin or
# Referer header does not match the server's own host:port, blocking
# cross-site form/fetch attacks from other origins.
# ------------------------------------------------------------------


@app.before_request
def _csrf_origin_check() -> None:
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    server_host = request.host  # e.g. "127.0.0.1:5001"
    origin = request.headers.get("Origin")
    referer = request.headers.get("Referer")
    # Prefer Origin (exact); fall back to Referer (URL prefix).
    if origin is not None:
        # Origin is scheme+host[:port], e.g. "http://127.0.0.1:5001"
        from urllib.parse import urlparse

        parsed = urlparse(origin)
        if parsed.netloc != server_host:
            abort(403)
    elif referer is not None:
        from urllib.parse import urlparse

        parsed = urlparse(referer)
        if parsed.netloc != server_host:
            abort(403)
    # If neither header is present (e.g. same-origin curl / non-browser
    # clients), allow through — the GUI is a local tool, not a public service.


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


# ------------------------------------------------------------------
# Config API
# ------------------------------------------------------------------


@app.route("/api/configs", methods=["GET"])
def list_configs():
    """List available experiment YAML configs."""
    configs = []
    for p in sorted(CONFIGS_DIR.glob("*.yaml")):
        try:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            configs.append(
                {
                    "name": p.stem,
                    "experiment_name": data.get("experiment_name", p.stem),
                    "env_type": data.get("environment", {}).get("type", "unknown"),
                }
            )
        except Exception:
            configs.append(
                {"name": p.stem, "experiment_name": p.stem, "env_type": "unknown"}
            )
    return jsonify(configs)


@app.route("/api/configs/<name>", methods=["GET"])
def get_config(name: str):
    """Return the full parsed YAML config."""
    if not _is_safe_config_name(name):
        return jsonify({"error": "Invalid config name"}), 400
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        return jsonify({"error": f"Config not found: {name}"}), 404
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return jsonify(data)


@app.route("/api/configs/<name>", methods=["PUT"])
def save_config(name: str):
    """Save a modified config back to YAML."""
    if not _is_safe_config_name(name):
        return jsonify({"error": "Invalid config name"}), 400
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Empty payload"}), 400
    try:
        validate_experiment_config(data)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Validation error: {exc}"}), 400
    path = CONFIGS_DIR / f"{name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return jsonify({"saved": name})


@app.route("/api/configs", methods=["POST"])
def create_config():
    """Create a new config from the wizard."""
    data = request.get_json(force=True)
    if not data or "experiment_name" not in data:
        return jsonify({"error": "experiment_name is required"}), 400
    name = data["experiment_name"].replace(" ", "_").lower()
    # Reject unsafe names.
    if ".." in name or "/" in name or "\\" in name:
        return jsonify({"error": "Invalid experiment name"}), 400
    try:
        validate_experiment_config(data)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Validation error: {exc}"}), 400
    path = CONFIGS_DIR / f"{name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return jsonify({"saved": name, "path": path.name})


# ------------------------------------------------------------------
# Schema API (for the wizard to know what fields are available)
# ------------------------------------------------------------------


@app.route("/api/schema", methods=["GET"])
def get_schema():
    """Return the config schema with defaults and descriptions for the wizard."""
    schema = {
        "walker_bullet": {
            "environment": {
                "type": {"value": "walker_bullet", "type": "fixed"},
                "seed": {"value": 42, "type": "int", "desc": "Environment random seed"},
                "sim": {
                    "gravity": {
                        "value": -9.81,
                        "type": "float",
                        "desc": "Gravity (m/s^2)",
                        "min": -20,
                        "max": 0,
                    },
                    "mass": {
                        "value": 28.0,
                        "type": "float",
                        "desc": "Torso mass (kg) — Atlas DRC: 28",
                        "min": 0.1,
                        "max": 200,
                    },
                    "friction": {
                        "value": 0.9,
                        "type": "float",
                        "desc": "Ground friction",
                        "min": 0.0,
                        "max": 2.0,
                    },
                    "max_force": {
                        "value": 35.0,
                        "type": "float",
                        "desc": "Global torque scale (legacy; 35 = 1.0× of per-joint Atlas caps)",
                        "min": 1,
                        "max": 500,
                    },
                    "timestep": {
                        "value": 1.0 / 240.0,
                        "type": "float",
                        "desc": "Physics timestep (s)",
                        "min": 0.0005,
                        "max": 0.05,
                    },
                    "frame_skip": {
                        "value": 4,
                        "type": "int",
                        "desc": "Physics ticks per agent step",
                        "min": 1,
                        "max": 16,
                    },
                    "settle_steps": {
                        "value": 30,
                        "type": "int",
                        "desc": "Sim ticks holding rest pose after reset",
                        "min": 0,
                        "max": 240,
                    },
                    "control": {
                        "mode": {
                            "value": "pd",
                            "type": "choice",
                            "choices": ["pd", "torque"],
                            "desc": "Actuator mode: PD position targets or raw torque",
                        },
                        "position_gain": {
                            "value": 0.1,
                            "type": "float",
                            "desc": "PD position gain (Kp)",
                            "min": 0.0,
                            "max": 10.0,
                        },
                        "velocity_gain": {
                            "value": 1.0,
                            "type": "float",
                            "desc": "PD velocity gain (Kd)",
                            "min": 0.0,
                            "max": 10.0,
                        },
                    },
                },
                "reward": {
                    "alive_bonus": {
                        "value": 5.0,
                        "type": "float",
                        "desc": "Bonus per step while alive (dominant)",
                        "min": 0,
                        "max": 20,
                    },
                    "forward_velocity_weight": {
                        "value": 1.5,
                        "type": "float",
                        "desc": "Peak weight of Gaussian velocity reward",
                        "min": 0,
                        "max": 20,
                    },
                    "target_velocity": {
                        "value": 1.0,
                        "type": "float",
                        "desc": "Target forward velocity (m/s)",
                        "min": 0,
                        "max": 5,
                    },
                    "velocity_sigma": {
                        "value": 0.5,
                        "type": "float",
                        "desc": "Sigma of velocity Gaussian (m/s)",
                        "min": 0.05,
                        "max": 5.0,
                    },
                    "orientation_penalty_weight": {
                        "value": 0.3,
                        "type": "float",
                        "desc": "Per-step tilt penalty (|roll|+|pitch|)",
                        "min": 0,
                        "max": 10,
                    },
                    "torque_penalty_weight": {
                        "value": 0.01,
                        "type": "float",
                        "desc": "Per-step torque-magnitude penalty",
                        "min": 0,
                        "max": 1,
                    },
                    "fall_penalty": {
                        "value": 10.0,
                        "type": "float",
                        "desc": "One-time penalty on fall (terminal step)",
                        "min": 0,
                        "max": 100,
                    },
                },
                "termination": {
                    "min_height": {
                        "value": 0.18,
                        "type": "float",
                        "desc": "Torso COM height below which fall is detected (m)",
                        "min": 0,
                        "max": 1,
                    },
                    "max_height": {
                        "value": 1.5,
                        "type": "float",
                        "desc": "Torso COM height above which 'flying' is detected (m)",
                        "min": 0.5,
                        "max": 10,
                    },
                    "max_steps": {
                        "value": 800,
                        "type": "int",
                        "desc": "Max steps per episode (truncation)",
                        "min": 50,
                        "max": 10000,
                    },
                },
                "reset_randomization": {
                    "position_xy_noise": {
                        "value": 0.02,
                        "type": "float",
                        "desc": "XY position noise at reset",
                        "min": 0,
                        "max": 1,
                    },
                    "yaw_noise": {
                        "value": 0.08,
                        "type": "float",
                        "desc": "Yaw noise at reset (rad)",
                        "min": 0,
                        "max": 1,
                    },
                },
                "domain_randomization": {
                    "mass_scale_range": {
                        "value": [0.95, 1.05],
                        "type": "range",
                        "desc": "Mass scale randomization [lo, hi]",
                        "min": 0.5,
                        "max": 2.0,
                    },
                    "friction_range": {
                        "value": [0.9, 1.1],
                        "type": "range",
                        "desc": "Friction randomization [lo, hi]",
                        "min": 0.1,
                        "max": 3.0,
                    },
                    "sensor_noise_std": {
                        "value": 0.0,
                        "type": "float",
                        "desc": "Gaussian sensor noise std",
                        "min": 0,
                        "max": 1,
                    },
                    "action_latency_steps": {
                        "value": 0,
                        "type": "int",
                        "desc": "Action latency (steps)",
                        "min": 0,
                        "max": 10,
                    },
                },
            },
            "training": _training_schema("walker_bullet"),
            "evaluation": _eval_schema(),
        },
        "organism_arena_parallel": {
            "environment": {
                "type": {"value": "organism_arena_parallel", "type": "fixed"},
                "seed": {"value": 11, "type": "int", "desc": "Environment random seed"},
                "num_agents": {
                    "value": 2,
                    "type": "int",
                    "desc": "Number of organisms (2 = duel, >2 = free-for-all)",
                    "min": 2,
                    "max": 8,
                },
                "sim": {
                    "arena_half_extent": {
                        "value": 1.2,
                        "type": "float",
                        "desc": "Arena half-size",
                        "min": 0.5,
                        "max": 5,
                    },
                    "move_speed": {
                        "value": 0.05,
                        "type": "float",
                        "desc": "Max movement per step (arena units)",
                        "min": 0.01,
                        "max": 0.5,
                    },
                    "spawn_jitter": {
                        "value": 0.1,
                        "type": "float",
                        "desc": "Spawn position jitter half-width (0 = fixed spawns)",
                        "min": 0,
                        "max": 1,
                    },
                },
                "morphology": {
                    "base_size": {
                        "value": 1.0,
                        "type": "float",
                        "desc": "Base organism size",
                        "min": 0.1,
                        "max": 5,
                    },
                    "episode_growth_scale": {
                        "value": 0.0,
                        "type": "float",
                        "desc": "Size growth per step (scales damage and max health)",
                        "min": 0,
                        "max": 0.1,
                    },
                    "health": {
                        "value": 1.2,
                        "type": "float",
                        "desc": "Health pool (scaled by size each step)",
                        "min": 0.1,
                        "max": 10,
                    },
                },
                "battle_rules": {
                    "damage": {
                        "value": 0.06,
                        "type": "float",
                        "desc": "Damage per hit",
                        "min": 0.01,
                        "max": 1,
                    },
                    "attack_range": {
                        "value": 0.2,
                        "type": "float",
                        "desc": "Attack range",
                        "min": 0.05,
                        "max": 2,
                    },
                    "cooldown_steps": {
                        "value": 3,
                        "type": "int",
                        "desc": "Cooldown between attacks",
                        "min": 0,
                        "max": 20,
                    },
                    "max_steps": {
                        "value": 400,
                        "type": "int",
                        "desc": "Max steps per episode",
                        "min": 50,
                        "max": 10000,
                    },
                    "win_health_threshold": {
                        "value": 0.0,
                        "type": "float",
                        "desc": "Health below which opponent loses",
                        "min": 0,
                        "max": 1,
                    },
                },
            },
            "training": _training_schema("organism_arena_parallel"),
            "evaluation": _eval_schema(),
        },
    }
    return jsonify(schema)


def _training_schema(env_type: str) -> dict[str, Any]:
    schema = {
        "policy": {
            "value": "MlpPolicy",
            "type": "choice",
            "choices": ["MlpPolicy", "CnnPolicy"],
            "desc": "Policy network type",
        },
        "total_timesteps": {
            "value": 2_000_000,
            "type": "int",
            "desc": "Total training timesteps",
            "min": 1000,
            "max": 50_000_000,
        },
        "num_envs": {
            "value": 8,
            "type": "int",
            "desc": "Parallel environments (SubprocVecEnv)",
            "min": 1,
            "max": 32,
        },
        "n_steps": {
            "value": 2048,
            "type": "int",
            "desc": "Steps per env per rollout",
            "min": 16,
            "max": 8192,
        },
        "batch_size": {
            "value": 512,
            "type": "int",
            "desc": "Minibatch size (must divide n_steps × num_envs)",
            "min": 8,
            "max": 8192,
        },
        "n_epochs": {
            "value": 10,
            "type": "int",
            "desc": "PPO update epochs per rollout",
            "min": 1,
            "max": 50,
        },
        "learning_rate": {
            "value": 0.0003,
            "type": "float",
            "desc": "Initial learning rate",
            "min": 0.000001,
            "max": 0.1,
        },
        "learning_rate_end": {
            "value": 0.00005,
            "type": "float",
            "desc": "Final LR (linear decay; omit for constant)",
            "min": 0.0,
            "max": 0.1,
        },
        "gamma": {
            "value": 0.99,
            "type": "float",
            "desc": "Discount factor",
            "min": 0.5,
            "max": 0.9999,
        },
        "gae_lambda": {
            "value": 0.95,
            "type": "float",
            "desc": "GAE-λ",
            "min": 0.5,
            "max": 1.0,
        },
        "clip_range": {
            "value": 0.2,
            "type": "float",
            "desc": "PPO clip range",
            "min": 0.05,
            "max": 1.0,
        },
        "ent_coef": {
            "value": 0.005,
            "type": "float",
            "desc": "Entropy bonus coefficient",
            "min": 0.0,
            "max": 0.1,
        },
        "vf_coef": {
            "value": 0.5,
            "type": "float",
            "desc": "Value-function loss coefficient",
            "min": 0.0,
            "max": 5.0,
        },
        "max_grad_norm": {
            "value": 0.5,
            "type": "float",
            "desc": "Gradient clipping norm",
            "min": 0.1,
            "max": 5.0,
        },
        "checkpoint_every": {
            "value": 50000,
            "type": "int",
            "desc": "Save checkpoint every N environment steps",
            "min": 100,
            "max": 1000000,
        },
        "normalize_observations": {
            "value": True,
            "type": "bool",
            "desc": "Normalize observations with VecNormalize",
        },
        "check_nans": {
            "value": False,
            "type": "bool",
            "desc": "Fail fast on NaN/Inf observations, rewards, or actions",
        },
        "device": {
            "value": "auto",
            "type": "choice",
            "choices": ["auto", "cpu", "cuda", "cuda:0"],
            "desc": "Training device",
        },
    }
    if env_type == "organism_arena_parallel":
        schema["num_envs"].update(
            {
                "value": 1,
                "desc": "Parallel environments (arena requires single-process training)",
                "max": 1,
            }
        )
    return schema


def _eval_schema() -> dict[str, Any]:
    return {
        "episodes": {
            "value": 5,
            "type": "int",
            "desc": "Evaluation episodes",
            "min": 1,
            "max": 100,
        },
    }


# ------------------------------------------------------------------
# Training API
# ------------------------------------------------------------------


@app.route("/api/train/start", methods=["POST"])
def start_training():
    """Start a training run with the given config."""
    cfg = request.get_json(force=True)
    if not cfg:
        return jsonify({"error": "Empty config"}), 400
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    result = manager.start_run(run_id, cfg)
    if "error" in result:
        # 409 for "already running"; 400 for bad config / validation errors.
        already_running = "already active" in result["error"]
        return jsonify(result), 409 if already_running else 400
    return jsonify(result)


@app.route("/api/train/stop/<run_id>", methods=["POST"])
def stop_training(run_id: str):
    result = manager.stop_run(run_id)
    if "error" in result:
        status = 404 if "Unknown run_id" in result["error"] else 409
        return jsonify(result), status
    return jsonify(result)


@app.route("/api/train/status/<run_id>", methods=["GET"])
def training_status(run_id: str):
    result = manager.get_status(run_id)
    # get_status includes an "error" field in normal payloads (for failed runs),
    # so only treat responses without run metadata as unknown run IDs.
    if "run_id" not in result:
        return jsonify(result), 404
    return jsonify(result)


@app.route("/api/train/runs", methods=["GET"])
def list_runs():
    return jsonify(manager.list_runs())


@app.route("/api/train/tune/<run_id>", methods=["POST"])
def tune_params(run_id: str):
    """Apply live parameter changes to a running experiment."""
    params = request.get_json(force=True)
    if not params:
        return jsonify({"error": "Empty params"}), 400
    result = manager.apply_tuning(run_id, params)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/train/frames/<run_id>", methods=["GET"])
def get_frames(run_id: str):
    """Get captured frames from a training run.

    Query params:
        since (int, default 0): only return frames with frame_index >= since.
            Pass the last seen frame_index + 1 to receive only new frames.
    """
    try:
        since = int(request.args.get("since", 0))
    except (TypeError, ValueError):
        since = 0
    result = manager.get_frames(run_id, since=since)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


# ------------------------------------------------------------------
# Outputs API
# ------------------------------------------------------------------

_DEFAULT_OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "outputs"

# Sidecar suffix mirrors VECNORM_SUFFIX in training.self_play_env_wrapper;
# hardcoded here to keep the GUI module free of a heavy training import.
_VECNORM_SUFFIX = "_vecnorm.pkl"


def _safe_output_subpath(rel_path: str) -> Path | None:
    """Resolve *rel_path* under the outputs dir, or ``None`` if it escapes.

    Guards the league endpoint against path traversal: only paths that stay
    inside ``_DEFAULT_OUTPUTS_DIR`` are returned.
    """
    if not rel_path or ".." in Path(rel_path).parts:
        return None
    outputs_dir = _DEFAULT_OUTPUTS_DIR.resolve()
    target = (outputs_dir / rel_path).resolve()
    try:
        target.relative_to(outputs_dir)
    except ValueError:
        return None
    return target


def _league_snapshots(league_dir: Path) -> list[dict[str, Any]]:
    """Return per-snapshot metadata for a self-play league directory.

    Each entry has the snapshot name, its timestep tag, byte size, age in
    seconds, and whether its obs-normaliser sidecar is present. Non-numeric
    stray files (e.g. a hand-copied ``selfplay_best.zip``) are skipped, matching
    ``LeagueSampler``. Sorted oldest→newest by timestep.
    """
    if not league_dir.is_dir():
        return []
    import time

    now = time.time()
    snaps: list[dict[str, Any]] = []
    for p in league_dir.glob("selfplay_*.zip"):
        tag = p.stem.rsplit("_", 1)[-1]
        if not tag.isdigit():
            continue
        st = p.stat()
        snaps.append(
            {
                "name": p.name,
                "timesteps": int(tag),
                "size_bytes": st.st_size,
                "age_seconds": max(0.0, now - st.st_mtime),
                "has_vecnorm": p.with_name(p.stem + _VECNORM_SUFFIX).exists(),
            }
        )
    snaps.sort(key=lambda s: s["timesteps"])
    return snaps


@app.route("/api/outputs", methods=["GET"])
def list_outputs():
    """List completed experiment outputs."""
    outputs_dir = _DEFAULT_OUTPUTS_DIR
    if not outputs_dir.exists():
        return jsonify([])
    results = []

    def _append_seed(experiment: str, run_id: str | None, seed_dir: Path) -> None:
        checkpoints = (
            list((seed_dir / "checkpoints").glob("*.zip"))
            if (seed_dir / "checkpoints").exists()
            else []
        )
        league_dir = seed_dir / "checkpoints" / "league"
        results.append(
            {
                "experiment": experiment,
                "run_id": run_id,
                "seed": seed_dir.name,
                "path": str(seed_dir.relative_to(outputs_dir)),
                "checkpoints": [p.name for p in sorted(checkpoints)],
                "has_final_model": (
                    seed_dir / "checkpoints" / "final_model.zip"
                ).exists(),
                # Cheap count so the dashboard can flag self-play runs; full
                # snapshot detail is served lazily by /api/league.
                "league_size": len(_league_snapshots(league_dir)),
            }
        )

    def _is_seed_dir(p: Path) -> bool:
        return p.is_dir() and not p.is_symlink() and p.name.startswith("seed_")

    for exp_dir in sorted(outputs_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.is_symlink():
            continue
        # Plain single-run / multi-seed layout: <experiment>/seed_<seed>/.
        for seed_dir in sorted(exp_dir.iterdir()):
            if _is_seed_dir(seed_dir):
                _append_seed(exp_dir.name, None, seed_dir)
        # Sweep / morphology variants: <experiment>/runs/<run_id>/seed_<seed>/.
        runs_dir = exp_dir / "runs"
        if runs_dir.is_dir() and not runs_dir.is_symlink():
            for run_dir in sorted(runs_dir.iterdir()):
                if not run_dir.is_dir() or run_dir.is_symlink():
                    continue
                for seed_dir in sorted(run_dir.iterdir()):
                    if _is_seed_dir(seed_dir):
                        _append_seed(exp_dir.name, run_dir.name, seed_dir)
    return jsonify(results)


@app.route("/api/league", methods=["GET"])
def get_league():
    """Return self-play league detail for one seed run.

    Query param ``path`` is a seed directory relative to the outputs root (the
    ``path`` field returned by ``/api/outputs``). Responds with the league size,
    the newest snapshot's timestep, and per-snapshot metadata. Unknown or
    league-less runs return an empty league rather than an error.
    """
    rel = request.args.get("path", "")
    target = _safe_output_subpath(rel)
    if target is None:
        abort(404)
    league_dir = target / "checkpoints" / "league"
    snapshots = _league_snapshots(league_dir)
    return jsonify(
        {
            "path": rel,
            "exists": league_dir.is_dir(),
            "league_size": len(snapshots),
            "newest_timesteps": snapshots[-1]["timesteps"] if snapshots else None,
            "snapshots": snapshots,
        }
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def run_gui(host: str = "127.0.0.1", port: int = 5001, debug: bool = False) -> None:
    print(f"\n  RL Experiment GUI running at http://{host}:{port}\n")
    # Reloader on: edits to .py files auto-restart the server. Note that this
    # also kills any in-progress training (it lives in this process). Stop a
    # run via the dashboard before editing if you want to preserve it.
    app.run(host=host, port=port, debug=debug, use_reloader=True)
