"""Flask web application for the RL Experiment GUI."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import yaml
from flask import Flask, jsonify, render_template, request

from rl_framework.gui.training_manager import TrainingManager

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "experiments"

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)
manager = TrainingManager()


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
            configs.append({
                "name": p.stem,
                "experiment_name": data.get("experiment_name", p.stem),
                "env_type": data.get("environment", {}).get("type", "unknown"),
            })
        except Exception:
            configs.append({"name": p.stem, "experiment_name": p.stem, "env_type": "unknown"})
    return jsonify(configs)


@app.route("/api/configs/<name>", methods=["GET"])
def get_config(name: str):
    """Return the full parsed YAML config."""
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        return jsonify({"error": f"Config not found: {name}"}), 404
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return jsonify(data)


@app.route("/api/configs/<name>", methods=["PUT"])
def save_config(name: str):
    """Save a modified config back to YAML."""
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Empty payload"}), 400
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
    path = CONFIGS_DIR / f"{name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return jsonify({"saved": name, "path": str(path)})


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
                    "gravity": {"value": -9.81, "type": "float", "desc": "Gravity (m/s^2)", "min": -20, "max": 0},
                    "mass": {"value": 3.0, "type": "float", "desc": "Body mass (kg)", "min": 0.1, "max": 50},
                    "friction": {"value": 0.9, "type": "float", "desc": "Ground friction", "min": 0.0, "max": 2.0},
                    "max_force": {"value": 35.0, "type": "float", "desc": "Max actuator force (N)", "min": 1, "max": 200},
                    "body_half_extents": {"value": [0.2, 0.1, 0.08], "type": "list_float", "desc": "Body size [x, y, z]"},
                },
                "reward": {
                    "alive_bonus": {"value": 1.0, "type": "float", "desc": "Bonus for staying alive", "min": 0, "max": 10},
                    "forward_velocity_weight": {"value": 2.0, "type": "float", "desc": "Weight on forward velocity reward", "min": 0, "max": 20},
                    "target_velocity": {"value": 1.0, "type": "float", "desc": "Target velocity (m/s)", "min": 0, "max": 5},
                    "orientation_penalty_weight": {"value": 1.0, "type": "float", "desc": "Penalty for tilting", "min": 0, "max": 10},
                    "torque_penalty_weight": {"value": 0.01, "type": "float", "desc": "Penalty for torque usage", "min": 0, "max": 1},
                },
                "termination": {
                    "min_height": {"value": 0.12, "type": "float", "desc": "Min body height before termination (m)", "min": 0, "max": 1},
                    "max_tilt_radians": {"value": 0.8, "type": "float", "desc": "Max body tilt before termination (rad)", "min": 0.1, "max": 3.14},
                    "max_steps": {"value": 800, "type": "int", "desc": "Max steps per episode", "min": 50, "max": 10000},
                },
                "reset_randomization": {
                    "position_xy_noise": {"value": 0.02, "type": "float", "desc": "XY position noise at reset", "min": 0, "max": 1},
                    "yaw_noise": {"value": 0.08, "type": "float", "desc": "Yaw noise at reset (rad)", "min": 0, "max": 1},
                },
                "domain_randomization": {
                    "mass_scale_range": {"value": [0.95, 1.05], "type": "range", "desc": "Mass scale randomization [lo, hi]", "min": 0.5, "max": 2.0},
                    "friction_range": {"value": [0.9, 1.1], "type": "range", "desc": "Friction randomization [lo, hi]", "min": 0.1, "max": 3.0},
                    "sensor_noise_std": {"value": 0.0, "type": "float", "desc": "Gaussian sensor noise std", "min": 0, "max": 1},
                    "action_latency_steps": {"value": 0, "type": "int", "desc": "Action latency (steps)", "min": 0, "max": 10},
                },
            },
            "training": _training_schema(),
            "evaluation": _eval_schema(),
        },
        "organism_arena_parallel": {
            "environment": {
                "type": {"value": "organism_arena_parallel", "type": "fixed"},
                "seed": {"value": 11, "type": "int", "desc": "Environment random seed"},
                "sim": {
                    "arena_half_extent": {"value": 1.2, "type": "float", "desc": "Arena half-size", "min": 0.5, "max": 5},
                },
                "morphology": {
                    "base_size": {"value": 1.0, "type": "float", "desc": "Base organism size", "min": 0.1, "max": 5},
                    "episode_growth_scale": {"value": 0.0, "type": "float", "desc": "Size growth per step", "min": 0, "max": 0.1},
                    "health": {"value": 1.2, "type": "float", "desc": "Initial health (scaled by size)", "min": 0.1, "max": 10},
                    "energy": {"value": 1.0, "type": "float", "desc": "Initial energy", "min": 0.1, "max": 10},
                },
                "battle_rules": {
                    "damage": {"value": 0.06, "type": "float", "desc": "Damage per hit", "min": 0.01, "max": 1},
                    "attack_range": {"value": 0.2, "type": "float", "desc": "Attack range", "min": 0.05, "max": 2},
                    "cooldown_steps": {"value": 3, "type": "int", "desc": "Cooldown between attacks", "min": 0, "max": 20},
                    "max_steps": {"value": 400, "type": "int", "desc": "Max steps per episode", "min": 50, "max": 10000},
                    "win_health_threshold": {"value": 0.0, "type": "float", "desc": "Health below which opponent loses", "min": 0, "max": 1},
                },
            },
            "training": _training_schema(),
            "evaluation": _eval_schema(),
        },
    }
    return jsonify(schema)


def _training_schema() -> dict[str, Any]:
    return {
        "policy": {"value": "MlpPolicy", "type": "choice", "choices": ["MlpPolicy", "CnnPolicy"], "desc": "Policy network type"},
        "total_timesteps": {"value": 20000, "type": "int", "desc": "Total training timesteps", "min": 1000, "max": 10000000},
        "learning_rate": {"value": 0.0003, "type": "float", "desc": "Learning rate", "min": 0.000001, "max": 0.1},
        "n_steps": {"value": 1024, "type": "int", "desc": "Rollout buffer size", "min": 16, "max": 8192},
        "batch_size": {"value": 256, "type": "int", "desc": "Minibatch size", "min": 8, "max": 4096},
        "checkpoint_every": {"value": 5000, "type": "int", "desc": "Save checkpoint every N steps", "min": 100, "max": 1000000},
        "normalize_observations": {"value": True, "type": "bool", "desc": "Normalize observations with VecNormalize"},
        "num_envs": {"value": 1, "type": "int", "desc": "Parallel environments", "min": 1, "max": 32},
    }


def _eval_schema() -> dict[str, Any]:
    return {
        "episodes": {"value": 5, "type": "int", "desc": "Evaluation episodes", "min": 1, "max": 100},
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
        return jsonify(result), 409
    return jsonify(result)


@app.route("/api/train/stop/<run_id>", methods=["POST"])
def stop_training(run_id: str):
    result = manager.stop_run(run_id)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


@app.route("/api/train/status/<run_id>", methods=["GET"])
def training_status(run_id: str):
    result = manager.get_status(run_id)
    if "error" in result:
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


# ------------------------------------------------------------------
# Outputs API
# ------------------------------------------------------------------

@app.route("/api/outputs", methods=["GET"])
def list_outputs():
    """List completed experiment outputs."""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return jsonify([])
    results = []
    for exp_dir in sorted(outputs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            checkpoints = list((seed_dir / "checkpoints").glob("*.zip")) if (seed_dir / "checkpoints").exists() else []
            results.append({
                "experiment": exp_dir.name,
                "seed": seed_dir.name,
                "path": str(seed_dir),
                "checkpoints": [p.name for p in sorted(checkpoints)],
                "has_final_model": (seed_dir / "checkpoints" / "final_model.zip").exists(),
            })
    return jsonify(results)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def run_gui(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    print(f"\n  RL Experiment GUI running at http://{host}:{port}\n")
    app.run(host=host, port=port, debug=debug, use_reloader=False)
