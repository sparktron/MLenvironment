"""Single COM push screening, paired by environment and policy RNG seeds.

Each unpushed episode supplies a trigger step for each raw contact phase.
Replays use identical policy noise up to that step. Trace rows are saved as
gzip JSONL; episode records and bootstrap comparisons are retained separately.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import subprocess
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pybullet as p
from stable_baselines3.common.utils import set_random_seed

from rl_framework.core.config.config import DEFAULT_CONFIG_DIR, load_config, to_container
from rl_framework.core.training.statistics import wilson_interval
from rl_framework.physics.locomotion.analysis import walker_diagnostics as wd
from rl_framework.physics.locomotion.perturbations import PushController

DIRECTIONS = {"forward": (1, 0, 0), "backward": (-1, 0, 0),
              "left": (0, 1, 0), "right": (0, -1, 0)}
PHASES = {(True, False): "right_support", (False, True): "left_support",
          (True, True): "double_support", (False, False): "flight"}
TRACE_COLUMNS = ["step", "x", "y", "z", "roll", "pitch", "yaw",
                 "vx", "vy", "vz", "wx", "wy", "wz", "right", "left"]


class DiagnosticPush(PushController):
    """Use the production per-physics-tick COM application, one scheduled pulse."""

    def configure(self, trigger, vector, duration):
        self.trigger = trigger
        self.command_vector = vector
        self.duration = duration
        self.applied_ticks = 0

    def maybe_start(self, spec, step_count, rng, sim_timestep):
        if self.trigger is None or step_count != self.trigger:
            return False
        self.vector = tuple(self.command_vector)
        self.ticks_remaining = round(self.duration / sim_timestep)
        return True

    def apply_tick(self, robot_id):
        super().apply_tick(robot_id)
        self.applied_ticks += 1


def snapshot(base):
    pos, quat = p.getBasePositionAndOrientation(base.robot_id, physicsClientId=base._connection)
    vel, ang = p.getBaseVelocity(base.robot_id, physicsClientId=base._connection)
    contacts = base.sensor.foot_contacts()
    return [base.step_count, *pos, *p.getEulerFromQuaternion(quat), *vel, *ang,
            *[int(c) for c in contacts]]


def rollout(model, env, base, controller, rng_seed, trigger, vector, args):
    set_random_seed(rng_seed, using_cuda=False)
    env.seed(rng_seed)
    controller.configure(trigger, vector, args.duration)
    obs = env.reset()
    trace, rewards, infos = [], [], []
    limit = args.search_end + args.horizon if trigger is None else trigger + args.horizon
    done = False
    while base.step_count < limit and not done:
        trace.append(snapshot(base))
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, dones, info = env.step(action)
        rewards.append(float(reward[0]))
        infos.append(info[0])
        done = bool(dones[0])
    last = infos[-1]
    fell = last.get("termination_reason") in {
        "torso_contact", "min_height", "max_height", "physics_divergence"}
    # DummyVecEnv resets on termination: never sample its reset state as terminal.
    if not done:
        trace.append(snapshot(base))
    return {"trace": trace, "rewards": rewards, "fell": fell,
            "termination_reason": last.get("termination_reason"),
            "end_step": len(rewards), "applied_ticks": controller.applied_ticks}


def metrics(run, trigger, horizon, dt):
    end = min(trigger + horizon, run["end_step"])
    a = np.asarray(run["trace"], dtype=float)
    post = a[(a[:, 0] >= trigger) & (a[:, 0] <= end)]
    fell = run["fell"] and run["end_step"] <= trigger + horizon
    return {
        "fell": bool(fell), "termination_reason": run["termination_reason"],
        "time_to_fall_s": (run["end_step"] - trigger) * dt if fell else None,
        "observed_s": (end - trigger) * dt,
        "return": float(sum(run["rewards"][trigger:end])),
        "max_abs_roll": float(np.max(np.abs(post[:, 4]))),
        "max_abs_pitch": float(np.max(np.abs(post[:, 5]))),
        "max_abs_vy": float(np.max(np.abs(post[:, 8]))),
        "max_abs_vx": float(np.max(np.abs(post[:, 7]))),
        "max_abs_wx": float(np.max(np.abs(post[:, 10]))),
        "max_abs_wy": float(np.max(np.abs(post[:, 11]))),
        "min_z": float(np.min(post[:, 3])),
        "flight_fraction": float(np.mean((post[:, 13] == 0) & (post[:, 14] == 0))),
        "support_transitions": int(np.sum(np.any(np.diff(post[:, 13:15], axis=0), axis=1))),
        "trajectory_samples": [row.tolist() for row in post
                               if int(row[0] - trigger) in {0, 6, 15, 30, 60, 120, 180}],
    }


def fresh_rollout(model, cfg, path, rng_seed, trigger, vector, args):
    # Reusing a reset world retains solver/contact history and fails pairing.
    env = wd._evaluation_env(cfg, path, 1)
    vec = env
    while hasattr(vec, "venv"):
        vec = vec.venv
    base = vec.envs[0].unwrapped
    controller = DiagnosticPush(base._connection)
    base._pushes = controller
    try:
        return rollout(model, env, base, controller, rng_seed, trigger, vector, args)
    finally:
        env.close()


def paired_interval(values):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    means = rng.choice(a, (2000, len(a)), replace=True).mean(axis=1)
    return [float(a.mean()), *np.quantile(means, [0.025, 0.975]).tolist()]


def summarise(records):
    groups = {}
    for r in records:
        key = f'{r["direction"]}/{r["force"]:g}/{r["phase"]}'
        groups.setdefault(key, []).append(r)
    result = {}
    for key, rows in groups.items():
        falls = sum(r["pushed"]["fell"] for r in rows)
        base_falls = sum(r["baseline"]["fell"] for r in rows)
        paired = [int(r["pushed"]["fell"]) - int(r["baseline"]["fell"]) for r in rows]
        times = [r["pushed"]["time_to_fall_s"] for r in rows if r["pushed"]["fell"]]
        result[key] = {
            "n": len(rows), "falls": falls, "baseline_falls": base_falls,
            "fall_ci95": list(wilson_interval(falls, len(rows))),
            "fall_diff_ci95": paired_interval(paired),
            "return_diff_ci95": paired_interval([
                r["pushed"]["return"] - r["baseline"]["return"] for r in rows]),
            "median_time_to_fall_s": float(np.median(times)) if times else None,
        }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--forces", default="60,120,180")
    ap.add_argument("--directions", default=",".join(DIRECTIONS))
    ap.add_argument("--phases", default=",".join(PHASES.values()))
    ap.add_argument("--duration", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--search-end", type=int, default=240)
    ap.add_argument("--horizon", type=int, default=180)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.episodes < 1 or not 0 < args.warmup < args.search_end or args.horizon < 1:
        ap.error("require episodes >= 1, 0 < warmup < search-end, horizon >= 1")
    forces = [float(x) for x in args.forces.split(",")]
    directions, phases = args.directions.split(","), args.phases.split(",")
    if not set(directions) <= DIRECTIONS.keys() or not set(phases) <= set(PHASES.values()):
        ap.error("unknown direction or phase")
    if min(forces) <= 0 or args.duration <= 0:
        ap.error("forces and duration must be positive")
    cfg = to_container(load_config("walker_reward_v9_flight_cost", str(DEFAULT_CONFIG_DIR)))
    cfg = deepcopy(cfg)
    cfg["seed"] = cfg["environment"]["seed"] = args.seed
    path = Path("outputs") / cfg["experiment_name"] / f"seed_{args.seed}/checkpoints/final_model.zip"
    model = wd._model_class(cfg).load(str(path), device="cpu")
    env = wd._evaluation_env(cfg, path, 1)
    vec = env
    while hasattr(vec, "venv"):
        vec = vec.venv
    base = vec.envs[0].unwrapped
    if base.terrain.preset != "flat":
        raise ValueError("single pushes require a flat baseline")
    controller = DiagnosticPush(base._connection)
    base._pushes = controller
    dt = base.sim.control_timestep
    if args.search_end + args.horizon > base.termination.max_steps:
        raise ValueError("search end + horizon exceeds episode limit")
    sim_timestep = base.sim.timestep
    env.close()
    args.out.mkdir(parents=True, exist_ok=True)
    sidecar = wd.find_vecnormalize_path_for_model(path)
    metadata = {
        "args": {**vars(args), "out": str(args.out)}, "config": cfg,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "normalization_sha256": hashlib.sha256(Path(sidecar).read_bytes()).hexdigest() if sidecar else None,
        "python": platform.python_version(),
        "platform": platform.platform(), "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "versions": {k: version(k) for k in ["numpy", "torch", "stable-baselines3", "pybullet"]},
        "trace_columns": TRACE_COLUMNS, "control_dt": dt,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    records, eligibility = [], []
    try:
        with gzip.open(args.out / "traces.jsonl.gz", "wt") as trace_file:
            for ep in range(args.episodes):
                rng_seed = 100000 + args.seed * 1000 + ep
                baseline = fresh_rollout(model, cfg, path, rng_seed, None, (0, 0, 0), args)
                triggers = {}
                for row in baseline["trace"]:
                    step = row[0]
                    if args.warmup <= step < args.search_end:
                        phase = PHASES[tuple(bool(x) for x in row[-2:])]
                        if phase in phases:
                            triggers.setdefault(phase, step)
                eligibility.append({"episode": ep, "rng_seed": rng_seed,
                                    "triggers": triggers, "baseline_fell": baseline["fell"]})
                trace_file.write(json.dumps({"episode": ep, "baseline": baseline}) + "\n")
                for phase, trigger in triggers.items():
                    for direction in directions:
                        for force in forces:
                            vector = np.asarray(DIRECTIONS[direction]) * force
                            run = fresh_rollout(model, cfg, path, rng_seed, trigger, vector, args)
                            # Common random numbers only count as pairing if replay is verified.
                            before = np.asarray(run["trace"][:trigger + 1])
                            reference = np.asarray(baseline["trace"][:trigger + 1])
                            error = float(np.max(np.abs(before - reference))) if before.shape == reference.shape else float("inf")
                            if error > 1e-9:
                                raise RuntimeError(f"unpaired replay at episode {ep}, error={error}")
                            expected_ticks = round(args.duration / sim_timestep)
                            if run["applied_ticks"] != expected_ticks:
                                raise RuntimeError("pulse truncated by termination or applied incorrectly")
                            record = {"episode": ep, "rng_seed": rng_seed, "phase": phase,
                                      "trigger": trigger, "direction": direction, "force": force,
                                      "prepush_max_error": error, "applied_ticks": run["applied_ticks"],
                                      "pushed": metrics(run, trigger, args.horizon, dt),
                                      "baseline": metrics(baseline, trigger, args.horizon, dt)}
                            records.append(record)
                            trace_file.write(json.dumps({"condition": record, "run": run}) + "\n")
                (args.out / "records.json").write_text(json.dumps(records, indent=2))
                (args.out / "summary.json").write_text(json.dumps(summarise(records), indent=2))
                (args.out / "eligibility.json").write_text(json.dumps(eligibility, indent=2))
                print(f"seed={args.seed} episode={ep + 1}/{args.episodes} phases={list(triggers)} trials={len(records)}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
