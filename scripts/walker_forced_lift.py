"""Telemetry only: force extra swing lift on a trained walker and price it.

The 2026-11-18 lift-cost entry showed that higher swings co-vary with *more*
reward, but that is observational. This script intervenes instead. For the first
``--pulse-steps`` control steps after each raw lift-off of a foot, it adds a
fixed offset to that leg's normalized hip and knee actions:
``dose * (hip -0.6, knee +1.0)``. Hip is negative-flexion and knee positive-flexion
in this robot; at that ratio the foot rises with little fore-aft shift (checked
kinematically). The policy stays in the loop and may react. Each dose replays
the same episode seeds, so comparisons with dose 0 are paired.

Reported per seed and dose: mean swing clearance (mm), fall rate with a Wilson
95% interval, mean episode return, and the paired return difference vs dose 0
with a 95% bootstrap interval over episodes.
Usage: python scripts/walker_forced_lift.py --seed 21 --out F.json
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from rl_framework.core.config.config import (
    DEFAULT_CONFIG_DIR,
    load_config,
    to_container,
)
from rl_framework.physics.locomotion.analysis import walker_diagnostics as wd

FALL_REASONS = {"torso_contact", "min_height", "max_height", "physics_divergence"}
# Action indices (rHip, rKnee) and (lHip, lKnee), and the per-unit-dose direction.
LEG_JOINTS = ((0, 1), (3, 4))
DIRECTION = (-0.6, 1.0)


def run_dose(model, env, seed, episodes, dose, pulse_steps):
    eps = []
    for ep in range(episodes):
        env.seed(seed + ep)
        obs = env.reset()
        done, ret, steps, clr, reason = False, 0.0, 0, [], None
        contact = [True, True]
        since_liftoff = [pulse_steps, pulse_steps]
        while not done:
            a, _ = model.predict(obs, deterministic=False)
            a = np.array(a, dtype=np.float32)
            for leg, (hip, knee) in enumerate(LEG_JOINTS):
                if not contact[leg] and since_liftoff[leg] < pulse_steps:
                    a[0, hip] += dose * DIRECTION[0]
                    a[0, knee] += dose * DIRECTION[1]
            a = np.clip(a, -1.0, 1.0)
            obs, r, d, infos = env.step(a)
            info = infos[0]
            now = (bool(info["right_foot_contact"]), bool(info["left_foot_contact"]))
            for leg in (0, 1):
                if contact[leg] and not now[leg]:
                    since_liftoff[leg] = 0
                elif not now[leg]:
                    since_liftoff[leg] += 1
            contact = list(now)
            e = info.get("swing_touchdown_event")
            if isinstance(e, dict) and e.get("side") is not None:
                clr.append(float(e["clearance"]))
            reason = info.get("termination_reason")
            ret += float(r[0])
            steps += 1
            done = bool(d[0])
        eps.append(
            {"return": ret, "steps": steps, "fell": reason in FALL_REASONS, "clr": clr}
        )
    return eps


def summarise(eps, base, boot=1000):
    clr = (
        np.concatenate([e["clr"] for e in eps])
        if any(e["clr"] for e in eps)
        else np.array([])
    )
    falls = sum(e["fell"] for e in eps)
    ret = np.array([e["return"] for e in eps])
    diff = ret - np.array([e["return"] for e in base])
    rng = np.random.default_rng(0)
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(boot)]
    lo, hi = wd.wilson_interval(float(falls), len(eps))
    return {
        "swings": int(len(clr)),
        "clearance_mm": float(clr.mean() * 1000) if len(clr) else float("nan"),
        "falls": int(falls),
        "episodes": len(eps),
        "fall_ci95": [float(lo), float(hi)],
        "return_mean": float(ret.mean()),
        "steps_mean": float(np.mean([e["steps"] for e in eps])),
        "return_diff_vs_0": float(diff.mean()),
        "return_diff_ci95": [
            float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)),
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="walker_reward_v9_flight_cost")
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--doses", default="0,0.2,0.3,0.4,0.5,0.6,0.8")
    ap.add_argument("--pulse-steps", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = deepcopy(to_container(load_config(args.config_name, args.config_dir)))
    cfg["seed"] = args.seed
    cfg["environment"]["seed"] = args.seed
    m = (
        Path("outputs")
        / cfg["experiment_name"]
        / f"seed_{args.seed}"
        / "checkpoints"
        / "final_model.zip"
    )
    path = wd.model_zip_path(m)
    model = wd._model_class(cfg).load(str(path), device="cpu")
    env = wd._evaluation_env(cfg, path, 1)
    doses = [float(x) for x in args.doses.split(",")]
    assert doses[0] == 0.0, "first dose must be 0 (the paired baseline)"
    res = {}
    try:
        base = None
        for dose in doses:
            eps = run_dose(model, env, args.seed, args.episodes, dose, args.pulse_steps)
            base = base or eps
            res[str(dose)] = summarise(eps, base)
            print(args.seed, dose, json.dumps(res[str(dose)]), flush=True)
    finally:
        env.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
