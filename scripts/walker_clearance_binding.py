"""Telemetry only: why is bilateral_clearance_target not reached?

Per v9 seed, collect individual swing events (side, clearance, duration) from
stochastic flat rollouts and test three explanations for the shortfall:
  (a) the reward is min(left, right), so the weaker leg binds;
  (b) lifting higher costs swing time (airtime) that the flight penalty charges;
  (c) the target is simply out of the policy's reach (share of swings >= target).
Usage: python scripts/walker_clearance_binding.py --seeds 21,22,23,24,25 --out F.json
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from rl_framework.core.config.config import DEFAULT_CONFIG_DIR, load_config, to_container
from rl_framework.physics.locomotion.analysis import walker_diagnostics as wd


def collect(cfg, model_path, episodes):
    path = wd.model_zip_path(model_path)
    model = wd._model_class(cfg).load(str(path), device="cpu")
    env = wd._evaluation_env(cfg, path, 1)
    ev = []
    try:
        seed = int(cfg.get("seed", 0))
        for ep in range(episodes):
            env.seed(seed + ep)
            obs = env.reset()
            done, t = False, 0
            while not done:
                a, _ = model.predict(obs, deterministic=False)
                obs, _, d, infos = env.step(a)
                e = infos[0].get("swing_touchdown_event")
                if isinstance(e, dict) and e.get("side") is not None:
                    ev.append((ep, t, int(e["side"]), float(e["clearance"]), float(e["duration"])))
                t += 1
                done = bool(d[0])
    finally:
        env.close()
    return ev


def analyse(ev, target=0.05):
    a = np.array(ev, dtype=float)
    side, clr, dur = a[:, 2], a[:, 3], a[:, 4]
    out = {"events": len(a), "frac_ge_target": float((clr >= target).mean())}
    for s, n in ((0, "right"), (1, "left")):
        m = side == s
        out[f"{n}_median_mm"] = float(np.median(clr[m]) * 1000)
    # Pair consecutive alternating swings within an episode; reward uses the min.
    mins, means = [], []
    for ep in np.unique(a[:, 0]):
        r = a[a[:, 0] == ep]
        for i in range(1, len(r)):
            if r[i, 2] != r[i - 1, 2]:
                mins.append(min(r[i, 3], r[i - 1, 3]))
                means.append((r[i, 3] + r[i - 1, 3]) / 2)
    out["pairs"] = len(mins)
    out["pair_mean_mm"] = float(np.mean(means) * 1000)
    out["pair_min_mm"] = float(np.mean(mins) * 1000)
    out["min_penalty_mm"] = out["pair_mean_mm"] - out["pair_min_mm"]
    out["corr_clearance_duration"] = float(np.corrcoef(clr, dur)[0, 1])
    slope = np.polyfit(clr * 1000, dur / (1 / 60), 1)[0]
    out["extra_swing_steps_per_10mm"] = float(slope * 10)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="walker_reward_v9_flight_cost")
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = to_container(load_config(args.config_name, args.config_dir))
    res = {}
    for s in [int(x) for x in args.seeds.split(",")]:
        cfg = deepcopy(base)
        cfg["seed"] = s
        cfg["environment"]["seed"] = s
        m = Path("outputs") / cfg["experiment_name"] / f"seed_{s}" / "checkpoints" / "final_model.zip"
        res[str(s)] = analyse(collect(cfg, m, args.episodes))
        print(s, json.dumps(res[str(s)]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
