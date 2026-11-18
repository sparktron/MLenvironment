"""Telemetry only: which reward components pay for swing lift?

The v13 Step 0 telemetry showed that flight is not what makes lift expensive
(0.3-7% of the extra swing steps are flight). This script prices every reward
component instead. For each swing it takes two windows:
  swing: raw lift-off to confirmed touchdown (steps t-n .. t-1);
  swing+post: the swing plus the next ``--post-steps`` steps (landing, stance,
  and the bilateral-clearance payoff, which only updates at touchdown).
For each component it computes the window's *excess*: the window sum minus
(window length x that episode's mean per-step value). That is what the swing
earned or cost compared with spending the same time at the episode's average.
The excess is regressed on clearance within episodes (both are demeaned per
episode, removing between-episode gait differences) and reported per 10 mm, with
a 95% bootstrap interval over episodes. It also reports how the chance of a fall
within ``--fall-horizon`` steps of touchdown changes with clearance.
Correlational: it shows what co-varies with lift, not what lift causes.
Usage: python scripts/walker_lift_cost.py --seeds 21,22,23,24,25 --out F.json
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


def collect(cfg, model_path, episodes, post_steps, fall_horizon):
    path = wd.model_zip_path(model_path)
    model = wd._model_class(cfg).load(str(path), device="cpu")
    env = wd._evaluation_env(cfg, path, 1)
    rows, names = [], None
    try:
        seed = int(cfg.get("seed", 0))
        for ep in range(episodes):
            env.seed(seed + ep)
            obs = env.reset()
            done, t, comps, events, reason = False, 0, [], [], None
            while not done:
                a, _ = model.predict(obs, deterministic=False)
                obs, _, d, infos = env.step(a)
                info = infos[0]
                rc = info["reward_components"]
                names = names or sorted(rc)
                comps.append([rc[k] for k in names])
                e = info.get("swing_touchdown_event")
                if isinstance(e, dict) and e.get("side") is not None:
                    n = int(round(float(e["duration"]) * 60))
                    events.append((t, float(e["clearance"]), n))
                reason = info.get("termination_reason")
                t += 1
                done = bool(d[0])
            c = np.asarray(comps)
            rate = c.mean(axis=0)
            fell = reason in FALL_REASONS
            for td, clr, n in events:
                lo = max(td - n, 0)
                hi = min(td + post_steps, len(c))
                swing = c[lo:td].sum(axis=0) - (td - lo) * rate
                post = c[lo:hi].sum(axis=0) - (hi - lo) * rate
                fall_soon = float(fell and len(c) - 1 - td <= fall_horizon)
                rows.append((ep, clr, fall_soon, swing, post))
    finally:
        env.close()
    return rows, names


def _within_slope(ep, x, y):
    """Slope of y on x after demeaning both within each episode (y may be 2-D)."""
    xd = x.copy()
    yd = y.copy()
    for e in np.unique(ep):
        m = ep == e
        xd[m] -= xd[m].mean()
        yd[m] -= yd[m].mean(axis=0)
    return (xd @ yd) / (xd @ xd)


def analyse(rows, names, boot):
    ep = np.array([r[0] for r in rows])
    clr_mm = np.array([r[1] for r in rows]) * 1000
    fall = np.array([r[2] for r in rows])
    windows = {
        "swing": np.array([r[3] for r in rows]),
        "swing_post": np.array([r[4] for r in rows]),
    }
    eps = np.unique(ep)
    idx = {e: np.flatnonzero(ep == e) for e in eps}
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(boot):
        # Relabel resampled episodes so a duplicate is demeaned as its own episode.
        pick = rng.choice(eps, len(eps))
        sel = np.concatenate([idx[e] for e in pick])
        lab = np.concatenate([np.full(len(idx[e]), i) for i, e in enumerate(pick)])
        samples.append((sel, lab))
    out = {"events": len(rows), "falls_after_touchdown": int(fall.sum())}
    for w, y in windows.items():
        y = np.column_stack([y, y.sum(axis=1)])
        keys = list(names) + ["TOTAL"]
        point = _within_slope(ep, clr_mm, y) * 10
        bs = np.array(
            [_within_slope(lab, clr_mm[sel], y[sel]) * 10 for sel, lab in samples]
        )
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        out[w] = {
            k: [float(point[i]), float(lo[i]), float(hi[i])]
            for i, k in enumerate(keys)
            if np.any(y[:, i] != 0)
        }
    fb = np.array(
        [_within_slope(lab, clr_mm[sel], fall[sel]) * 10 for sel, lab in samples]
    )
    out["fall_prob_per_10mm"] = [
        float(_within_slope(ep, clr_mm, fall) * 10),
        *map(float, np.percentile(fb, [2.5, 97.5])),
    ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="walker_reward_v9_flight_cost")
    ap.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--post-steps", type=int, default=30)
    ap.add_argument("--fall-horizon", type=int, default=60)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = to_container(load_config(args.config_name, args.config_dir))
    res = {}
    for s in [int(x) for x in args.seeds.split(",")]:
        cfg = deepcopy(base)
        cfg["seed"] = s
        cfg["environment"]["seed"] = s
        m = (
            Path("outputs")
            / cfg["experiment_name"]
            / f"seed_{s}"
            / "checkpoints"
            / "final_model.zip"
        )
        rows, names = collect(cfg, m, args.episodes, args.post_steps, args.fall_horizon)
        res[str(s)] = analyse(rows, names, args.boot)
        print(s, json.dumps(res[str(s)]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
