"""Run the walker transfer suite (and optional clearance telemetry) per seed.

Usage:
    python scripts/walker_transfer_report.py --config-name walker_reward_v9_flight_cost \
        --seeds 21,22,23,24,25 --out outputs/_reports/v9_transfer.json \
        [--clearance-point sole] [--swing-telemetry]

Reads ``outputs/<experiment_name>/seed_<seed>/checkpoints/final_model.zip`` and
writes one JSON with per-seed, per-terrain results (fall rates carry Wilson
95% intervals). Evaluation only; no training.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl_framework.core.config.config import load_config, to_container  # noqa: E402
from rl_framework.physics.locomotion.walker_diagnostics import (  # noqa: E402
    collect_walker_swing_event_telemetry,
    evaluate_walker_transfer_suite,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--config-dir", default="src/rl_framework/configs/experiments")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clearance-point", choices=["com", "sole"], default=None)
    ap.add_argument("--swing-telemetry", action="store_true")
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()

    base = to_container(load_config(args.config_name, args.config_dir))
    out: dict = {
        "config": args.config_name,
        "clearance_point": args.clearance_point or "config-default",
        "episodes": args.episodes,
        "seeds": {},
    }
    for seed in [int(s) for s in args.seeds.split(",")]:
        cfg = deepcopy(base)
        cfg["seed"] = seed
        cfg["environment"]["seed"] = seed
        if args.clearance_point:
            cfg["environment"].setdefault("gait", {})[
                "clearance_point"
            ] = args.clearance_point
        model = (
            Path(cfg.get("output", {}).get("base_dir", "outputs"))
            / cfg["experiment_name"]
            / f"seed_{seed}"
            / "checkpoints"
            / "final_model.zip"
        )
        if not model.is_file():
            raise FileNotFoundError(model)
        entry: dict = {
            "model": str(model),
            "suite": evaluate_walker_transfer_suite(
                cfg, model, episodes=args.episodes
            ),
        }
        if args.swing_telemetry:
            entry["swing_events"] = collect_walker_swing_event_telemetry(
                cfg, model, episodes=20
            )
        out["seeds"][str(seed)] = entry
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[report] seed {seed} done", flush=True)


if __name__ == "__main__":
    main()
