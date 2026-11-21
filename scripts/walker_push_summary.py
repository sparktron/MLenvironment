"""Retain push-screening evidence and summarize with episode-level pairing."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_push_diagnostic import paired_interval, summarise


def group_metrics(rows):
    episodes = defaultdict(list)
    for r in rows:
        episodes[r["episode"]].append(r)
    falls, excess, returns = [], [], []
    for rr in episodes.values():
        falls.append(np.mean([r["pushed"]["fell"] for r in rr]))
        excess.append(np.mean([int(r["pushed"]["fell"]) - int(r["baseline"]["fell"]) for r in rr]))
        returns.append(np.mean([r["pushed"]["return"] - r["baseline"]["return"] for r in rr]))
    return {
        "episodes": len(episodes), "trials": len(rows),
        "fall_rate": float(np.mean(falls)), "excess_falls_ci95": paired_interval(excess),
        "return_diff_ci95": paired_interval(returns),
    }


def analyze(rows):
    groups = defaultdict(list)
    for r in rows:
        family = "lateral" if r["direction"] in {"left", "right"} else "sagittal"
        for key in (f'{r["force"]:g}/{family}', f'{r["force"]:g}/{r["direction"]}',
                    f'{r["force"]:g}/{family}/{r["phase"]}'):
            groups[key].append(r)
    result = {key: group_metrics(rr) for key, rr in groups.items()}
    contrasts = {}
    for force in sorted({r["force"] for r in rows}):
        by_episode = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["force"] == force:
                family = "lateral" if r["direction"] in {"left", "right"} else "sagittal"
                by_episode[r["episode"]][family].append(
                    int(r["pushed"]["fell"]) - int(r["baseline"]["fell"]))
        delta = [np.mean(rr["lateral"]) - np.mean(rr["sagittal"])
                 for rr in by_episode.values() if rr["lateral"] and rr["sagittal"]]
        contrasts[f"{force:g}"] = paired_interval(delta)
    return {"groups": result, "lateral_minus_sagittal_excess_ci95": contrasts,
            "cells": summarise(rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs/_reports/push_diagnostic"))
    ap.add_argument("--dest", type=Path, default=Path("docs/experiments/push_diagnostic_2026-11-20"))
    ap.add_argument("--seeds", default="21,22,23,24,25")
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    result = {}
    for seed in args.seeds.split(","):
        folder = args.root / f"seed_{seed}"
        payload = {key: json.loads((folder / f"{key}.json").read_text())
                   for key in ("metadata", "records", "eligibility")}
        if len(payload["eligibility"]) != payload["metadata"]["args"]["episodes"]:
            raise ValueError(f"seed {seed} incomplete")
        result[seed] = analyze(payload["records"])
        (args.dest / f"seed_{seed}.json.gz").write_bytes(gzip.compress(
            json.dumps(payload, separators=(",", ":")).encode(), mtime=0))
    (args.dest / "summary.json").write_text(json.dumps(result, indent=2))
    for seed, r in result.items():
        print(seed, json.dumps({"contrasts": r["lateral_minus_sagittal_excess_ci95"],
                               "families": {k: v for k, v in r["groups"].items()
                                            if k.count("/") == 1 and k.split("/")[1] in {"lateral", "sagittal"}}}))


if __name__ == "__main__":
    main()
