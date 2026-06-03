#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


MATRIX_VERSION = "2026-05-28-small-first-v2"


@dataclass(frozen=True)
class Regime:
    name: str
    device: str
    max_workers: int


REGIMES = (
    Regime(name="CPU-4workers", device="cpu", max_workers=4),
    Regime(name="CPU-8workers", device="cpu", max_workers=8),
    Regime(name="GPU-1worker", device="cuda", max_workers=1),
    Regime(name="GPU-2workers", device="cuda", max_workers=2),
)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dbg(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[debug {_ts()}] {msg}", flush=True)


def _matrix_order() -> list[str]:
    return [regime.name for regime in REGIMES]


def _matrix_metadata() -> dict:
    return {
        "matrix_version": MATRIX_VERSION,
        "script_path": str(Path(__file__).resolve()),
        "order": _matrix_order(),
    }


def _write_progress_event(progress_log: Path | None, event: dict) -> None:
    if progress_log is None:
        return
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": _ts(), **event}
    with progress_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _append_progress_log(progress_log: Path | None, event: dict) -> None:
    """Backward-compatible alias for older tests/imports."""
    _write_progress_event(progress_log, event)


def _run_regime(
    config_name: str,
    seeds: str,
    config_dir: str,
    regime: Regime,
    inactivity_timeout_s: float,
    heartbeat_s: float,
    total_timesteps: int | None,
    debug: bool,
    progress_log: Path | None,
    ordinal: int,
    total_regimes: int,
) -> dict:
    _dbg(
        debug,
        f"enter _run_regime(name={regime.name}, device={regime.device}, workers={regime.max_workers})",
    )
    with tempfile.TemporaryDirectory(prefix="bench_matrix_") as tmpdir:
        result_path = Path(tmpdir) / f"{regime.name}.json"
        _dbg(debug, f"created tempdir={tmpdir} result_path={result_path}")
        cmd = [
            sys.executable,
            "-m",
            "rl_framework.cli.main",
            "multi-seed",
            "--config-name",
            config_name,
            "--config-dir",
            config_dir,
            "--seeds",
            seeds,
            "--max-workers",
            str(regime.max_workers),
            "--device",
            regime.device,
            "--total-timesteps",
            str(total_timesteps),
            "--json-out",
            str(result_path),
        ]
        _dbg(debug, f"built command={' '.join(cmd)}")
        _write_progress_event(
            progress_log,
            {
                "event": "regime_started",
                "ordinal": ordinal,
                "total_regimes": total_regimes,
                "name": regime.name,
                "device": regime.device,
                "max_workers": regime.max_workers,
                "command": cmd,
            },
        )
        start = time.perf_counter()
        print(f"[exec] {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd)
        print(f"[pid] {regime.name} pid={proc.pid}", flush=True)
        _dbg(debug, f"spawned process pid={proc.pid}")
        last_heartbeat = start
        while True:
            now = time.perf_counter()
            elapsed = now - start
            if now - last_heartbeat >= heartbeat_s:
                print(
                    f"[heartbeat] {regime.name} running for {elapsed:.0f}s...",
                    flush=True,
                )
                last_heartbeat = now
            if elapsed > inactivity_timeout_s:
                proc.kill()
                _dbg(
                    debug,
                    f"killed process pid={proc.pid} due to timeout elapsed={elapsed:.1f}s",
                )
                _append_progress_log(
                    progress_log,
                    {
                        "event": "regime_timed_out",
                        "ordinal": ordinal,
                        "total_regimes": total_regimes,
                        "name": regime.name,
                        "device": regime.device,
                        "max_workers": regime.max_workers,
                        "elapsed_s": elapsed,
                        "timeout_s": inactivity_timeout_s,
                    },
                )
                raise RuntimeError(
                    f"Benchmark regime '{regime.name}' timed out after {inactivity_timeout_s:.0f}s."
                )
            return_code = proc.poll()
            if return_code is not None:
                _dbg(debug, f"process pid={proc.pid} exited return_code={return_code}")
                break
            time.sleep(1.0)

        elapsed_s = time.perf_counter() - start
        _dbg(debug, f"regime elapsed_s={elapsed_s:.3f}")
        if return_code != 0:
            _write_progress_event(
                progress_log,
                {
                    "event": "regime_failed",
                    "ordinal": ordinal,
                    "total_regimes": total_regimes,
                    "name": regime.name,
                    "device": regime.device,
                    "max_workers": regime.max_workers,
                    "elapsed_s": elapsed_s,
                    "exit_code": return_code,
                },
            )
            raise RuntimeError(
                f"Benchmark regime '{regime.name}' failed (exit={return_code}).\n"
                f"Command: {' '.join(cmd)}\n"
                "Inspect terminal output above for details."
            )
        if not result_path.exists():
            _dbg(debug, f"result json missing path={result_path}")
            _write_progress_event(
                progress_log,
                {
                    "event": "regime_missing_result",
                    "ordinal": ordinal,
                    "total_regimes": total_regimes,
                    "name": regime.name,
                    "device": regime.device,
                    "max_workers": regime.max_workers,
                    "elapsed_s": elapsed_s,
                    "result_path": str(result_path),
                },
            )
            raise RuntimeError(
                f"Benchmark regime '{regime.name}' succeeded but did not write JSON result file: {result_path}"
            )
        _dbg(debug, f"reading result json from {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        _dbg(
            debug,
            "parsed result keys=" + ",".join(sorted(result.keys())),
        )
        row = {
            "name": regime.name,
            "device": regime.device,
            "max_workers": regime.max_workers,
            "elapsed_s": elapsed_s,
            "mean_return_mean": float(result["mean_return_mean"]),
            "mean_return_std": float(result["mean_return_std"]),
        }
        _append_progress_log(
            progress_log,
            {
                "event": "regime_completed",
                "ordinal": ordinal,
                "total_regimes": total_regimes,
                **row,
            },
        )
        row = {
            "name": regime.name,
            "device": regime.device,
            "max_workers": regime.max_workers,
            "elapsed_s": elapsed_s,
            "mean_return_mean": float(result["mean_return_mean"]),
            "mean_return_std": float(result["mean_return_std"]),
        }
        _write_progress_event(
            progress_log,
            {
                "event": "regime_completed",
                "ordinal": ordinal,
                "total_regimes": total_regimes,
                **row,
            },
        )
        print(f"[done] {regime.name} elapsed={elapsed_s:.1f}s", flush=True)
    _dbg(debug, f"exit _run_regime(name={regime.name})")
    return row


def _pick_winner(rows: list[dict], reward_tolerance_ratio: float) -> tuple[dict, str]:
    best_reward = max(row["mean_return_mean"] for row in rows)
    reward_floor = best_reward - abs(best_reward) * reward_tolerance_ratio
    eligible = [row for row in rows if row["mean_return_mean"] >= reward_floor]
    winner = min(eligible, key=lambda row: row["elapsed_s"])
    rule = (
        "Winner rule: choose the fastest regime among those with "
        f"mean_return_mean >= {reward_floor:.4f} "
        f"({reward_tolerance_ratio * 100:.1f}% of best reward {best_reward:.4f})."
    )
    return winner, rule


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fixed 4-regime benchmark matrix and pick the fastest acceptable regime."
    )
    parser.add_argument(
        "--config-name", required=True, help="Experiment config name (without .yaml)."
    )
    parser.add_argument("--config-dir", default="src/rl_framework/configs/experiments")
    parser.add_argument(
        "--seeds",
        default="0,1,2,3",
        help="Comma-separated seeds to use for all regimes.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=20000,
        help="Training timesteps override passed to each regime (default: 20000).",
    )
    parser.add_argument(
        "--reward-tolerance-ratio",
        type=float,
        default=0.03,
        help="Acceptable reward drop ratio versus best reward when selecting by speed (default: 0.03 = 3%%).",
    )
    parser.add_argument(
        "--inactivity-timeout-s",
        type=float,
        default=300.0,
        help="Fail a regime if it runs longer than this many seconds (default: 300).",
    )
    parser.add_argument(
        "--heartbeat-s",
        type=float,
        default=30.0,
        help="Print a heartbeat while waiting for output (default: 30).",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable debug logs.",
    )
    parser.add_argument(
        "--progress-log",
        default="outputs/benchmark_device_matrix_progress.jsonl",
        help=(
            "Append JSONL progress events to this path so completed regimes are visible after a crash "
            "(default: outputs/benchmark_device_matrix_progress.jsonl; pass an empty string to disable)."
        ),
    )
    args = parser.parse_args()

    if not (0.0 <= args.reward_tolerance_ratio < 1.0):
        raise SystemExit("--reward-tolerance-ratio must be in [0.0, 1.0).")
    if args.inactivity_timeout_s <= 0:
        raise SystemExit("--inactivity-timeout-s must be > 0.")
    if args.heartbeat_s <= 0:
        raise SystemExit("--heartbeat-s must be > 0.")
    if args.total_timesteps is not None and args.total_timesteps <= 0:
        raise SystemExit("--total-timesteps must be > 0.")

    rows: list[dict] = []
    progress_log = Path(args.progress_log) if args.progress_log else None
    metadata = _matrix_metadata()
    order = metadata["order"]
    _dbg(
        args.debug,
        f"benchmark start config={args.config_name} seeds={args.seeds} total_timesteps={args.total_timesteps} "
        f"matrix_version={metadata['matrix_version']} script_path={metadata['script_path']}",
    )
    print(f"[matrix] version: {metadata['matrix_version']}", flush=True)
    print(f"[matrix] script: {metadata['script_path']}", flush=True)
    print(f"[matrix] order: {' -> '.join(order)}", flush=True)
    if progress_log is not None:
        print(f"[progress-log] {progress_log}", flush=True)
    _write_progress_event(
        progress_log,
        {
            "event": "matrix_started",
            "matrix_version": metadata["matrix_version"],
            "script_path": metadata["script_path"],
            "config_name": args.config_name,
            "config_dir": args.config_dir,
            "seeds": args.seeds,
            "total_timesteps": args.total_timesteps,
            "order": order,
        },
    )
    for ordinal, regime in enumerate(REGIMES, start=1):
        print(
            f"[run {ordinal}/{len(REGIMES)}] {regime.name}  "
            f"device={regime.device}  max_workers={regime.max_workers}",
            flush=True,
        )
        rows.append(
            _run_regime(
                args.config_name,
                args.seeds,
                args.config_dir,
                regime,
                inactivity_timeout_s=args.inactivity_timeout_s,
                heartbeat_s=args.heartbeat_s,
                total_timesteps=args.total_timesteps,
                debug=args.debug,
            )
        )
        try:
            rows.append(
                _run_regime(
                    args.config_name,
                    args.seeds,
                    args.config_dir,
                    regime,
                    inactivity_timeout_s=args.inactivity_timeout_s,
                    heartbeat_s=args.heartbeat_s,
                    total_timesteps=args.total_timesteps,
                    debug=args.debug,
                    progress_log=progress_log,
                    ordinal=ordinal,
                    total_regimes=len(REGIMES),
                )
            )
        except Exception as exc:
            _write_progress_event(
                progress_log,
                {
                    "event": "matrix_failed",
                    "failed_regime": regime.name,
                    "completed_regimes": [row["name"] for row in rows],
                    "error": str(exc),
                },
            )
            if rows:
                print("[partial-results] completed regimes before failure:", flush=True)
                print(json.dumps(rows, indent=2), flush=True)
            raise
        _dbg(args.debug, f"collected row for {regime.name}: {rows[-1]}")

    winner, rule = _pick_winner(
        rows, reward_tolerance_ratio=args.reward_tolerance_ratio
    )
    _dbg(args.debug, f"winner={winner['name']} rule={rule}")
    payload = {"results": rows, "winner": winner, "decision_rule": rule}
    _write_progress_event(progress_log, {"event": "matrix_completed", **payload})
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
