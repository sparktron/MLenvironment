"""Reproducible Priority-3 learning-quality studies.

The runner deliberately keeps candidate settings out of shipped defaults until
they clear a multi-seed/budget gate. State is written after every training run
so long studies can resume without repeating completed work.
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from rl_framework.core.studies import registered_studies
from rl_framework.training.sb3_runner import train
from rl_framework.utils.checkpoint import model_zip_path
from rl_framework.utils.config import (
    load_config,
    to_container,
    validate_experiment_config,
)
from rl_framework.utils.config_merge import set_nested



# Gait thresholds are a *band*, not a set of floors. The 2026-07-26 correction
# (see docs/walker_learning_history.md) found that the original "at least as
# good as the observed baseline" calibration froze the chatter it was meant to
# diagnose: the baseline shuffled at ~20 alternating touchdowns per 100 steps
# with 10.9 mm strides and 0.7 mm clearance, so a floor of 15 demanded ~11 Hz
# stepping and *rejected* real walking.
#
# Cadence is derived from gait mechanics rather than from that baseline. One
# gait cycle produces two alternating touchdowns, so at 60 Hz control:
#     alternating touchdowns / 100 steps = 100 * 2 * f / 60 = 3.33 * f
# for a cycle frequency f in Hz. Plausible bipedal walking spans f = 0.45-2.5 Hz
# (human/Atlas-class walking sits at 0.8-1.4 Hz), giving 1.5-8.5. Anything
# above the ceiling is contact chatter; anything below it is not stepping.
#
# Stride and clearance floors make a touchdown mean something. An open-loop
# scripted gait at this study's own `action_scale: 0.25` / `position_gain: 0.1`
# reached 33.7 mm of foot clearance, so 20 mm is ~60% of demonstrated actuator
# capability rather than an invented number. Progress per alternating touchdown
# is set to the minimum consistent with the 3 m success gate at the fastest
# permitted cadence (3 m / ~67 touchdowns), and a same-foot stride is one full
# cycle, i.e. about twice the per-touchdown pelvis advance.
#
# GATE V3 (2026-08-02): THE OBJECTIVE CHANGED, SO THE SUPPORT CRITERIA CHANGED.
#
# This is a gate change made with a candidate in view, which this project
# otherwise forbids. It is legitimate here for one specific reason, and the
# reason must survive scrutiny: the *target* was redefined by the project owner,
# not the candidate's numbers. The stated goal is a framework that demonstrably
# improves a locomotion policy over iterations, and running is an acceptable --
# explicitly preferred -- outcome. Gate v2's `min_double_support_fraction` and
# `max_flight_fraction: 0.10` encoded "walking specifically", a target nobody
# is pursuing any more. They were correct for the old objective and are simply
# the wrong question for the new one.
#
# Guard against the obvious abuse of that argument: v9 is NOT the reason. Its
# values were not used to place any v3 threshold, and the anti-degenerate work
# the old criteria performed is preserved rather than dropped (see below). If a
# future reader suspects this was a goalpost move, the check is whether the
# thresholds were derived from degenerate cases or from a candidate -- they were
# derived from degenerate cases.
#
# What changed and why:
#
# 1. `min_double_support_fraction` is REMOVED, not relaxed. Running has zero
#    double support by definition, so a floor on it is incoherent for a target
#    that admits running -- any nonzero value would forbid the preferred gait.
#
# 2. `max_flight_fraction` returns to 0.60, its pre-v2 value, which had an
#    independent derivation the v2 tightening discarded: open-loop scripted
#    gaits at this robot's own settings measured 17-56% flight, so 60% is the
#    envelope of commanded gaits rather than an invented number. Human sprinting
#    peaks near 40%. The ceiling now excludes only degenerate leaping.
#
# 3. `max_contact_duty_imbalance: 0.15` is ADDED, and it is what actually
#    replaces the removed criteria. The 2026-07-28 entry recorded the real
#    hazard: an asymmetric BOUND (right foot in contact 27.3% against the left
#    at 15.0%, 57.6% flight, 0.1% double support) that a naive gate would score
#    perfectly. The old flight ceiling excluded it only as a side effect, using
#    flight as a proxy for asymmetry. A symmetry criterion targets the defect
#    directly, so the flight ceiling no longer has to do two jobs at once.
#
#    Threshold derivation, independent of any candidate: 0.15 means one leg
#    carries 57.5% of stance time and the other 42.5%. That is the line past
#    which a gait is meaningfully one-legged. Validity check (NOT calibration):
#    the 2026-07-28 bound measures 0.289 and is excluded; healthy alternating
#    gaits measure ~0.02. Both sit far from the threshold, which is what a
#    discriminating criterion should look like.
#
# Cadence, stride, clearance, progress, slip, survival, fall rate, displacement,
# peak height and same-foot-sequence limits are ALL UNCHANGED. Every guard
# against contact chatter, self-launching, standing, and sliding is retained --
# the loosening is confined to the two criteria that encoded walking specifically.
#
# Slip is the planar velocity of the slowest material contact point, not the
# foot-link origin. Link-origin velocity includes ordinary heel/toe pivoting and
# is therefore not a physical measure of sliding. The existing conservative
# 0.18 m/s ceiling is retained; changing the instrument does not justify
# loosening the threshold.
#
# 2026-08-02 recalibration attempt, recorded here because it did NOT move the
# threshold. The intended method -- measure a reference gait at the 1.0 m/s
# target, as the clearance floor was calibrated from an open-loop scripted gait
# -- is not available: clearance is an actuator-capability measure and survives
# a rollout that falls in a second, whereas slip only means something during
# sustained balanced locomotion at speed, and the only policies that do that are
# the candidates themselves. Calibrating from candidates is the error that made
# gate v1 certify chatter.
#
# What the measurement did establish (20 stochastic episodes each, both at
# 1.04 m/s mean forward speed):
#
#                       whole-phase   mid-stance     transient share
#   v7 contact point         0.1206       0.0764                 37%
#   v8 contact point         0.1667       0.1503                 10%
#
# The gate metric averages the whole contact phase, so it includes the landing
# transient -- the foot still moving as it arrives, carrying roughly the body's
# forward speed. That component scales with locomotion speed; ideal mid-stance
# slip is zero at any speed. The transient's share is also gait-dependent (37%
# vs 10%), so the gated metric can separate two candidates on landing dynamics
# rather than on sliding. `gait_mid_stance_contact_point_slip_speed` was added as
# TELEMETRY ONLY at that point, following the 2026-07-28 precedent: a corrected
# instrument is not swapped into a live gate with a candidate in view.
#
# GATE V4 (2026-08-02): THE SLIP GATE NOW USES THE MID-STANCE METRIC.
#
# Decided by the project owner, on the grounds that the whole-phase metric was
# not providing the feedback it was introduced to provide. That is the correct
# reading: the criterion exists to catch a foot SLIDING under load, and 10-41%
# of what it reported was a landing transient -- the foot still moving as it
# arrives, which is not sliding, scales with locomotion speed, and varies by
# gait. A criterion whose value depends that heavily on a quantity it was not
# meant to measure cannot do its job.
#
# `max_slip_speed` stays 0.18. Correcting an instrument is not a reason to move
# a threshold, exactly as in the 2026-07-28 link-origin -> contact-point change.
# Against the corrected metric 0.18 is a STRICTER test of sliding than it was
# against the contaminated one, since the contamination only ever inflated the
# reported value.
#
# DISCLOSURE, because the 2026-07-28 precedent demands it: this change moves a
# candidate across the line. v9 seed 23 measured 0.2178 whole-phase (fail) and
# 0.1283 mid-stance (pass), so it goes from 12/13 to 13/13. The precedent's
# actual requirement is not "never switch" but "when correcting an instrument
# moves a candidate across a line, check what else is now on the passing side".
# That check was run over every existing checkpoint; see the 2026-08-02 entry in
# docs/walker_learning_history.md for what did and did not move.
# The slip-recovery screen found that a bounded 0.5 slip cost was the strongest
# tested contact intervention, but stochastic actions still produced excessive
# slip and contact chatter. The next isolated question is whether observing the
# previously applied command gives PPO enough short-term state to recover and
# sustain a real swing. Both new lineages keep the measured slip intervention
# fixed; only the opt-in observation changes.





class StopOnWallClock(BaseCallback):
    """Stop SB3 learning once a monotonic wall-clock budget is exhausted."""

    def __init__(self, seconds: float):
        super().__init__(verbose=0)
        self.seconds = float(seconds)
        self._started_at = 0.0

    def _on_training_start(self) -> None:
        self._started_at = time.perf_counter()

    def _on_step(self) -> bool:
        return time.perf_counter() - self._started_at < self.seconds


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def _load_cfg(config_name: str, config_dir: str | Path) -> dict[str, Any]:
    return to_container(load_config(config_name, config_dir))


def _apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        set_nested(cfg, key, value, strict=False)


def _prepare_cfg(
    cfg: dict[str, Any],
    *,
    experiment_name: str,
    seed: int,
    step_budget: int,
    output_dir: Path,
) -> dict[str, Any]:
    prepared = deepcopy(cfg)
    prepared["experiment_name"] = experiment_name
    prepared["seed"] = int(seed)
    prepared.setdefault("environment", {})["seed"] = int(seed)
    prepared.setdefault("training", {})["total_timesteps"] = int(step_budget)
    if str(prepared["training"].get("algorithm", "PPO")).upper() == "PPO":
        # PPO always finishes a complete rollout, so an oversized shipped
        # n_steps*num_envs would silently turn a 10k study smoke into a 49k
        # run. Shrink only when needed so the requested comparison budget is
        # respected within one small rollout of overshoot.
        configured_envs = int(prepared["training"].get("num_envs", 1))
        configured_steps = int(prepared["training"].get("n_steps", 1024))
        target_rollout = min(
            configured_envs * configured_steps,
            max(32, step_budget // 10),
        )
        num_envs = min(configured_envs, max(1, target_rollout // 2))
        n_steps = min(configured_steps, max(2, target_rollout // num_envs))
        prepared["training"]["num_envs"] = num_envs
        prepared["training"]["n_steps"] = n_steps
        rollout_size = n_steps * num_envs
        preferred_batch = min(
            int(prepared["training"].get("batch_size", 256)), rollout_size
        )
        prepared["training"]["batch_size"] = next(
            size for size in range(preferred_batch, 1, -1) if rollout_size % size == 0
        )
    prepared["training"]["checkpoint_every"] = min(
        int(prepared["training"].get("checkpoint_every", step_budget)), step_budget
    )
    prepared.setdefault("evaluation", {})["episodes"] = max(
        1, int(prepared.get("evaluation", {}).get("episodes", 5))
    )
    prepared.setdefault("output", {})["base_dir"] = str(output_dir / "models")
    prepared["output"].pop("run_id", None)
    validate_experiment_config(prepared)
    return prepared


def _study_identity(
    studies: list[str],
    seeds: list[int],
    step_budget: int | None,
    wall_clock_seconds: float,
    eval_episodes: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "studies": studies,
        "seeds": seeds,
        "step_budget": step_budget,
        "wall_clock_seconds": wall_clock_seconds,
        "eval_episodes": eval_episodes,
    }


def _new_state(identity: dict[str, Any]) -> dict[str, Any]:
    return {"identity": identity, "runs": {}, "results": {}, "completed": False}


def _load_state(path: Path, identity: dict[str, Any], resume: bool) -> dict[str, Any]:
    if not resume:
        return _new_state(identity)
    if not path.exists():
        raise FileNotFoundError(f"No quality-study state found at {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("identity") != identity:
        raise ValueError(
            "Quality-study resume state does not match this invocation; "
            "use a new output directory or the original budgets/seeds."
        )
    return state


def _train_run(
    key: str,
    cfg: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    *,
    wall_clock_seconds: float | None = None,
    resume_from: str | Path | None = None,
    extra_callbacks: list[BaseCallback] | None = None,
) -> dict[str, Any]:
    previous = state["runs"].get(key)
    if previous and previous.get("status") == "completed":
        path = Path(previous["model_path"])
        if path.is_file():
            return previous

    callbacks: list[BaseCallback] = list(extra_callbacks or [])
    if wall_clock_seconds is not None:
        callbacks.append(StopOnWallClock(wall_clock_seconds))
    record: dict[str, Any] = {
        "status": "running",
        "config": cfg,
        "resume_from": str(resume_from) if resume_from is not None else None,
    }
    state["runs"][key] = record
    _atomic_json_write(state_path, state)
    started_at = time.perf_counter()
    try:
        model = train(
            cfg,
            extra_callbacks=callbacks or None,
            resume_from=resume_from,
        )
        path = model_zip_path(model)
        elapsed = time.perf_counter() - started_at
        record.update(
            {
                "status": "completed",
                "model_path": str(path),
                "training_seconds": elapsed,
            }
        )
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "error": str(exc),
                "training_seconds": time.perf_counter() - started_at,
            }
        )
    _atomic_json_write(state_path, state)
    return record






































































def _format_report(result: dict[str, Any]) -> str:
    lines = ["# Learning Quality Study", ""]
    identity = result["identity"]
    lines.extend(
        [
            f"- Studies: {', '.join(identity['studies'])}",
            f"- Seeds: {identity['seeds']}",
            f"- Step budget override: {identity['step_budget']}",
            f"- Wall-clock budget: {identity['wall_clock_seconds']} seconds",
            "",
            "Candidate settings are promoted only when their result reports its readiness gate as true.",
            "",
        ]
    )
    for study, payload in result["results"].items():
        lines.extend(
            [
                f"## {study.title()}",
                "",
                "```json",
                json.dumps(payload, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def run_quality_study(
    study: str,
    *,
    seeds: list[int],
    config_dir: str | Path = "src/rl_framework/configs/experiments",
    output_dir: str | Path = "outputs/quality_studies",
    step_budget: int | None = None,
    wall_clock_seconds: float = 900.0,
    eval_episodes: int = 20,
    source_dir: str | Path = "outputs/walker_stochastic_balance_candidate",
    approved_variant: str | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one registered study or the default set of studies."""
    available = registered_studies()
    by_name = {item.name: item for item in available}
    studies = [item.name for item in available if item.include_in_all] if study == "all" else [study]
    invalid = set(studies) - set(by_name)
    if invalid:
        raise ValueError(f"Unknown quality study: {sorted(invalid)}")
    if not seeds:
        raise ValueError("Quality studies need at least one seed")
    if step_budget is not None and step_budget <= 0:
        raise ValueError("step_budget must be positive")
    if wall_clock_seconds <= 0 or eval_episodes <= 0:
        raise ValueError("wall_clock_seconds and eval_episodes must be positive")

    output = Path(output_dir)
    identity = _study_identity(
        studies, seeds, step_budget, wall_clock_seconds, eval_episodes
    )
    if any(by_name[name].uses_source for name in studies):
        identity["source_dir"] = str(Path(source_dir))
    plan = {
        "identity": identity,
        "planned_runs": {
            item.result_key: (
                item.variants * (len(seeds) if item.per_seed else 1)
                if item.name in studies else 0
            )
            for item in available
        },
        "dry_run": dry_run,
    }
    if dry_run:
        return plan

    state_path = output / "state.json"
    state = _load_state(state_path, identity, resume)
    config_path = Path(config_dir)
    for name in studies:
        spec = by_name[name]
        kwargs = {
            "seeds": seeds,
            "step_budget": step_budget or spec.default_steps,
            "eval_episodes": eval_episodes,
            "config_dir": config_path,
            "output_dir": output,
            "state": state,
            "state_path": state_path,
        }
        if spec.uses_source:
            kwargs["source_dir"] = Path(source_dir)
        if spec.uses_approval:
            kwargs["approved_variant"] = approved_variant
        if spec.uses_wall_clock:
            kwargs["wall_clock_seconds"] = wall_clock_seconds
        state["results"][spec.result_key] = spec.run(**kwargs)
    state["completed"] = all(
        run.get("status") == "completed" for run in state["runs"].values()
    )
    _atomic_json_write(state_path, state)
    report = {
        "identity": identity,
        "results": state["results"],
        "completed": state["completed"],
    }
    _atomic_json_write(output / "report.json", report)
    (output / "report.md").write_text(_format_report(report), encoding="utf-8")
    return report
