# UI Review Roadmap

Last reviewed: 2026-06-30
Implementation status: completed for the P1/P2/P3 items listed below on 2026-06-30.

Scope: Flask experiment GUI in `src/rl_framework/gui`, including the new-experiment wizard, dashboard, outputs browser, and responsive layout. This file now records the review, implementation decisions, and regression checklist for the completed UI pass.

## Review Evidence

- Read `src/rl_framework/gui/app.py`, `src/rl_framework/gui/static/app.js`, `src/rl_framework/gui/static/style.css`, `src/rl_framework/gui/templates/index.html`, `src/rl_framework/gui/training_manager.py`, and `tests/test_gui_api.py`.
- Launched the GUI with `source .venv/bin/activate && python -m rl_framework.cli.main gui --port 5011`.
- Browser-tested the wizard, dashboard, outputs tab, template loading, environment switching, and responsive breakpoints at 1280x720 and 390x844.
- Ran `node --check src/rl_framework/gui/static/app.js`.

## Confirmed Bugs

### P1: Walker environment card shows obsolete observation/action sizes

Status: **Done** — `index.html` now shows 35-dimensional observations and 10-dimensional actions; `tests/test_gui_api.py` checks the rendered template text.

`src/rl_framework/gui/templates/index.html` says the walker has a 13-dimensional observation and 3-dimensional action. The current `WalkerBulletEnv` exposes 35 observations and 10 actions.

Impact: users can choose or tune a walker experiment with a misleading mental model of the policy input/output contract.

Fix plan:
- Update the card copy to 35-dimensional observation and 10-dimensional action details.
- Prefer deriving these facts from a single schema/source if the GUI grows more environment metadata.
- Add a lightweight test that rendered template text matches current env dimensions or add a backend schema endpoint for card metadata.

Validation:
- Browser check the New Experiment tab.
- Run the narrow GUI API/template test that covers the card metadata once added.

### P1: Template-loaded training state can bleed into another environment

Status: **Done** — switching environment cards clears incompatible template state and resets visible top-level defaults; training/evaluation prefill now only reuses `currentConfig` when it matches the selected environment.

Loading a walker template sets `currentConfig`, then returning to step 1 and choosing the arena carries template training values into the new arena flow. In browser verification, direct arena selection produced `num_envs: 1`, but switching from `robot_walk_basic` to arena produced `num_envs: 8` in the review preview.

Relevant code:
- `src/rl_framework/gui/static/app.js` stores template config in `currentConfig` in `loadTemplates()`.
- `renderTrainingParams()` uses `(prefill || currentConfig).training`, even after the selected environment changes.

Impact: the wizard can generate invalid or confusing arena configs from a previous walker template. The server validator catches `organism_arena_parallel` with `num_envs != 1`, but the user only discovers it after reaching launch or reviewing JSON.

Fix plan:
- Track whether `currentConfig` came from a template and clear it when the user manually selects a different environment.
- When `selectedEnv` changes, rebuild training/evaluation defaults from `schema[selectedEnv]`.
- Consider keeping only fields that are valid for both envs and within the selected schema's min/max bounds.

Validation:
- Browser regression: load `robot_walk_basic`, go back, select Organism Arena, advance to review, assert `training.num_envs === 1`.
- Add a focused JS/browser test if frontend test tooling is introduced; otherwise document this manual browser check.

### P2: Dashboard controls remain active when no run is selectable

Status: **Done** — dashboard state is centralized in `setDashboardRunState()`, no-run state clears metrics/frame history, and Stop/live-tuning controls are enabled only for running runs.

With no active runs, the dashboard shows a "No runs" selector while the Stop button and live tuning Apply button remain enabled. Clicking them only yields error toasts.

Relevant code:
- `src/rl_framework/gui/templates/index.html` renders Stop and Apply as always-enabled controls.
- `refreshRuns()` returns early for an empty run list without clearing or disabling dashboard actions.
- Stop and tuning handlers guard in JavaScript, but only after the user clicks.

Impact: users have to discover state through failed actions, and the dashboard reads as interactive even when it has no runnable target.

Fix plan:
- Add a central `setDashboardRunState(run)` helper that updates selector, badge, metrics, Stop button, tuning inputs, and empty state together.
- Disable Stop unless the selected run status is `running`.
- Disable live tuning unless a selected run is `running`.
- Reset metric cards, frame scrubber, and chart placeholder when no run is selected.

Validation:
- Browser check no-run dashboard: Stop and Apply disabled, status badge neutral, metrics reset.
- Browser check running/completed synthetic states if a small frontend fixture route or test harness is added.

## Usability And Flow Improvements

### P2: Outputs titles omit variant `run_id`

Status: **Done** — output item titles render `experiment / run_id / seed` for nested variants and keep the full path in muted metadata.

The outputs API returns `run_id` for sweep and morphology variants, but the UI title only renders `experiment / seed`; the variant is only visible inside the path line.

Impact: repeated variants under the same experiment are harder to scan, compare, and select.

Fix plan:
- Render `experiment / run_id / seed` when `run_id` is present.
- Keep the full path in muted metadata for copy/debug use.

Validation:
- Browser check outputs containing `runs/<run_id>/seed_<seed>` and plain `experiment/seed_<seed>` layouts.

### P3: Mobile wizard progress labels wrap awkwardly

Status: **Done** — step labels are wrapped in dedicated spans; mobile CSS switches to compact step numbers and shows the active step label below the progress bar.

At 390px width the four-step progress bar remains a single row. "Review & Launch" wraps into multiple lines and consumes disproportionate height.

Impact: the wizard remains usable, but the first viewport feels cramped and the step indicator competes with the actual task content.

Fix plan:
- At small widths, render compact step labels or numbers-only progress with the active step label shown below.
- Keep Back and Next/Launch controls anchored at the bottom of each wizard panel and in the same left/right positions.

Validation:
- Browser screenshot/check at approximately 390x844 and a desktop viewport.

### P3: Dashboard visualization wastes horizontal space on desktop

Status: **Done** — desktop dashboard uses a two-column `dashboard-main` grid pairing visualization with reward history while preserving single-column mobile layout.

The visualization panel is full-width while the canvas is capped at 640px, leaving a large empty area to the right on desktop.

Impact: dashboard scan density is lower than necessary, and users must scroll farther to reach reward/tuning sections.

Fix plan:
- Use a dashboard grid that pairs visualization with reward chart or run status on wider screens.
- Preserve single-column stacking on mobile.

Validation:
- Browser check 1280x720 and mobile.

## Implementation Order

1. ~~Fix confirmed P1 correctness issues: walker card metadata and template environment-switch state.~~
2. ~~Fix dashboard disabled/empty states for Stop and tuning controls.~~
3. ~~Improve outputs titles for variant runs.~~
4. ~~Polish responsive wizard progress and dashboard desktop layout.~~
5. ~~Add the smallest practical frontend regression harness, or document the browser checklist in tests if the repo continues without frontend tooling.~~

## Regression Checklist

- New Experiment: direct walker and direct arena flows produce valid config previews.
- Template flow: loading a walker template, then switching to arena, resets arena-only defaults.
- Dashboard no-run state: destructive or mutating controls are disabled.
- Dashboard selected-run state: Stop is enabled only for running runs; live tuning is enabled only for running runs.
- Outputs: plain seed runs and nested variant runs have distinct, readable titles.
- Responsive: 390px width has no text overlap, no inaccessible primary action, and consistent Back/Next placement.
