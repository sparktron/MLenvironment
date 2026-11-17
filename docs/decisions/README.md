# Decision records

Lightweight ADRs for `MLenvironment`. One decision per file, `NNN-slug.md`,
template at `ai-hub/templates/ADR.md`.

**Scope boundary.** `docs/walker_learning_history.md` remains the canonical
append-only ledger for *experiment* results and hypotheses, including refuted
ones. This directory is for *architectural and design* decisions about the
framework. When a result drives a design change, the result goes in the history
and the change gets an ADR here that cites it.

Only decisions already documented with clear reasoning were converted — one.
ADR usage starts properly from here **going forward**; history was not mined.

| ADR | Title | Status |
|---|---|---|
| [001](001-terrain-mix-not-curriculum.md) | Vary terrain by per-episode sampling, not by curriculum | Accepted |
| [003](003-env-structure-and-declarative-config.md) | Split the walker env; declare config and live parameters as data | Accepted |
| [004](004-editable-install-src-layout.md) | Install the src layout editable; drop the root `rl_framework/` shim | Accepted |
