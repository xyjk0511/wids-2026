---
phase: 04-stronger-anchor
plan: 01
subsystem: tooling
tags: [rsf, gbsa, sksurv, anchor, reproduction, comparison]

requires: []
provides:
  - "exp30_reproduce_anchor.py: configurable RSF+GBSA pipeline, default=0.96624 settings"
  - "exp30_compare_anchors.py: per-horizon Spearman + blend eligibility verdict"
affects: [04-02]

tech-stack:
  added: []
  patterns:
    - "JSON config drives model hyperparams/weights — no code changes to test variants"
    - "Blend eligibility gate: rho < 0.99 on any horizon = ELIGIBLE"

key-files:
  created:
    - scripts/exp30_reproduce_anchor.py
    - scripts/exp30_compare_anchors.py
  modified: []

key-decisions:
  - "Task 2 executed before Task 1 (tooling independent of notebook artifacts)"
  - "exp30_compare_anchors.py pre-existed with correct logic — kept as-is"
  - "Blend eligibility threshold: Spearman rho < 0.99 on any horizon"

requirements-completed: [ANCHOR-01]

duration: 8min
completed: 2026-02-23
---

# Phase 4 Plan 01: Stronger Anchor — Reproduction Tooling Summary

**Configurable RSF+GBSA reproduction script + Spearman blend eligibility tool built; gated on user Kaggle notebook fork (Task 1 checkpoint pending)**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-02-23T05:20:54Z
- **Completed:** 2026-02-23T05:28:00Z
- **Tasks:** 1/2 complete (Task 1 = checkpoint:human-action, awaiting user)
- **Files modified:** 2

## Accomplishments

- exp30_reproduce_anchor.py: --config JSON + --out args, default=0.96624 settings, prints distribution stats per horizon
- exp30_compare_anchors.py: per-horizon Spearman, max/mean diff, ELIGIBLE/SKIP verdict, p48 flagged
- Verified: --help works; compare against submission_0.96624 vs exp17_reproduced prints correct Spearman + ELIGIBLE

## Task Commits

1. **Task 2: Build configurable reproduction + comparison scripts** - `4bf8377` (feat)

## Files Created/Modified

- `scripts/exp30_reproduce_anchor.py` - Configurable RSF+GBSA pipeline with JSON config
- `scripts/exp30_compare_anchors.py` - Per-horizon Spearman + blend eligibility (pre-existing, verified correct)

## Decisions Made

- Task 2 executed before Task 1 since tooling is independent of notebook artifacts
- exp30_compare_anchors.py already present with correct logic; no rewrite needed

## Deviations from Plan

None - plan executed as written. Task 1 is a human-action checkpoint, not a deviation.

## Issues Encountered

- exp30_compare_anchors.py already existed — verified it met plan requirements, kept as-is
- prob_72h Spearman = nan (both submissions constant 1.0) — expected behavior

## Next Phase Readiness

- Tooling ready: any notebook submission CSV can be compared against 0.96624 immediately
- Blocked on: user providing notebook IDs + artifact paths from Kaggle (Task 1)
- Once artifacts arrive: `python scripts/exp30_compare_anchors.py --sub_a submissions/submission_0.96624.csv --sub_b submissions/anchor_<id>/submission.csv`

---
*Phase: 04-stronger-anchor*
*Completed: 2026-02-23*
