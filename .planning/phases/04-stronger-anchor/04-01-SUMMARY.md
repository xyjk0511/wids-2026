---
phase: 04-stronger-anchor
plan: 01
subsystem: tooling
tags: [rsf, gbsa, sksurv, anchor, reproduction, comparison, kaggle]

requires: []
provides:
  - "exp30_reproduce_anchor.py: configurable RSF+GBSA pipeline, default=0.96624 settings"
  - "exp30_compare_anchors.py: per-horizon Spearman + blend eligibility verdict"
  - "Track 1 result: suman2208 LB=0.96086 — below gate, Track 1 closed"
  - "OOF artifacts in submissions/anchor_suman2208/ for future reference"
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
  - "Track 1 closed: suman2208 LB=0.96086 < 0.96624 gate, stop-loss triggered"
  - "rhythmghai/ridge-stacker private (403) — no other accessible notebooks above 0.966"
  - "Switching to Track 3 (04-02): RSF hyperparam grid"

requirements-completed: [ANCHOR-01]

duration: ~2 sessions
completed: 2026-02-23
---

# Phase 4 Plan 01: Stronger Anchor — Reproduction Tooling + Track 1 Summary

**Reproduction tooling built; Track 1 closed after suman2208 scored LB=0.96086 (below 0.96624 gate); no other accessible notebooks above 0.966; switching to Track 3 RSF hyperparam grid**

## Performance

- **Duration:** ~2 sessions
- **Started:** 2026-02-23T05:20:54Z
- **Completed:** 2026-02-23T05:56:00Z
- **Tasks:** 2/2 complete
- **Files modified:** 2 scripts + 3 artifacts

## Accomplishments

- exp30_reproduce_anchor.py: --config JSON + --out args, default=0.96624 settings, prints distribution stats per horizon
- exp30_compare_anchors.py: per-horizon Spearman, max/mean diff, ELIGIBLE/SKIP verdict, p48 flagged
- Track 1 executed: suman2208 forked, ran on Kaggle, LB=0.96086 — stop-loss triggered (< 0.96624 gate)
- Artifacts saved: submissions/anchor_suman2208/ (submission.csv, oof_preds.csv, run_metadata.txt)

## Task Commits

1. **Task 2: Build configurable reproduction + comparison scripts** - `4bf8377` (feat)
2. **Task 1: Fork + run suman2208 on Kaggle** - human action (artifacts saved, no code commit)

## Files Created/Modified

- `scripts/exp30_reproduce_anchor.py` - Configurable RSF+GBSA pipeline with JSON config
- `scripts/exp30_compare_anchors.py` - Per-horizon Spearman + blend eligibility (pre-existing, verified correct)
- `submissions/anchor_suman2208/submission.csv` - 95 rows, prob_48h std=0.414
- `submissions/anchor_suman2208/oof_preds.csv` - 221 rows, 12 cols
- `submissions/anchor_suman2208/run_metadata.txt` - lb_score: 0.96086

## Decisions Made

- Task 2 executed before Task 1 since tooling is independent of notebook artifacts
- Track 1 stop-loss triggered: LB=0.96086 < 0.96624 gate on first submission
- rhythmghai/ridge-stacker was private (403) — no other candidates accessible above 0.966
- Proceeding to Track 3 (04-02): RSF hyperparam grid search

## Deviations from Plan

None - plan executed exactly as written. Stop-loss rule was pre-defined and triggered correctly.

## Issues Encountered

- rhythmghai/ridge-stacker returned 403 (private) — highest-ranked candidate inaccessible
- suman2208 scored LB=0.96086, below the 0.96624 gate — Track 1 closed per stop-loss rule

## Next Phase Readiness

- Track 3 (04-02) is the active path: RSF hyperparam grid search
- Reproduction tooling ready to ingest any config for Track 3 experiments
- Submissions remaining: ~17 (1 used in Track 1)

---
*Phase: 04-stronger-anchor*
*Completed: 2026-02-23*
