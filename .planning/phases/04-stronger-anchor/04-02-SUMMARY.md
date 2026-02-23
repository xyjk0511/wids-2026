---
phase: 04-stronger-anchor
plan: 02
subsystem: hyperparam-grid
tags: [rsf, gbsa, hyperparam, blend, grid-search, kaggle]

requires: ["04-01"]
provides:
  - "exp30_blend_anchors.py: 3-gate admission + OOF weight search"
  - "exp30_hyperparam_grid.py: 4 RSF configs + version check, top-2 submissions saved"
  - "submission_exp30_grid_r1.csv: n=500 mf=sqrt msl=3, rho_p48=0.9710"
  - "submission_exp30_grid_r2.csv: n=500 mf=0.5 msl=3, rho_p48=0.9702"
affects: []

tech-stack:
  added: []
  patterns:
    - "Blend admission gates: lb_b > ref, rho48 in [0.90,0.99], distribution not collapsed"
    - "Grid sorted by rho deviation from 1.0 — most different ranking = most interesting"

key-files:
  created:
    - scripts/exp30_blend_anchors.py
    - scripts/exp30_hyperparam_grid.py
    - submissions/submission_exp30_grid_r1.csv
    - submissions/submission_exp30_grid_r2.csv
  modified: []

key-decisions:
  - "Gate 1 correctly fails for suman2208 (LB=0.96086 < 0.96624) — no blend produced"
  - "Grid sorted by rho deviation: most different p48 ranking = best LB candidate"
  - "n=500 mf=sqrt msl=3 is top candidate (rho_p48=0.9710 vs ref)"
  - "sksurv version check: ref config on 0.22.2 gives rho_p48=0.9572 — version gap exists"

requirements-completed: [ANCHOR-02, ANCHOR-03]

duration: ~5 min
completed: 2026-02-23
---

# Phase 4 Plan 02: Track 2+3 — Blend Gates + Hyperparam Grid Summary

**Blend script enforces 3 admission gates (Gate 1 fails for current anchor); RSF hyperparam grid completed 4 configs — top candidate n=500/sqrt/msl=3 has rho_p48=0.9710 vs reference**

## Performance

- Duration: ~5 min
- Started: 2026-02-23T05:58:24Z
- Completed: 2026-02-23T06:03:11Z
- Tasks: 2/3 complete (Task 3 = checkpoint, awaiting human review)
- Files created: 2 scripts + 2 submission CSVs

## Accomplishments

- exp30_blend_anchors.py: Gate 1/2/3 admission + OOF coarse+fine weight search; exits cleanly on any gate failure
- exp30_hyperparam_grid.py: 4 RSF configs + Run 5 version check; top-2 saved as submission CSVs

Grid results (rho_p48 vs ref 0.96624):

  n=500 mf=sqrt msl=3  rho_p48=0.9710  <- top candidate
  n=500 mf=0.5  msl=3  rho_p48=0.9702  <- second candidate
  n=500 mf=0.5  msl=5  rho_p48=0.9685
  n=200 mf=sqrt msl=3  rho_p48=0.9602
  REF   mf=0.5  msl=5  rho_p48=0.9572  (sksurv=0.22.2 version gap)

## Task Commits

1. Task 1: Blend admission gates + OOF weight search — fb42d7c
2. Task 2: RSF hyperparam grid — 1155ac4

## Deviations from Plan

None — plan executed exactly as written. Gate 1 failure for suman2208 was expected per context.

## Next Steps (awaiting human review — Task 3)

- Submit submission_exp30_grid_r1.csv (n=500/sqrt/msl=3) to Kaggle
- Submit submission_exp30_grid_r2.csv (n=500/0.5/msl=3) if quota allows
- Report LB scores to determine if any config beats 0.96624
