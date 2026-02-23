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
  - "Track 3 grid failed: prediction distribution compressed (std=0.102 vs ref 0.364) — pipeline inconsistency, not hyperparameter choice"
  - "LB=0.91089 (R1) and LB=0.90860 (R2) catastrophically below 0.96624 — grid branch closed"
  - "Root cause: exp30_hyperparam_grid.py feature engineering/postprocessing diverged from reference pipeline"
  - "Current best remains 0.96624 public / 0.96681 private estimate"

requirements-completed: [ANCHOR-02, ANCHOR-03]

duration: ~5 min
completed: 2026-02-23
---

# Phase 4 Plan 02: Track 2+3 — Blend Gates + Hyperparam Grid Summary

**RSF hyperparam grid produced catastrophically compressed predictions (prob_48h std=0.102 vs ref 0.364) — LB=0.91089/0.90860, pipeline inconsistency identified; blend gate infrastructure built but never triggered**

## Performance

- Duration: ~2 sessions
- Started: 2026-02-22
- Completed: 2026-02-23
- Tasks: 3/3
- Files created: 2 scripts + 2 submission CSVs

## Accomplishments

- exp30_blend_anchors.py: 3-gate admission (LB, Spearman, distribution) + OOF coarse+fine weight search
- exp30_hyperparam_grid.py: 4 RSF configs + sksurv version check; top-2 saved as submission CSVs
- Submitted R1 and R2 to Kaggle; LB results confirmed catastrophic failure

Grid OOF results (rho_p48 vs ref 0.96624):

  n=500 mf=sqrt msl=3  rho_p48=0.9710  <- R1 submitted, LB=0.91089
  n=500 mf=0.5  msl=3  rho_p48=0.9702  <- R2 submitted, LB=0.90860
  n=500 mf=0.5  msl=5  rho_p48=0.9685
  n=200 mf=sqrt msl=3  rho_p48=0.9602
  REF   mf=0.5  msl=5  rho_p48=0.9572  (sksurv=0.22.2 version gap)

## Task Commits

1. Task 1: Blend admission gates + OOF weight search — fb42d7c
2. Task 2: RSF hyperparam grid — 1155ac4
3. Task 3: Review results + Kaggle LB — checkpoint:human-verify (no code commit)

## Deviations from Plan

None in implementation. Outcome was failure due to pipeline inconsistency discovered at submission time.

## Issues Encountered

**Critical: Prediction distribution compression (LB=0.91)**
- prob_48h std=0.102 in grid submissions vs std=0.364 in reference 0.96624
- Root cause: exp30_hyperparam_grid.py feature engineering and/or postprocessing diverged from reference pipeline
- Effect: predictions clustered near center, destroying ranking signal entirely
- Resolution: Track 3 closed. No further grid runs without auditing pipeline parity first.

## Next Phase Readiness

- Phase 4 fully closed. Both tracks failed to exceed 0.96624.
- Current best: 0.96624 public LB / 0.96681 private estimate
- Blend gate infrastructure reusable if a qualifying anchor appears
- Next work must audit pipeline parity before any grid/hyperparam experiments
