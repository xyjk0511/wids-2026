---
phase: 02-model-diversity-ensemble
plan: "02"
subsystem: calibration
tags: [calibration, isotonic, platt, piecewise-linear, oof, brier]
dependency_graph:
  requires: [submissions/submission_0.96624.csv]
  provides: [submissions/submission_exp32_cal.csv]
  affects: [Phase 3 conformal calibration]
tech_stack:
  added: []
  patterns: [5-fold CV calibration, anchor-incremental blending, independent calibration]
key_files:
  created: [scripts/exp32_calibration_methods.py, submissions/submission_exp32_cal.csv]
  modified: []
decisions:
  - "Platt scaling Track-B (independent) on 48h is best: dHybrid=+0.0059, CI stable"
  - "Track B (independent) consistently outperforms Track A across all methods"
  - "48h calibration yields larger gains than 24h across all methods"
metrics:
  duration_seconds: 172
  completed_date: "2026-02-22"
  tasks_completed: 2
  files_created: 2
---

# Phase 2 Plan 02: Calibration Methods Comparison Summary

Platt scaling applied independently to 48h predictions (Track B) yields +0.0059 OOF hybrid improvement with zero CI degradation over the RSF+EST 50/50 baseline.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Calibration comparison experiment | fb3fac4 | scripts/exp32_calibration_methods.py |
| 2 | Generate best calibration submission | fb3fac4 | submissions/submission_exp32_cal.csv |

## Results

### OOF Comparison (all 24 combinations)

| Method | Track | Horizon | WBrier | CI | Hybrid | dHybrid |
|--------|-------|---------|--------|-----|--------|---------|
| isotonic | B | 24h | 0.03982 | 0.9320 | 0.9517 | +0.0037 |
| platt | B | 24h | 0.03979 | 0.9320 | 0.9518 | +0.0038 |
| piecewise | B | 24h | 0.04084 | 0.9320 | 0.9510 | +0.0030 |
| isotonic | B | 48h | 0.03796 | 0.9320 | 0.9530 | +0.0050 |
| **platt** | **B** | **48h** | **0.03677** | **0.9320** | **0.9539** | **+0.0059** |
| piecewise | B | 48h | 0.03837 | 0.9320 | 0.9527 | +0.0048 |

Track A (alpha=0.1/0.2/0.3) consistently underperforms Track B across all methods.

### Baseline vs Best

| Metric | Baseline | Best Cal | Delta |
|--------|----------|----------|-------|
| Hybrid | 0.9480 | 0.9539 | +0.0059 |
| CI | 0.9320 | 0.9320 | 0.0000 |
| WBrier | 0.0452 | 0.0368 | -0.0084 |

Best config: Platt / Track B / 48h only. Submission: submissions/submission_exp32_cal.csv

## Decisions Made

1. Platt/Track-B/48h selected as best by OOF hybrid without CI degradation
2. Track B dominates Track A: direct calibration better than anchor-incremental blending
3. 48h gains larger than 24h — 48h probabilities have more calibration headroom

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- scripts/exp32_calibration_methods.py: FOUND
- submissions/submission_exp32_cal.csv: FOUND
- Commit fb3fac4: FOUND
