---
phase: 02-model-diversity-ensemble
plan: "01"
subsystem: stacking
tags: [ipcw, stacking, gbsa, meta-learner, gate-check]
dependency_graph:
  requires: []
  provides: [exp31_ipcw_stacking]
  affects: [submission]
tech_stack:
  added: []
  patterns: [IPCW weighting, two-stage CV, go/no-go gate]
key_files:
  created:
    - scripts/exp31_ipcw_stacking.py
  modified: []
decisions:
  - "Ridge meta-learner selected over LR for near-constant horizons (72h single-class fix)"
  - "Spearman gate uses test predictions from quick full-retrain, not OOF (dimension mismatch fix)"
metrics:
  duration_seconds: 269
  completed_date: "2026-02-22"
  tasks_completed: 2
  files_changed: 1
---

# Phase 2 Plan 01: IPCW Stacking Experiment Summary

IPCW-aware stacking with RSF+EST+GBSA base models, Ridge/LR meta-learners, two-stage CV, and go/no-go gate. Stage 1 OOF hybrid=0.96610 < gate threshold 0.9697 — no signal, no submission generated.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Implement IPCW stacking experiment script | b6e29a0 | scripts/exp31_ipcw_stacking.py |
| 2 | Generate submission if gate passes | b6e29a0 | scripts/exp31_ipcw_stacking.py (gate logic) |

## Results

Stage 1 (5x1 CV):
- Ridge OOF hybrid = 0.96610 (gate threshold = 0.9697) — BELOW threshold
- LR OOF hybrid = 0.90728 — well below
- Decision: "No signal -- stopping"

Stage 2 was not run (Stage 1 failed gate).

IPCW weight stats: min=1.000, max=20.000, p5=1.000, p95=1.147

## Deviations from Plan

**1. [Rule 1 - Bug] LR fails on single-class 72h horizon**
- Found during: Task 1 verification
- Fix: Skip LR fit when all eligible labels are one class, pass through constant
- Commit: b6e29a0

**2. [Rule 1 - Bug] Spearman dimension mismatch (train=221 vs test=95)**
- Found during: Task 1 verification
- Fix: Spearman gate uses test predictions from quick full-retrain, not OOF
- Commit: b6e29a0

## Interpretation

IPCW stacking shows no improvement at Stage 1 (0.96610 < 0.9697 gate). Consistent with Phase 1 finding that meta-learning on 221 samples overfits. IPCW weights mostly 1.0 (p95=1.147) — minimal censoring signal.

## Self-Check: PASSED
- scripts/exp31_ipcw_stacking.py: FOUND
- Commit b6e29a0: FOUND
