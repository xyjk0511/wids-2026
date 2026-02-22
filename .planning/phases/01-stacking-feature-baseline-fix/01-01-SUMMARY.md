---
phase: 01-stacking-feature-baseline-fix
plan: 01
subsystem: modeling
tags: [logistic-regression, stacking, calibration, survival, cv-baseline]

requires: []
provides:
  - LR-head baseline CV score (Hybrid=0.9707) with --head-model logit --head-base-only --calibration-mode none
  - Submission CSV for LB validation
affects: [02-feature-selection, 03-model-diversity]

tech-stack:
  added: []
  patterns:
    - "LR stacking head with base-model predictions only (9 cols: 3 models x 3 horizons)"
    - "Calibration bypassed via --calibration-mode none"

key-files:
  created:
    - submissions/submission.csv
  modified: []

key-decisions:
  - "Use .venv_sksurv22 Python environment (has lifelines + sksurv, .venv does not)"
  - "LR head CV Hybrid=0.9707 vs SurvBlend baseline 0.9720 -- small drop, LB is true test"

patterns-established:
  - "Run pipeline with: .venv_sksurv22/Scripts/python -m src.train <flags>"

requirements-completed: []

duration: 8min
completed: 2026-02-22
---

# Phase 01 Plan 01: LR-Head Baseline Summary

**LR stacking head (logit, base-only, no calibration) CV Hybrid=0.9707 -- submission ready for LB validation**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-02-22T03:16:31Z
- **Completed:** 2026-02-22T03:24:00Z
- **Tasks:** 1/2 (paused at checkpoint)
- **Files modified:** 0 (run-and-record task)

## Accomplishments
- Pipeline ran to completion: --feature-level v96624_plus --head-model logit --head-base-only --calibration-mode none
- Head OOF: Hybrid=0.9707, CI=0.9391, WBrier=0.0158 (B24h=0.0282, B48h=0.0183, B72h=0.0000)
- SurvBlend baseline: Hybrid=0.9720
- No NaN, all probs in [0,1], monotonicity PASS, 72h all-ones PASS
- Submission CSV at submissions/submission.csv

## Task Commits

1. **Task 1: Run LR-head baseline and record CV** - bfd6dc7 (feat)

## Files Created/Modified
- submissions/submission.csv - LR-head baseline submission for Kaggle LB upload

## Decisions Made
- Used .venv_sksurv22 Python environment -- .venv is missing lifelines module
- CV drop vs XGB head is small (0.9707 vs 0.9720); proceeding to LB validation as planned

## Deviations from Plan

**1. [Rule 3 - Blocking] Used correct Python venv with lifelines installed**
- Found during: Task 1
- Issue: ModuleNotFoundError: No module named 'lifelines' when using default python
- Fix: Used .venv_sksurv22/Scripts/python which has lifelines and sksurv
- Files modified: None (runtime fix only)
- Committed in: bfd6dc7

**Total deviations:** 1 auto-fixed (blocking -- wrong venv)

## Issues Encountered
- .venv missing lifelines -- must use .venv_sksurv22 for all pipeline runs
- 12h median=0.019 vs target ~0.15 -- compression present, LB will determine if acceptable

## Next Phase Readiness
- Submission CSV ready for Kaggle upload
- Awaiting LB score: >= 0.9670 proceed to Plan 02, < 0.9670 investigate C/class_weight

---
*Phase: 01-stacking-feature-baseline-fix*
*Completed: 2026-02-22*
