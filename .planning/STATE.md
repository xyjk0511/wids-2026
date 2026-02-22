# Project State — WiDS 2026

## Current Phase: 1 (In Progress)
## Milestone: 1 (Score 0.975+)

## Progress
- [x] Codebase mapping complete (7 documents)
- [x] PROJECT.md created
- [x] REQUIREMENTS.md created
- [x] ROADMAP.md created
- [~] Phase 1: Stacking & Feature Baseline Fix (IN PROGRESS — Plan 01 at checkpoint)
- [ ] Phase 2: Model Diversity Ensemble
- [ ] Phase 3: Conformal Calibration
- [ ] Phase 4: Integration & Fine-tuning

## Key Metrics
- PB: 0.96783 (Exp23, 2026-02-20)
- LR-head CV: Hybrid=0.9707, **LB=0.96274** (Plan 01, 2026-02-22) ← FAILED
- Target: 0.975+
- Submissions used: ~38
- Submissions remaining: >20

## Decisions
- Use .venv_sksurv22 Python env for all pipeline runs (has lifelines + sksurv)
- LR head CV=0.9707 vs XGB baseline 0.9720 — small drop, LB is true test

## Stopped At: 01-01 checkpoint FAILED — LB 0.96274 < 0.9670 threshold
## Last Updated: 2026-02-22
