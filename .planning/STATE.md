# Project State — WiDS 2026

## Current Phase: 2 (IN PROGRESS — Plan 02 complete)
## Milestone: 1 (Score 0.975+)

## Progress
- [x] Codebase mapping complete (7 documents)
- [x] PROJECT.md created
- [x] REQUIREMENTS.md created
- [x] ROADMAP.md created
- [x] Phase 1: Stacking & Feature Baseline Fix — CLOSED (head overfits, both plans failed)
- [~] Phase 2: Model Diversity Ensemble (Plan 02 complete — Platt/B/48h +0.0059 hybrid)
- [ ] Phase 3: Conformal Calibration
- [ ] Phase 4: Integration & Fine-tuning

## Key Metrics
- PB: 0.96783 (Exp23, 2026-02-20)
- Plan 01 LR-head: LB=0.96274 ← FAILED
- Plan 02 XGB-head 19feat: LB=0.95511 ← FAILED (catastrophic)
- Target: 0.975+
- Submissions used: ~40
- Submissions remaining: >18

## Decisions
- Use .venv_sksurv22 Python env for all pipeline runs (has lifelines + sksurv)
- **Stacking heads abandoned**: 221 samples too small, XGB/LR heads overfit severely
- RSF+EST baseline + logit post-processing remains best path (PB=0.96783)
- **Platt/Track-B/48h best calibration**: dHybrid=+0.0059, CI stable (02-02)
- Track B (independent) dominates Track A (anchor-incremental) across all methods

## Stopped At: Phase 2 Plan 02 complete — submission_exp32_cal.csv ready for LB
## Last Updated: 2026-02-22
