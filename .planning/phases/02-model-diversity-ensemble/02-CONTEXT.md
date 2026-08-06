# Phase 2 Context: Model Diversity Ensemble (Rewritten)

> Generated from discuss-phase session 2026-02-22

## Phase Redefinition

Original Phase 2 tasks (CoxPH, WeibullAFT, XGBoostCox) ALL experimentally invalidated (Exp27/28/30). Seed expansion 3→5 already completed. Phase 2 is **rewritten** with new tasks.

**New Goal**: Improve LB through IPCW-aware stacking and new calibration methods.
**Success Criteria**: LB > 0.968 (currently PB=0.96783)

---

## Decision 1: IPCW Stacking

**What**: Censoring-aware meta-learning using KM inverse probability weights.

**Locked decisions**:
- **Meta-learner**: Compare Ridge(alpha=1.0) vs LR(C=0.01), select by OOF
- **Base model pool**: RSF + EST + GBSA (3 models, 12 meta-features)
- **CV protocol**: Two-stage — 5×1 quick check, 5×10 only if signal
- **Go/no-go gate**: OOF hybrid > 0.9697 AND Spearman vs 0.96624 stable

---

## Decision 2: Calibration Strategy

**What**: New calibration methods beyond logit (A,B) which plateaued at lam=6.0.

**Locked decisions**:
- **Scope**: New methods (isotonic/Platt/piecewise linear)
- **Target horizons**: 24h and 48h only
- **Anchor strategy**: Two tracks — anchor-incremental + independent calibration

---

## Deferred Ideas

- 12h ranking improvement (separate phase if needed)
- Phase 3 conformal calibration (CSD/CiPOT)
