# Roadmap — WiDS 2026 Push to 0.975+

## Milestone 1: Score 0.975+

### Phase 1: Stacking & Feature Baseline Fix
**Goal**: Remove known overfitting sources, establish clean baseline
**Requirements**: R2 (Feature Selection), R5 (Stacking Simplification)
**Plans:** 1/2 plans executed

Plans:
- [ ] 01-01-PLAN.md — LR-head baseline (config flags only, zero code changes)
- [ ] 01-02-PLAN.md — Backward elimination for optimal feature subset

**Success**: CV stable or improved; LB >= 0.96783 (current PB)
**Estimated submissions**: 2-3

### Phase 2: Model Diversity Ensemble (Rewritten)
**Goal**: Improve LB through IPCW-aware stacking and new calibration methods
**Requirements**: R1 (Model Diversity) — R6 (Seed Expansion) pre-completed (5 seeds already in train.py)
**Plans:** 2 plans

Plans:
- [ ] 02-01-PLAN.md — IPCW stacking with RSF+EST+GBSA and Ridge/LR meta-learners
- [ ] 02-02-PLAN.md — Calibration method comparison (isotonic/Platt/piecewise) for 24h/48h

**Success**: LB > 0.968 (PB=0.96783); R6 already complete (5 seeds)
**Estimated submissions**: 2-3
**Depends on**: Phase 1 closed (pivot strategy)

### Phase 3: Conformal Calibration
**Goal**: Reduce WBrier through SOTA calibration
**Requirements**: R3 (CSD/CiPOT), R4 (KM-Sampling)
**Tasks**:
- Implement CSD (Conformalized Survival Distribution) calibration
- Implement KM-sampling for censored data utilization
- Compare conformal vs current odds-scale calibration on OOF
- Apply best calibration method to submission pipeline
**Success**: WBrier improves >= 0.001 on OOF; LB improves
**Estimated submissions**: 3-5
**Depends on**: Phase 2 ensemble

### Phase 4: Integration & Fine-tuning
**Goal**: Combine all improvements, fine-tune for maximum score
**Requirements**: All
**Tasks**:
- Integrate Phase 1-3 improvements into unified pipeline
- Re-run logit-space (A,B) calibration on new base predictions
- Fine-tune gap-gated parameters for new distribution
- Final LB validation
**Success**: LB >= 0.975
**Estimated submissions**: 3-5
**Depends on**: Phase 3 calibration
