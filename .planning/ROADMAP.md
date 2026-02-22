# Roadmap — WiDS 2026 Push to 0.975+

## Milestone 1: Score 0.975+

### Phase 1: Stacking & Feature Baseline Fix
**Goal**: Remove known overfitting sources, establish clean baseline
**Requirements**: R2 (Feature Selection), R5 (Stacking Simplification)
**Tasks**:
- Replace XGB stacking heads with LogisticRegression/Isotonic
- Feature ablation: find optimal subset between 16-24 features
- Run CV + submit to LB to establish new clean baseline
**Success**: CV stable or improved; LB >= 0.96783 (current PB)
**Estimated submissions**: 2-3

### Phase 2: Model Diversity Ensemble
**Goal**: Improve CI through model diversity
**Requirements**: R1 (Model Diversity), R6 (Seed Expansion)
**Tasks**:
- Activate CoxPH and WeibullAFT in ensemble pipeline
- Add XGBoostCox (survival:cox objective) to base models
- Expand seed averaging from 3 to 5 seeds
- Re-optimize ensemble weights with new model pool
**Success**: OOF CI improves >= 0.002; LB improves
**Estimated submissions**: 3-4
**Depends on**: Phase 1 baseline

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
