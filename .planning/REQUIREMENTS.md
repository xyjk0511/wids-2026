# Requirements — WiDS 2026 Push to 0.975+

## Success Criteria
- Kaggle LB hybrid score >= 0.975
- CI(12h) >= 0.955 AND WBrier <= 0.012
- CV-LB gap maintained <= 0.015

## R1: Model Diversity for CI
**Priority**: High
**Rationale**: Only RSF+EST used; CoxPH/WeibullAFT/XGBoostCox available but dormant. Adding diverse models improves ranking (CI).
**Acceptance**: Ensemble includes >= 3 model types (RSF+EST+GBSA); CI improves on OOF. (Revised from >= 4: CoxPH, WeibullAFT, XGBoostCox experimentally invalidated in Exp27/28/30 — CI too low or unstable on N=221.)
**Risk**: More models may increase CV-LB gap on 221 samples.

## R2: Feature Selection Optimization
**Priority**: High
**Rationale**: 16→36 features caused CV -0.0057 (Exp15). Need principled selection.
**Acceptance**: Identify optimal feature subset (likely 18-24 features); CV >= current baseline.
**Risk**: Feature importance unstable with small N.

## R3: Distribution Calibration (CSD/CiPOT)
**Priority**: High
**Rationale**: WBrier has 2.3x leverage over CI. Conformal methods preserve ranking while improving calibration.
**Acceptance**: WBrier improves on OOF without CI degradation.
**Risk**: Conformal methods need held-out calibration set from already-small data.

## R4: KM-Sampling for Censored Data
**Priority**: Medium
**Rationale**: Censored observations currently underutilized in calibration. KM-reweighting expands effective sample size.
**Acceptance**: Calibration uses censored data; WBrier improves.
**Risk**: Implementation complexity; may not help if censoring is informative.

## R5: Stacking Head Simplification
**Priority**: Medium
**Rationale**: XGB heads (200 trees, depth=3) on ~136 samples likely overfit. Simpler heads may generalize better.
**Acceptance**: Replace with LogisticRegression or isotonic; CV stable or improved.
**Risk**: May lose nonlinear signal capture.

## R6: Seed Expansion & Variance Reduction
**Priority**: Low
**Rationale**: 3 seeds → 5+ reduces prediction variance. Exp8 showed +0.00035 LB from seed averaging.
**Acceptance**: 5+ seeds; LB improves or holds.
**Risk**: Diminishing returns; longer runtime.

## Non-Requirements
- Deep learning models (insufficient data for 221 samples)
- External data sources (competition rules)
- Full pipeline rewrite (preserve anchor framework)
