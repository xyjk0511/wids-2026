# Phase 1: Stacking & Feature Baseline Fix - Context

**Gathered:** 2026-02-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove known overfitting sources in stacking heads and feature set, establish clean baseline. Success = CV stable or improved, LB >= 0.96783 (current PB). No new model types added — base model diversity is Phase 2.

</domain>

<decisions>
## Implementation Decisions

### Head Replacement Strategy
- Replace XGB heads (200 trees, depth=3) with LogisticRegression (C=1.0, balanced class_weight, lbfgs solver)
- Remove post-head calibration layer (Platt/Isotonic) — LR outputs probabilities directly
- Keep inner CV anti-leakage design (3-fold inner CV for base_train_oof)

### Feature Selection Method
- Start from V96624_PLUS feature set (21 features) as baseline
- Backward elimination using permutation importance ranking
- Evaluation metric: Hybrid Score (CI + WBrier combined)
- All horizons share the same feature subset (no per-horizon selection)

### Baseline Validation Criteria
- CV + LB dual confirmation required for adopting changes
- CV drop tolerance: up to 0.001 acceptable if LB improves
- Submission cadence and rollback strategy: Claude's discretion

### Stacking Architecture Simplification
- Keep two-layer structure (base models → LR head), only replace head
- Base models stay as current 3: RSF, EST, XGBCox
- Head input: only base model predictions (`use_orig_features=False`)
- Base feature representation: raw probabilities (`base_feature_mode="raw"`)

### Claude's Discretion
- Submission cadence (per-change vs batched)
- Rollback strategy when LB drops
- Exact stopping criterion for backward elimination (e.g., stop when any removal hurts > threshold)

</decisions>

<specifics>
## Specific Ideas

- LR head with only raw base predictions reduces input from ~30+ to 9 features (3 models x 3 horizons), dramatically reducing overfitting risk on 136-sample folds
- V96624_PLUS includes engineered features (has_growth, is_approaching, log_dist_min) plus top-5 permutation importance features — a curated starting point

</specifics>

<deferred>
## Deferred Ideas

- Adding CoxPH/WeibullAFT to base models — Phase 2
- Per-horizon feature selection — could revisit if shared set underperforms
- Conformal calibration methods (CSD/CiPOT) — Phase 3

</deferred>

---

*Phase: 01-stacking-feature-baseline-fix*
*Context gathered: 2026-02-22*
