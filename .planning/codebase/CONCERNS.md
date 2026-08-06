# Codebase Concerns

**Analysis Date:** 2026-02-22

## Tech Debt

**Hardcoded Magic Numbers in Feature Engineering:**
- Issue: Multiple hardcoded constants scattered across `src/features.py` without centralized configuration
- Files: `src/features.py` (lines 55-56, 68, 85)
- Impact: Difficult to tune hyperparameters; values like sigmoid center=2600, scale=1000, exp decay 5000m are not easily adjustable
- Fix approach: Extract magic numbers to `src/config.py` as named constants

**Hardcoded 72h Probability to 1.0:**
- Issue: 72h horizon always set to 1.0 in multiple places without justification
- Files: `src/models.py` (lines 307, 377, 432, 498), `src/monotonic.py` (lines 120, 172), `src/train.py` (line 319)
- Impact: Inflexible; if competition rules change, requires code changes across multiple files
- Fix approach: Create `src/config.py` constant `PROB_72H_FIXED = 1.0` and reference it everywhere

**Inconsistent Seed Management:**
- Issue: Multiple seed lists defined in different modules without coordination
- Files: `src/models.py` (line 13), `src/train.py` (line 616)
- Impact: Seeds are duplicated; changing one list doesn't update others
- Fix approach: Centralize all seed lists in `src/config.py`

## Known Bugs

**Probability Distribution Mismatch (Full Retrain vs CV):**
- Symptoms: Full retrain produces different probability distributions than CV fold averaging
- Files: `src/train.py` (lines 615-683 full retrain vs lines 105-190 CV)
- Trigger: Running with `--decoupled` flag; test predictions differ from OOF
- Root cause: Full retrain sees all data at once; CV folds see partial data; different tree splits result

**Platt Scaling Instability with Small Calibration Sets:**
- Symptoms: Calibration fails silently when `cal_elig.sum() < 5` in stacking
- Files: `src/stacking.py` (lines 305-311)
- Trigger: Rare horizons with few eligible samples
- Impact: Some folds may have uncalibrated heads while others are calibrated

**Rank Transform Numerical Instability:**
- Symptoms: `_to_rank01()` in `src/stacking.py` doesn't handle ties explicitly
- Files: `src/stacking.py` (line 59)
- Trigger: When many samples have identical predictions
- Impact: Rank ties broken arbitrarily; can cause OOF variance across runs

## Security Considerations

**No Input Validation on Feature Sets:**
- Risk: `get_feature_set()` in `src/features.py` silently filters missing columns without warning
- Files: `src/features.py` (lines 96-128)
- Recommendations: Add logging when features are missing; raise error if critical features absent

**Hardcoded File Paths:**
- Risk: `src/config.py` uses hardcoded paths (lines 4-10) that assume specific directory structure
- Files: `src/config.py`
- Recommendations: Add validation that required files exist at startup

## Performance Bottlenecks

**Grid Search Complexity in Decoupled Path:**
- Problem: `run_cv_decoupled()` performs 11×11×11 = 1331 weight combinations per horizon
- Files: `src/train.py` (lines 250-266, 289-306)
- Cause: Exhaustive search; no early stopping
- Improvement path: Use Bayesian optimization or random search with early stopping

**Inner CV in Stacking:**
- Problem: `train_horizon_heads()` runs inner CV (3 splits) inside outer CV (15 folds) = 45 inner CV runs
- Files: `src/stacking.py` (lines 103-123, 226-229)
- Cause: Anti-leakage design requires this
- Improvement path: Parallelize inner CV across folds

## Fragile Areas

**Stratification Label Merging Logic:**
- Files: `src/train.py` (lines 48-75)
- Why fragile: `_merge_rare_labels()` uses while loop with complex heuristic
- Safe modification: Add unit tests for edge cases
- Test coverage: No tests exist for this function

**Horizon Head Fallback to 0.5:**
- Files: `src/stacking.py` (lines 262-266)
- Why fragile: When `fit_elig.sum() < 5`, head defaults to 0.5; silent failure mode
- Safe modification: Log warning when fallback occurs
- Test coverage: No tests for insufficient eligible samples

**Monotonicity Projection with Weights:**
- Files: `src/monotonic.py` (lines 73-106)
- Why fragile: `project_monotone_l2()` uses PAVA; if weights are zero or negative, behavior undefined
- Safe modification: Add assertions that weights are positive
- Test coverage: No tests for edge cases

**Base Model Prediction Averaging:**
- Files: `src/train.py` (lines 647-652)
- Why fragile: Assumes all seeds produce predictions; if one seed fails, entire averaging breaks
- Safe modification: Add try-catch per seed; log failures
- Test coverage: No error handling for individual seed failures

## Scaling Limits

**Memory Usage in Full Retrain:**
- Current capacity: ~5 seeds × 4 models × 4 horizons × 2 datasets = 160 model instances
- Limit: If dataset grows 10x or more seeds added, memory could exceed 16GB
- Scaling path: Implement seed-by-seed prediction aggregation (predict, save, clear)

**CV Fold Count:**
- Current: 5 splits × 10 repeats = 50 folds
- Limit: Each fold retrains 4 models; 50 folds × 4 models = 200 model fits; runtime ~30-60 min
- Scaling path: Reduce repeats to 3-5 for faster iteration

## Dependencies at Risk

**scikit-survival (sksurv) Version Pinning:**
- Risk: `src/models.py` imports from `sksurv.ensemble`; no version constraint
- Impact: API changes in sksurv could break RSF/EST models
- Migration plan: Pin to `sksurv>=0.20`

**XGBoost Objective Function:**
- Risk: `src/models.py` uses `survival:cox` objective (line 589); may be deprecated
- Impact: Future XGBoost versions may remove this objective
- Migration plan: Monitor XGBoost changelog

**Lifelines Compatibility:**
- Risk: `src/models.py` uses CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter
- Impact: Lifelines API changes could break these models
- Migration plan: Pin to `lifelines>=0.27`

## Test Coverage Gaps

**Untested Stratification Logic:**
- What's not tested: `_strat_labels()` and `_merge_rare_labels()` functions
- Files: `src/train.py` (lines 78-97, 48-75)
- Risk: Edge cases could cause silent failures
- Priority: High (affects CV split validity)

**Untested Monotonicity Enforcement:**
- What's not tested: `enforce_monotonicity()` and `project_monotone_l2()` with edge cases
- Files: `src/monotonic.py` (lines 10-26, 73-106)
- Risk: Violations could slip through if logic is modified
- Priority: High (affects submission validity)

**Untested Calibration Fallback:**
- What's not tested: Behavior when calibration set is too small
- Files: `src/stacking.py` (lines 134-176)
- Risk: Silent fallback to uncalibrated predictions
- Priority: Medium (affects head OOF quality)

**Untested Probability Clipping:**
- What's not tested: Edge cases in `_validate_and_save()` (floor/ceiling violations, NaN handling)
- Files: `src/train.py` (lines 497-556)
- Risk: Invalid submissions if clipping logic fails
- Priority: High (affects submission validity)

**No Integration Tests:**
- What's not tested: Full pipeline from data load to submission generation
- Files: All of `src/train.py`
- Risk: Changes to one module could break downstream modules
- Priority: Medium (would catch integration bugs early)

---

*Concerns audit: 2026-02-22*
