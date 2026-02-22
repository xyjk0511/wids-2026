# Architecture

**Analysis Date:** 2026-02-22

## Pattern Overview

**Overall:** Multi-model survival analysis ensemble with calibration pipeline

**Key Characteristics:**
- Survival prediction for 4 horizons (12h, 24h, 48h, 72h)
- Multiple base models (RSF, EST, XGBoost, CoxPH, Weibull) with weighted ensemble
- Per-horizon stacking heads for meta-learning
- Monotonicity enforcement and probability calibration
- Cross-validation with stratified splits on event/time bins

## Layers

**Data Layer:**
- Purpose: Load, preprocess, and feature engineering
- Location: `src/config.py`, `src/features.py`
- Contains: Data paths, feature definitions, redundancy removal, engineered features
- Depends on: Raw CSV files (train.csv, test.csv)
- Used by: Training pipeline

**Feature Engineering Layer:**
- Purpose: Transform raw features and create derived signals
- Location: `src/features.py`
- Contains: `remove_redundant()`, `add_engineered()`, `get_feature_set()`
- Depends on: Config feature lists
- Used by: Model training

**Model Layer:**
- Purpose: Base survival models with unified interface
- Location: `src/models.py`
- Contains: `BaseSurvivalModel` ABC, RSF, EST, XGBoostAFT, CoxPH, WeibullAFT, RankXGB
- Depends on: scikit-learn, lifelines, xgboost, sksurv
- Used by: Training pipeline, stacking

**Label Layer:**
- Purpose: Convert survival targets to binary classification labels per horizon
- Location: `src/labels.py`
- Contains: `build_horizon_labels()` - creates eligible masks and binary targets
- Depends on: Time and event arrays
- Used by: Stacking heads, evaluation

**Evaluation Layer:**
- Purpose: Compute competition metrics (Brier, C-index, hybrid score)
- Location: `src/evaluation.py`
- Contains: `horizon_brier_score()`, `c_index()`, `hybrid_score()`, `combined_score()`
- Depends on: Survival targets and predictions
- Used by: CV scoring, weight optimization

**Calibration Layer:**
- Purpose: Adjust probability distributions for better calibration
- Location: `src/calibration.py`
- Contains: `platt_scaling()`, `odds_scale()`, `fit_odds_scale_brier()`
- Depends on: Raw probabilities and labels
- Used by: Post-processing pipeline

**Ensemble Layer:**
- Purpose: Combine multiple model predictions with optimized weights
- Location: `src/ensemble.py`
- Contains: `ensemble_predict()`, `optimize_weights()`, `stack_meta_learner()`
- Depends on: Individual model predictions
- Used by: Final submission

**Stacking Layer:**
- Purpose: Train per-horizon meta-learners on base model outputs
- Location: `src/stacking.py`
- Contains: `train_horizon_heads()`, `predict_horizon_heads()`, anti-leakage CV design
- Depends on: Base model predictions, labels
- Used by: Training pipeline

**Post-processing Layer:**
- Purpose: Enforce monotonicity and apply final calibration
- Location: `src/monotonic.py`, `src/surv_post.py`
- Contains: `enforce_monotonicity()`, `submission_postprocess()`, survival function conversion
- Depends on: Ensemble predictions
- Used by: Final submission generation

**Training Orchestration:**
- Purpose: Coordinate entire pipeline from data to submission
- Location: `src/train.py`
- Contains: `load_data()`, `_strat_labels()`, CV loop, model training, calibration, submission
- Depends on: All layers above
- Used by: CLI entry point

## Data Flow

**Training Pipeline:**

1. Load train/test data → apply feature engineering (remove redundant, add engineered)
2. Build stratification labels (event/time bins) for RepeatedStratifiedKFold
3. For each outer fold:
   - Train base models (RSF, EST, XGBoost, CoxPH, Weibull) on fold training set
   - Generate OOF predictions on fold validation set
   - For each horizon: train stacking head (LogisticRegression/IsotonicRegression) on OOF
4. Optimize ensemble weights using hybrid score on validation OOF
5. Apply calibration (odds scaling or Platt scaling) per horizon
6. Enforce monotonicity (prob_12h ≤ prob_24h ≤ prob_48h ≤ prob_72h)
7. Generate submission CSV

**Prediction Pipeline:**

1. Load test data → apply same feature engineering as training
2. Generate base model predictions on full test set
3. Apply stacking heads to base predictions
4. Ensemble with optimized weights
5. Apply calibration and monotonicity enforcement
6. Output submission

**State Management:**
- CV folds: RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
- Stratification: Composite labels (event_flag + time_bin) to maintain class balance
- OOF predictions: Accumulated across folds for weight optimization
- Calibration state: Per-horizon calibrators (odds scale or Platt) fitted on validation data

## Key Abstractions

**BaseSurvivalModel:**
- Purpose: Unified interface for all survival models
- Examples: `src/models.py` - RSF, EST, XGBoostAFT, CoxPH, WeibullAFT, RankXGB
- Pattern: Abstract base class with `fit()` and `predict_proba()` methods
- Returns: `{horizon: np.ndarray}` probability dictionary

**Horizon-based Prediction:**
- Purpose: Multi-horizon survival prediction (12h, 24h, 48h, 72h)
- Examples: All models return `dict[int, np.ndarray]` keyed by horizon
- Pattern: Enables per-horizon calibration and monotonicity enforcement

**Stratification Labels:**
- Purpose: Maintain class balance in CV splits
- Examples: `src/train.py` - `_strat_labels()` creates composite labels
- Pattern: Combines event flag (0/1) with time bins (0-3) → 8 classes

**Calibration Transformations:**
- Purpose: Adjust raw probabilities to match true event rates
- Examples: `src/calibration.py` - odds_scale, platt_scaling
- Pattern: Monotone transformations preserve ranking while adjusting magnitude

## Entry Points

**Main Training Script:**
- Location: `src/train.py`
- Triggers: `python -m src.train` or direct import
- Responsibilities:
  - Parse CLI arguments (feature_level, strat_mode, etc.)
  - Execute full CV pipeline
  - Generate submission.csv
  - Print CV metrics and LB feedback

**Experiment Scripts:**
- Location: `scripts/exp*.py`
- Triggers: `python scripts/exp*.py`
- Responsibilities: Test specific hypotheses (calibration methods, feature ablations, etc.)

## Error Handling

**Strategy:** Explicit validation at boundaries with informative messages

**Patterns:**
- Feature validation: Check feature existence before model training
- Label validation: Ensure eligible masks are non-empty for each horizon
- Probability bounds: Clip to [0, 1] after all transformations
- Stratification: Merge rare labels to maintain minimum fold size
- Survival function evaluation: Handle boundary cases with clip/strict/left_survival_one policies

## Cross-Cutting Concerns

**Logging:** Print-based progress tracking in `src/train.py` (fold counts, metric summaries)

**Validation:**
- Feature set consistency across train/test
- Probability monotonicity enforcement
- Eligible sample counts per horizon

**Reproducibility:**
- Fixed RANDOM_STATE=42 in config
- Seed averaging (SEED_AVG_SEEDS=[42, 123, 456]) for ensemble stability
- RepeatedStratifiedKFold for deterministic splits

---

*Architecture analysis: 2026-02-22*
