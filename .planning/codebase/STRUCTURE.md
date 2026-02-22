# Codebase Structure

**Analysis Date:** 2026-02-22

## Directory Layout

```
/d/wids/
├── src/                    # Core pipeline modules
│   ├── __init__.py
│   ├── train.py           # Main training orchestration
│   ├── config.py          # Paths, features, CV settings
│   ├── features.py        # Feature engineering
│   ├── models.py          # Base survival models
│   ├── labels.py          # Horizon label builder
│   ├── evaluation.py      # Metrics (Brier, C-index, hybrid)
│   ├── calibration.py     # Probability calibration
│   ├── ensemble.py        # Weighted ensemble
│   ├── stacking.py        # Per-horizon meta-learners
│   ├── monotonic.py       # Monotonicity enforcement
│   └── surv_post.py       # Survival function conversion
├── scripts/               # Experiment scripts
│   ├── exp16_step_a.py
│   ├── exp17_ablation.py
│   ├── exp18_subgroup_odds_scale.py
│   ├── exp19_cal_bagging.py
│   ├── exp21_candidate_bag.py
│   └── ... (other experiments)
├── submissions/           # Generated submission CSVs
├── notebooks/             # Jupyter notebooks (EDA, analysis)
├── logs/                  # Training logs
├── train.csv              # Training data
├── test.csv               # Test data
├── sample_submission.csv  # Submission template
├── metaData.csv           # Feature metadata
├── experiments.md         # Experiment log and results
└── CLAUDE.md              # Project instructions
```

## Directory Purposes

**src/:**
- Purpose: Core pipeline implementation
- Contains: Python modules for data, models, training, evaluation
- Key files: `train.py` (entry point), `config.py` (constants), `models.py` (7 model classes)

**scripts/:**
- Purpose: Experimental validation and ablation studies
- Contains: One-off scripts testing calibration methods, feature importance, hyperparameters
- Key files: `exp19_cal_bagging.py`, `exp21_candidate_bag.py` (recent experiments)

**submissions/:**
- Purpose: Store generated submission files
- Contains: CSV files with event_id and prob_*h columns
- Generated: By `src/train.py` after each full pipeline run

**notebooks/:**
- Purpose: Exploratory data analysis and visualization
- Contains: Jupyter notebooks for feature analysis, model debugging
- Key files: `eda.ipynb` (main EDA)

**logs/:**
- Purpose: Training run logs and diagnostics
- Contains: Text logs from pipeline execution
- Generated: By training scripts

## Key File Locations

**Entry Points:**
- `src/train.py`: Main training script - run with `python -m src.train`
- `scripts/exp*.py`: Individual experiment scripts

**Configuration:**
- `src/config.py`: All constants (paths, features, CV settings, horizons)
- `CLAUDE.md`: Project-level instructions and rules

**Core Logic:**
- `src/models.py`: 7 survival model implementations (RSF, EST, XGBoost, CoxPH, Weibull, RankXGB, MultiHorizonLGBM)
- `src/features.py`: Feature engineering pipeline
- `src/train.py`: CV loop, model training, calibration, submission generation
- `src/stacking.py`: Per-horizon meta-learner training
- `src/ensemble.py`: Weight optimization and ensemble prediction

**Testing & Evaluation:**
- `src/evaluation.py`: Brier score, C-index, hybrid score computation
- `src/labels.py`: Binary label construction per horizon

**Post-processing:**
- `src/calibration.py`: Odds scaling, Platt scaling
- `src/monotonic.py`: Monotonicity enforcement
- `src/surv_post.py`: Survival function to probability conversion

## Naming Conventions

**Files:**
- Core modules: lowercase with underscores (`train.py`, `models.py`)
- Experiment scripts: `exp{number}_{description}.py` (e.g., `exp19_cal_bagging.py`)
- Data files: lowercase with underscores (`train.csv`, `test.csv`)

**Directories:**
- Source code: `src/`
- Experiments: `scripts/`
- Output: `submissions/`, `logs/`
- Analysis: `notebooks/`

**Functions:**
- Public: lowercase with underscores (`build_horizon_labels()`, `enforce_monotonicity()`)
- Private: leading underscore (`_strat_labels()`, `_merge_rare_labels()`)
- Classes: PascalCase (`BaseSurvivalModel`, `CoxPH`, `RSF`)

**Variables:**
- Constants: UPPERCASE (`HORIZONS`, `N_SPLITS`, `RANDOM_STATE`)
- Data: lowercase (`train`, `test`, `X_train`, `y_time`)
- Predictions: `prob_dict`, `preds`, `oof_preds`

## Where to Add New Code

**New Feature:**
- Primary code: `src/features.py` - add to `add_engineered()` function
- Tests: `scripts/exp*.py` - create experiment script to validate
- Config: `src/config.py` - add to feature lists if needed

**New Model:**
- Implementation: `src/models.py` - inherit from `BaseSurvivalModel`
- Integration: `src/stacking.py` - add to `_make_base()` and `BASE_NAMES`
- Testing: `scripts/exp*.py` - create experiment to benchmark

**New Calibration Method:**
- Implementation: `src/calibration.py` - add function
- Integration: `src/train.py` - add to calibration pipeline
- Testing: `scripts/exp*.py` - create experiment to compare methods

**Utilities:**
- Shared helpers: `src/` - create new module if >100 lines, else add to existing
- Evaluation metrics: `src/evaluation.py`
- Post-processing: `src/monotonic.py` or `src/surv_post.py`

## Special Directories

**submissions/:**
- Purpose: Store submission CSVs for Kaggle
- Generated: Yes (by `src/train.py`)
- Committed: No (in .gitignore)

**logs/:**
- Purpose: Store training run logs
- Generated: Yes (by training scripts)
- Committed: No (in .gitignore)

**notebooks/:**
- Purpose: Exploratory analysis
- Generated: Yes (manual creation)
- Committed: Yes (tracked in git)

**.planning/codebase/:**
- Purpose: GSD codebase analysis documents
- Generated: Yes (by GSD mapper)
- Committed: Yes (tracked in git)

---

*Structure analysis: 2026-02-22*
