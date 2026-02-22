# Technology Stack

**Analysis Date:** 2026-02-22

## Languages

**Primary:**
- Python 3.x - All source code, training pipeline, model implementations

## Runtime

**Environment:**
- Python 3.x (via `.venv` and `.venv_sksurv22` virtual environments)

**Package Manager:**
- pip
- Lockfile: Not detected (requirements.txt not found in repo root)

## Frameworks

**Core ML/Survival Analysis:**
- scikit-survival (sksurv) - Random Survival Forest (RSF), Extra Survival Trees (EST), Gradient Boosting Survival Analysis (GBSA)
- lifelines - Cox Proportional Hazards, Weibull AFT, LogNormal AFT, Kaplan-Meier estimation, concordance index
- scikit-learn (sklearn) - StandardScaler, LogisticRegression, IsotonicRegression, RepeatedStratifiedKFold, StratifiedKFold
- XGBoost - Multi-horizon binary classifiers, XGBoost Cox survival objective
- LightGBM - Multi-horizon binary classifiers
- CatBoost - Multi-horizon binary classifiers

**Data Processing:**
- pandas - DataFrame operations, CSV I/O
- numpy - Numerical arrays, mathematical operations

**Optimization:**
- scipy.optimize - minimize, minimize_scalar, brentq (for weight optimization, calibration)

## Key Dependencies

**Critical:**
- scikit-survival - Enables RSF/EST/GBSA survival models (core ensemble members)
- lifelines - Provides Cox/AFT models and concordance index metric
- XGBoost - Gradient boosting baseline for ensemble diversity
- LightGBM - Boosting alternative for ensemble
- CatBoost - Categorical boosting alternative for ensemble
- pandas - Data loading and manipulation
- numpy - Numerical computation backbone
- scipy - Optimization for weight search and calibration

**Infrastructure:**
- None detected - No cloud SDKs, databases, or external services

## Configuration

**Environment:**
- Paths configured in `src/config.py` (PROJECT_DIR, DATA_DIR, TRAIN_PATH, TEST_PATH, SUBMISSIONS_DIR)
- Feature sets defined in `src/config.py` (FEATURES_MINIMAL, FEATURES_MEDIUM, FEATURES_FULL, FEATURES_V96624_*)
- CV configuration: N_SPLITS=5, N_REPEATS=10, RANDOM_STATE=42
- Horizons: [12, 24, 48, 72] hours

**Build:**
- No build configuration detected (pure Python, no compilation)

## Platform Requirements

**Development:**
- Windows 11 (based on environment)
- Python 3.x with pip
- Virtual environment support

**Production:**
- Python 3.x runtime
- CSV input files (train.csv, test.csv)
- Local filesystem for data and submissions

---

*Stack analysis: 2026-02-22*
