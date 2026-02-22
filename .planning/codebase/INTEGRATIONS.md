# External Integrations

**Analysis Date:** 2026-02-22

## APIs & External Services

**None detected** - This is a standalone machine learning competition project with no external API integrations.

## Data Storage

**Databases:**
- None - Project uses local CSV files only

**File Storage:**
- Local filesystem only
  - Input: `train.csv`, `test.csv`, `sample_submission.csv` in project root
  - Output: `submissions/submission.csv`
  - Metadata: `metaData.csv`

**Caching:**
- None - No caching layer detected

## Authentication & Identity

**Auth Provider:**
- Not applicable - No external services requiring authentication

## Monitoring & Observability

**Error Tracking:**
- None - No error tracking service integrated

**Logs:**
- Console output via print statements
- Log files in `logs/` directory (kaggle_comp.txt, kaggle_home.txt, etc.) for manual tracking

## CI/CD & Deployment

**Hosting:**
- Kaggle competition platform (external, not integrated via API)
- Manual submission workflow

**CI Pipeline:**
- None - No automated CI/CD detected

## Environment Configuration

**Required env vars:**
- None detected - All configuration via Python files and command-line arguments

**Secrets location:**
- Not applicable - No secrets or credentials in codebase

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Data Flow

**Training Pipeline:**
1. Load CSV data (`train.csv`, `test.csv`) via pandas
2. Feature engineering in `src/features.py`
3. Train multiple survival models (RSF, EST, LGBM, CatBoost, XGBoost, Cox, AFT)
4. Cross-validation with RepeatedStratifiedKFold
5. Ensemble weight optimization via scipy.optimize
6. Calibration and monotonicity enforcement
7. Generate submission CSV
8. Manual upload to Kaggle

**Evaluation:**
- Hybrid score (C-index + weighted Brier score) computed locally
- No external evaluation service

---

*Integration audit: 2026-02-22*
