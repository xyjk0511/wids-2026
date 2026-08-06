# Testing Patterns

**Analysis Date:** 2026-02-22

## Test Framework

**Runner:**
- Not detected - no pytest, unittest, or vitest configuration found
- No `pytest.ini`, `setup.cfg`, or `pyproject.toml` with test config

**Assertion Library:**
- Not applicable - no test framework detected

**Run Commands:**
- Manual validation: `python -m src.train` (runs full CV pipeline)
- No automated test suite

## Test File Organization

**Location:**
- Not detected - no `tests/` directory or `*_test.py` files found
- Testing appears to be manual/exploratory via scripts in `scripts/` directory

**Naming:**
- Not applicable - no test files present

**Structure:**
- Experimental scripts in `scripts/` directory: `exp16_step_a.py`, `exp17_ablation.py`, etc.
- These serve as ad-hoc validation rather than formal tests

## Test Structure

**Suite Organization:**
- Not applicable - no formal test suite

**Patterns:**
- Manual validation in `src/train.py`: `_validate_and_save()` function performs checks
- Diagnostic output: validation checks for 72h all-ones, monotonicity, probability distribution
- Example from `src/train.py` lines 512-519:
```python
h72_ok = (sub[PROB_COLS[HORIZONS.index(72)]] == 1.0).all()
mono_ok = all(
    all(sub[PROB_COLS[j]].iloc[i] >= sub[PROB_COLS[j-1]].iloc[i] - 1e-9
        for j in range(1, len(PROB_COLS)))
    for i in range(len(sub))
)
print(f"  72h all-ones: {'PASS' if h72_ok else 'FAIL'}  Monotonicity: {'PASS' if mono_ok else 'FAIL'}")
```

## Mocking

**Framework:**
- Not detected - no mocking library used

**Patterns:**
- Not applicable - no formal tests

**What to Mock:**
- Not applicable

**What NOT to Mock:**
- Not applicable

## Fixtures and Factories

**Test Data:**
- Not applicable - no test fixtures

**Location:**
- Not applicable

## Coverage

**Requirements:**
- Not enforced - no coverage tool configured

**View Coverage:**
- Not applicable

## Test Types

**Unit Tests:**
- Not detected - no unit tests present

**Integration Tests:**
- Manual integration via `src/train.py` main pipeline
- Full CV pipeline with cross-validation: `run_cv()` function
- Decoupled per-horizon strategy: `run_cv_decoupled()` function

**E2E Tests:**
- Not detected - no E2E test framework

## Common Patterns

**Validation Checks:**
- Probability bounds: `np.clip(p, 0.0, 1.0)`
- Monotonicity enforcement: `enforce_monotonicity()` in `src/monotonic.py`
- Distribution diagnostics: quantile analysis, near-zero counts
- Spearman correlation vs reference: `spearmanr(sub[col], ref[col])`

**Example from `src/train.py` lines 521-538:**
```python
print("\n=== Probability Distribution (vs 0.96624 target) ===")
for col in PROB_COLS:
    vals = sub[col]
    print(f"  {col}: min={vals.min():.4f} median={vals.median():.4f} max={vals.max():.4f}")

print("\n=== Floor Diagnostics ===")
for h in [12, 24, 48]:
    col = f"prob_{h}h"
    vals = sub[col]
    n_exact = np.isclose(vals, floor_for_diag, atol=1e-12).sum()
    n_near_zero = (vals <= 1e-5).sum()
    print(f"  {col}: near_floor({floor_for_diag:g})={n_exact}/{len(vals)}")
```

**Async Testing:**
- Not applicable - no async code

**Error Testing:**
- Explicit error raising: `raise ValueError(f"Unknown strat mode: {mode}")`
- Type validation: `if v.ndim != 1: raise ValueError(...)`

## Validation Strategy

**Manual CV Validation:**
- Run `python -m src.train` to execute full cross-validation
- Outputs per-model OOF scores, weight search results, and diagnostics
- Compares multiple strategies (baseline, per-horizon, decoupled)

**Submission Validation:**
- `_validate_and_save()` checks:
  - 72h horizon all 1.0
  - Monotonicity across horizons
  - Probability distribution statistics
  - Floor diagnostics (near-zero counts)
  - Spearman correlation vs reference submission

**Diagnostic Output:**
- Hybrid score, C-index, weighted Brier score per model
- Per-horizon Brier scores
- Probability quantiles (min, p25, median, p75, max)
- Near-zero probability counts

---

*Testing analysis: 2026-02-22*
