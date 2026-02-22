# Coding Conventions

**Analysis Date:** 2026-02-22

## Naming Patterns

**Files:**
- Lowercase with underscores: `train.py`, `models.py`, `calibration.py`
- Descriptive module names reflecting functionality: `surv_post.py`, `monotonic.py`

**Functions:**
- Lowercase with underscores: `load_data()`, `run_cv()`, `enforce_monotonicity()`
- Private functions prefixed with single underscore: `_merge_rare_labels()`, `_strat_labels()`, `_pava_1d()`
- Descriptive names indicating purpose: `horizon_brier_score()`, `build_horizon_labels()`

**Variables:**
- Lowercase with underscores: `y_time`, `y_event`, `feature_cols`, `best_leaf`
- Abbreviated for common patterns: `X` (features), `y` (targets), `h` (horizon), `w` (weight)
- Prefixed for related groups: `oof_*` (out-of-fold), `test_*` (test predictions), `best_*` (optimal values)

**Types:**
- Type hints used throughout: `np.ndarray`, `dict[int, np.ndarray]`, `list[str]`
- Union types for optional parameters: `list[str] | None`
- Return type annotations on all functions

**Constants:**
- Uppercase with underscores: `HORIZONS`, `RANDOM_STATE`, `N_SPLITS`, `FEATURES_MEDIUM`
- Grouped logically in `src/config.py`

## Code Style

**Formatting:**
- No explicit formatter configured
- Follows PEP 8 conventions implicitly
- Line length ~88 characters
- Consistent 4-space indentation

**Linting:**
- No `.eslintrc` or `pylintrc` found
- Code follows implicit PEP 8 style

## Import Organization

**Order:**
1. Standard library: `argparse`, `warnings`, `collections`
2. Third-party scientific: `numpy`, `pandas`, `scipy`, `sklearn`, `lifelines`
3. Local project imports: `from src.config import ...`

**Path Aliases:**
- Relative imports from `src/` package: `from src.config import HORIZONS`
- No path aliases configured

## Error Handling

**Patterns:**
- Explicit validation with `ValueError`: `raise ValueError(f"Unknown strat mode: {mode}")`
- Type coercion with `np.asarray()`: `y_time = np.asarray(y_time, dtype=float)`
- Clipping to valid ranges: `np.clip(p, 0.0, 1.0)`, `np.clip(w, 1e-12, None)`
- Try-except for optional operations: `try: ref = pd.read_csv(ref_path) except FileNotFoundError: ...`

## Logging

**Framework:** `print()` statements (no logging library)

**Patterns:**
- Progress indicators: `print(f"  Repeat {rep}/{n_repeats} done")`
- Section headers: `print("\n=== OOF Scores (per-model) ===")`
- Diagnostic output with formatting: `print(f"  {col}: min={vals.min():.4f} median={vals.median():.4f}")`
- Warnings: `print(f"  [WARN] Reference '{ref_path}' not found")`
- 2-space indent for nested output

## Comments

**When to Comment:**
- Non-obvious algorithm logic: `# bins: <=12, (12,24], (24,48], >48`
- Complex transformations: `# per-sample 1D interpolation`
- Rationale for design choices: `# standardize features for CoxPH stability`

**Docstrings:**
- Module-level on all files: `"""Main training script: RSF single model + postprocessing."""`
- Function docstrings with Args/Returns
- Class docstrings: `"""Common interface for all survival models."""`

## Function Design

**Size:**
- Typical range: 20-60 lines
- Larger functions (100+ lines) for complex workflows: `run_cv()`, `run_cv_decoupled()`
- Helper functions extracted for clarity: `_merge_rare_labels()`, `_strat_labels()`

**Parameters:**
- Explicit keyword arguments for configuration: `min_samples_leaf=5`
- Default values from module constants: `n_splits=N_SPLITS`
- Type hints on all parameters

**Return Values:**
- Single return for simple operations: `return best_leaf`
- Dictionary for multi-horizon results: `return {h: prob_array for h in HORIZONS}`
- Tuple for related outputs: `return labels, eligible`
- Explicit type conversion: `return float(ci)`

## Module Design

**Exports:**
- No explicit `__all__` declarations
- Public functions used directly: `from src.models import RSF, EST`
- Private functions prefixed with `_`

**Barrel Files:**
- No barrel files (no `__init__.py` re-exports)
- Direct imports from specific modules

---

*Convention analysis: 2026-02-22*
