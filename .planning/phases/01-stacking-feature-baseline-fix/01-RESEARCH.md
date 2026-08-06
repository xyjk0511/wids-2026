# Phase 1: Stacking & Feature Baseline Fix - Research

**Researched:** 2026-02-22
**Domain:** Stacking head replacement (XGB→LR) + feature set baseline
**Confidence:** HIGH (all findings from direct source code inspection)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Replace XGB heads with LogisticRegression (C=1.0, balanced class_weight, lbfgs)
- Remove post-head calibration layer — LR outputs probabilities directly
- Keep inner CV anti-leakage design (3-fold inner CV for base_train_oof)
- Start from V96624_PLUS feature set (21 features) as baseline
- Backward elimination using permutation importance ranking
- Evaluation metric: Hybrid Score (CI + WBrier combined)
- All horizons share the same feature subset (no per-horizon selection)
- Keep two-layer structure (base models → LR head), only replace head
- Base models stay as current 3: RSF, EST, XGBCox
- Head input: only base model predictions (use_orig_features=False)
- Base feature representation: raw probabilities (base_feature_mode="raw")
</user_constraints>

---

## Summary

The stacking pipeline in `src/stacking.py` already has full LR head support via
`head_model="logit"` (lines 274-281). Switching heads requires only changing the
`--head-model` CLI flag — no new code needed for the head itself.

The calibration layer is bypassed by `calibration_mode="none"` (stacking.py:138-139),
already implemented. `use_orig_features=False` maps to `--head-base-only` (train.py:585).

**Primary recommendation:** Target config is fully expressible via existing CLI flags:
`--head-model logit --head-base-only --calibration-mode none --feature-level v96624_plus`
The only new code needed is a backward elimination script.

---

## Component Map

### 1. Head Model — `src/stacking.py`

| What | Location | Current default | Target |
|------|----------|----------------|--------|
| Head type branch | lines 274-299 | `head_model="xgb"` | `head_model="logit"` |
| LR config | lines 275-281 | C=1.0, balanced, lbfgs | Already matches spec |
| XGB config | lines 285-299 | 200 trees, depth=3 | Unused after switch |
| Calibration call | lines 304-311 | `_fit_calibrator(...)` | `calibration_mode="none"` |
| `use_orig_features` | line 80 | True | False via `--head-base-only` |
| `base_feature_mode` | lines 89-99 | "raw" | Already "raw" |

The LR branch at line 274 uses `class_weight="balanced"` and does NOT use `spw`.
`spw` is computed at lines 271-272 but only consumed by the XGB branch — harmless.

### 2. Calibration — `src/stacking.py`

`_fit_calibrator` (lines 134-176): fits Platt + Isotonic, picks best.
With `calibration_mode="none"` returns `(None, "none")` immediately (line 138-139).
`_apply_calibrator(None, probs)` returns `probs` unchanged (lines 127-128).

`src/calibration.py` contains standalone utilities NOT called from the stacking path.
No changes needed there.

### 3. Feature Set — `src/config.py`

`FEATURES_V96624_PLUS` (lines 138-144) = 13 base + 3 engineered + 5 perm-importance = **21 features**.

```
FEATURES_V96624_BASE        lines 116-130   13 features
FEATURES_V96624_ENGINEERED  lines 131-135    3 features (has_growth, is_approaching, log_dist_min)
+ top-5 perm importance     lines 139-144    5 features (log_dist, eta_hours, dt_first_last_0_5h,
                                                          growth_dist_ratio, threat_static)
```

Engineered features require `add_engineered()`. `load_data(feature_level="v96624_plus")`
calls it at train.py:38-40. No risk if using the CLI flag.

### 4. Head Feature Matrix (with target config)

With `use_orig_features=False` + `base_feature_mode="raw"`, head input = **9 columns**:
`RSF_12h_raw`, `RSF_24h_raw`, `RSF_48h_raw`,
`EST_12h_raw`, `EST_24h_raw`, `EST_48h_raw`,
`XGBCox_12h_raw`, `XGBCox_24h_raw`, `XGBCox_48h_raw`

Column alignment in `predict_horizon_heads` uses `reindex(columns=meta["head_feature_cols"])`
(stacking.py:364) — consistent between train and predict paths.

### 5. Train.py Invocation — `src/train.py`

Target CLI:
```bash
python -m src.train \
  --feature-level v96624_plus \
  --head-model logit \
  --head-base-only \
  --calibration-mode none
```

Wiring at train.py lines 783-796:
```python
head_oof, heads = train_horizon_heads(
    head_model=args.head_model,                # "logit"
    base_feature_mode=args.base_feature_mode,  # "raw" (default)
    use_orig_features=not args.head_base_only, # False
    calibration_mode=args.calibration_mode,    # "none"
)
```

### 6. Inner CV Anti-Leakage — `src/stacking.py`

`_inner_cv_base_oof` (lines 103-123): 3-fold StratifiedKFold on `ye_tr`.
Produces leak-free base predictions for head training. Preserved unchanged.
`n_inner_splits=3` passed from train.py line 789.

---

## What Needs to Be Built

### Already Implemented (zero new code)
- LR head: `head_model="logit"` branch at stacking.py:274-281
- Calibration bypass: `calibration_mode="none"` at stacking.py:138-139
- `use_orig_features=False`: `--head-base-only` flag at train.py:585
- `base_feature_mode="raw"`: already default
- `FEATURES_V96624_PLUS`: defined at config.py:138-144
- `feature_level="v96624_plus"`: routing at train.py:38-40

### Needs New Code
1. **Backward elimination script** — not present anywhere. Needs to:
   - Start with all 21 V96624_PLUS features
   - Compute permutation importance on OOF predictions (shuffle each feature, measure Hybrid Score drop)
   - Iteratively drop lowest-importance feature
   - Evaluate Hybrid Score after each drop via run_cv()
   - Output ranked feature list + score curve

---

## Risks & Edge Cases

### Risk 1: LR probability compression
LR with `class_weight="balanced"` on imbalanced data may output probabilities
clustered near 0.5. With calibration removed, this goes directly to submission.
Monitor 12h distribution (target: min~0.036, median~0.15, max~0.99).

### Risk 2: Backward elimination CV noise
Permutation importance on OOF is noisy with small N. Use full 5×10 CV for
elimination evaluation, not a fast proxy. Drop only when improvement is consistent.

### Risk 3: Engineered feature availability
`FEATURES_V96624_PLUS` includes 8 engineered features. If `add_engineered()` is
not called (wrong `feature_level`), those columns will be missing and cause KeyError.
Always use `--feature-level v96624_plus`.

---

## Implementation Order

1. Verify current baseline CV with default config (record Hybrid Score)
2. Run target config: `--head-model logit --head-base-only --calibration-mode none --feature-level v96624_plus`
3. Compare CV — if stable or improved, proceed
4. Build and run backward elimination script
5. Final CV with LR head + optimal feature subset

---

## Sources

All findings from direct source inspection (HIGH confidence):
- `src/stacking.py` — head implementation, calibration, inner CV (lines cited above)
- `src/config.py` — feature set definitions (lines 116-144)
- `src/train.py` — CLI flags, pipeline orchestration (lines 559-796)
- `src/calibration.py` — standalone utilities, not in stacking path
- `src/evaluation.py` — hybrid_score at lines 101-109
