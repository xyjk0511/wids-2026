# Phase 2: Model Diversity Ensemble (Rewritten) - Research

**Researched:** 2026-02-22
**Domain:** IPCW-aware survival stacking + calibration for small-N censored data
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Decision 1: IPCW Stacking**
- Meta-learner: Compare Ridge(alpha=1.0) vs LR(C=0.01), select by OOF
- Base model pool: RSF + EST + GBSA (3 models, 12 meta-features = 3 x 4 horizons)
- CV protocol: Two-stage -- 5x1 quick check, 5x10 only if signal
- Go/no-go gate: OOF hybrid > 0.9697 AND Spearman vs 0.96624 stable

**Decision 2: Calibration Strategy**
- Scope: New methods (isotonic/Platt/piecewise linear)
- Target horizons: 24h and 48h only
- Anchor strategy: Two tracks -- anchor-incremental + independent calibration

### Deferred Ideas (OUT OF SCOPE)
- 12h ranking improvement (separate phase if needed)
- Phase 3 conformal calibration (CSD/CiPOT)
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| R1 | Model Diversity -- ensemble >= 4 model types; CI improves on OOF | IPCW stacking adds GBSA; RSF+EST+GBSA = 3 survival models |
| R6 | Seed Expansion -- 5+ seeds; LB improves or holds | Already done: RETRAIN_SEEDS=[42,123,456,789,2026] in train.py line 847 |
</phase_requirements>

---

## Summary

Phase 2 is a full pivot. Exp27/28/30 proved CoxPH/WeibullAFT/XGBoostCox do not improve LB. Two independent tracks: (1) IPCW-aware stacking, (2) new calibration methods for 24h/48h.

**IPCW stacking** replaces naive OOF meta-learning with KM-reweighted stacking. On 221 samples with ~40% censoring, naive OOF treats censored samples as observed -- IPCW corrects this bias. Meta-learner takes 12 features (RSF/EST/GBSA x 4 horizons). GBSA exists in src/models.py lines 270-298 but is NOT in BASE_NAMES in stacking.py.

**Calibration track** targets 24h/48h only. Logit (A,B) plateaued at lam=6.0 (PB=0.96783). Isotonic/Platt already in src/stacking.py _fit_calibrator() but only inside fold heads, not standalone.

**R6 is already complete.** RETRAIN_SEEDS=[42,123,456,789,2026] confirmed in src/train.py line 847.

**Primary recommendation:** Implement IPCW stacking as scripts/exp30_ipcw_stacking.py. Do not modify src/train.py. Instantiate GBSA directly (not via _make_base). Run 5x1 first; only proceed to 5x10 if OOF hybrid > 0.9697.

---

## Standard Stack

| Library | Purpose | Status |
|---------|---------|--------|
| sksurv (.venv_sksurv22) | RSF, EST, GBSA models | In production |
| lifelines | KM estimator for IPCW weights | In production (c_index) |
| sklearn | Ridge, LogisticRegression, IsotonicRegression | In production |
| numpy/scipy | Array ops, optimization | In production |

---

## Architecture Patterns

### Pattern 1: IPCW Weight Computation

```python
from lifelines import KaplanMeierFitter

def compute_ipcw_weights(y_time, y_event, clip_min=0.05):
    kmf = KaplanMeierFitter()
    kmf.fit(y_time, event_observed=(1 - y_event))  # flip: censoring=event
    G_t = kmf.survival_function_at_times(y_time).values
    G_t = np.clip(G_t, clip_min, 1.0)
    return np.where(y_event == 1, 1.0 / G_t, 1.0)
```

### Pattern 2: Horizon-Specific IPCW Labels

```python
# event=1, time<=h (label=1): weight = 1/G(t_i)
# event=0, time>=h (label=0): weight = 1.0 (observed to survive)
# event=0, time<h:  excluded by eligible mask
labels, eligible = build_horizon_labels(y_time, y_event, h)
ipcw_all = compute_ipcw_weights(y_time, y_event)
w_h = np.where((y_event == 1) & (y_time <= h), ipcw_all, 1.0)
```

### Pattern 3: Two-Stage CV Protocol

```python
# Stage 1: quick check (5x1)
oof_q = run_cv(train, feature_cols, n_splits=5, n_repeats=1, ...)
score_q, _ = hybrid_score(y_time, y_event, oof_q)
if score_q <= 0.9697:
    print("No signal -- stopping"); return
# Stage 2: full CV (5x10)
oof_full = run_cv(train, feature_cols, n_splits=5, n_repeats=10, ...)
```

### Pattern 4: Go/No-Go Gate Check

```python
from scipy.stats import spearmanr
ref = pd.read_csv("submissions/submission_0.96624.csv")
score, det = hybrid_score(y_time, y_event, oof_preds)
spearman_ok = all(
    spearmanr(oof_preds[h], ref[f"prob_{h}h"])[0] > 0.90
    for h in [12, 24, 48]
)
gate_pass = score > 0.9697 and spearman_ok
```

### Anti-Patterns to Avoid

- **Modifying train.py for experiments:** All Phase 2 work goes in scripts/exp*.py.
- **Fitting calibration on training predictions:** Always use OOF predictions.
- **Skipping eligible mask:** build_horizon_labels() returns eligible mask -- always apply it.
- **Adding CoxPH/WeibullAFT/XGBoostCox back:** Exp27/28/30 invalidated these.

---

## Don't Hand-Roll

| Problem | Use Instead |
|---------|-------------|
| KM censoring estimator | lifelines.KaplanMeierFitter (already installed) |
| Isotonic regression | sklearn.isotonic.IsotonicRegression (already in stacking.py) |
| Ridge with sample weights | sklearn.linear_model.Ridge + sample_weight arg |
| GBSA model | src.models.GBSA (lines 270-298 of models.py) |

---

## Common Pitfalls

### Pitfall 1: GBSA Not Wired Into _make_base()
**What:** stacking.py _make_base() has no "GBSA" case -- raises ValueError.
**Fix:** Instantiate GBSA() directly in the experiment script, not via _make_base().

### Pitfall 2: Ridge Outputs Outside [0,1]
**What:** Ridge is regression -- outputs can be negative or >1.
**Fix:** Clip Ridge output to [0,1], or use LogisticRegression(C=0.01). Test both per CONTEXT.md.

### Pitfall 3: GBSA Compute Cost
**What:** GBSA is ~3-5x slower than RSF (sequential boosting, no intra-model parallelism).
**Fix:** Mandatory two-stage protocol. 5x1 first. Only run 5x10 if signal confirmed.

### Pitfall 4: Isotonic Overfitting on ~130 Eligible Samples
**What:** Isotonic regression memorizes training data on small N.
**Fix:** Fit on OOF predictions only. The existing _fit_calibrator() in stacking.py already handles this with 80/20 train/cal split inside each fold.

### Pitfall 5: 72h Meta-Feature Is Constant
**What:** After postprocess, 72h predictions are always 1.0. Near-constant column may cause LR convergence warnings.
**Fix:** Include 72h per spec (Ridge assigns near-zero weight automatically). For LR, add max_iter=2000.

---

## Key Facts for Planning

1. **R6 is done.** No seed expansion work needed. RETRAIN_SEEDS=[42,123,456,789,2026] in train.py line 847.
2. **GBSA exists in models.py but not in stacking.py BASE_NAMES.** Instantiate directly in experiment script.
3. **Calibration infrastructure already exists.** _fit_calibrator() in stacking.py handles isotonic/Platt.
4. **Current best is anchor-based.** 0.96624 submission is the anchor. All calibration experiments compare Spearman vs this anchor.
5. **221 training samples, ~40% censoring.** Ridge(alpha=1.0) and LR(C=0.01) are both heavily regularized -- appropriate for this N.
6. **Experiment scripts pattern.** All recent experiments (exp22-exp29) are standalone scripts in scripts/ importing from src/. Do not modify src/train.py.

---

## Open Questions

1. **IPCW clip threshold:** Start with 0.05. Report weight distribution (min/max/p5/p95) in experiment output.
2. **Independent calibration OOF source:** Use RSF+EST 50/50 blend OOF to isolate calibration effect from model changes.
3. **72h meta-features:** Include per spec; Ridge handles near-constant columns automatically.

---

## Sources

### Primary (HIGH confidence -- codebase verified)
- D:/wids/src/models.py -- GBSA at lines 270-298; all model classes confirmed
- D:/wids/src/stacking.py -- _fit_calibrator() isotonic/Platt at lines 134-176; BASE_NAMES at line 19
- D:/wids/src/train.py -- RETRAIN_SEEDS=[42,123,456,789,2026] at line 847; R6 confirmed done
- D:/wids/src/evaluation.py -- hybrid_score formula: 0.3*CI + 0.7*(1-WBrier)
- D:/wids/.planning/phases/02-model-diversity-ensemble/02-CONTEXT.md -- locked decisions

### Secondary (MEDIUM confidence)
- IPCW survival stacking: Standard technique; lifelines.KaplanMeierFitter confirmed available

### Tertiary (LOW confidence)
- GBSA compute time (3-5x RSF): Not benchmarked; estimate from general knowledge

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries confirmed installed and used in existing code
- Architecture: HIGH -- codebase fully read, patterns verified
- Pitfalls: HIGH for code-level (verified in source); MEDIUM for IPCW correctness details
- GBSA compute cost: LOW -- not benchmarked

**Research date:** 2026-02-22
**Valid until:** 2026-03-08
