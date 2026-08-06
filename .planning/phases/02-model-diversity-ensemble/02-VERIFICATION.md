---
phase: 02-model-diversity-ensemble
verified: 2026-02-22T09:02:00Z
status: human_needed
score: 5/5 must-haves verified
human_verification:
  - test: Submit submissions/submission_exp32_cal.csv to Kaggle
    expected: LB > 0.96783 (current PB); OOF showed +0.0059 hybrid improvement
    why_human: Calibration gains on OOF (N=221) do not always transfer to LB; only Kaggle scoring confirms real improvement
---

# Phase 2: Model Diversity Ensemble Verification Report

**Phase Goal:** Improve LB through IPCW-aware stacking and new calibration methods
**Verified:** 2026-02-22T09:02:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | IPCW stacking with RSF+EST+GBSA produces OOF hybrid score | VERIFIED | exp31 runs end-to-end; Stage 1 OOF hybrid=0.96610 printed |
| 2 | Ridge vs LR meta-learner compared on OOF | VERIFIED | Both fitted per horizon; scores printed for Ridge-S1 and LR-S1 |
| 3 | Go/no-go gate evaluated: OOF hybrid > 0.9697 AND Spearman > 0.90 | VERIFIED | Gate logic at line 249; gate failed (0.96610 < 0.9697), no submission — correct |
| 4 | Two-stage CV: 5x1 quick check, 5x10 only if signal | VERIFIED | Stage 1 n_repeats=1 at line 241, Stage 2 n_repeats=10 at line 255, guarded by gate |
| 5 | Isotonic, Platt, piecewise-linear compared on 24h and 48h | VERIFIED | All 3 methods in METHODS dict; loop over [24,48] at line 216 |
| 6 | Two tracks evaluated: anchor-incremental and independent | VERIFIED | Track B at line 222, Track A (alpha=0.1/0.2/0.3) at line 238 |
| 7 | Best calibration method selected by OOF hybrid without CI degradation | VERIFIED | CI constraint ci >= ci_base - 0.001 at line 260; Platt/B/48h selected |

**Score:** 5/5 plan must-haves verified (7/7 truths verified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| scripts/exp31_ipcw_stacking.py | IPCW stacking with GBSA + gate check | VERIFIED | 279 lines, substantive |
| scripts/exp32_calibration_methods.py | Calibration method comparison | VERIFIED | 326 lines, substantive |
| submissions/submission_exp32_cal.csv | Best calibration submission | VERIFIED | 96 rows (95 test + header), correct columns |
| submissions/submission_exp31_ipcw.csv | IPCW submission if gate passes | VERIFIED ABSENT | Gate failed (0.96610 < 0.9697) — correct, no file expected |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| exp31_ipcw_stacking.py | src/models.py | from src.models import RSF, EST, GBSA | WIRED | Line 24; all 3 instantiated in make_base_models() |
| exp31_ipcw_stacking.py | src/evaluation.py | from src.evaluation import hybrid_score | WIRED | Line 25; called in eval_gate() |
| exp32_calibration_methods.py | src/evaluation.py | from src.evaluation import hybrid_score, horizon_brier_score, c_index | WIRED | Line 26; hybrid_score called throughout |
| exp32_calibration_methods.py | submissions/submission_0.96624.csv | ANCHOR_PATH = submissions/submission_0.96624.csv | WIRED | Line 33; loaded at line 274 for test prediction base |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| R1 | 02-01, 02-02 | Ensemble >= 3 model types; CI improves on OOF | SATISFIED | exp31 uses RSF+EST+GBSA; exp32 OOF CI=0.9320 stable vs baseline |
| R6 | 02-01 (acknowledged) | 5+ seeds; LB improves or holds | SATISFIED (pre-completed) | PLAN acknowledges R6 complete; confirmed in exp32 SEEDS=[42,123,456,789,2026] |

No orphaned requirements — REQUIREMENTS.md maps R1 and R6 to Phase 2; both accounted for.

### Anti-Patterns Found

None. No TODO/FIXME/placeholder comments, no stub returns in either script.

### Human Verification Required

#### 1. LB Validation of exp32 Calibration

**Test:** Submit submissions/submission_exp32_cal.csv to Kaggle
**Expected:** LB > 0.96783 (current PB); OOF showed +0.0059 hybrid improvement (Platt/B/48h)
**Why human:** Calibration gains on OOF (N=221) do not always transfer to LB test set; only Kaggle scoring confirms real improvement

### Gaps Summary

No gaps. All automated checks pass:
- Both scripts exist and are substantive (no stubs, no placeholders)
- All key imports wired and used
- Gate logic correct: exp31 stopped at Stage 1, no spurious submission generated
- exp32 submission has correct shape (95 rows, 5 columns matching reference)
- Both commits (b6e29a0, fb3fac4) verified in git log
- R1 satisfied: 3 model types in exp31; calibration improves WBrier in exp32
- R6 pre-completion acknowledged and confirmed (5 seeds in exp32)

Phase goal structurally achieved: IPCW stacking tested and correctly closed (no signal), Platt/B/48h calibration shows +0.0059 OOF hybrid with submission ready for LB validation.

---

_Verified: 2026-02-22T09:02:00Z_
_Verifier: Claude (gsd-verifier)_
