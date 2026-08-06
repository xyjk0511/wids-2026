---
phase: 04-stronger-anchor
verified: 2026-02-23T06:24:26Z
status: gaps_found
score: 3/7 must-haves verified
re_verification: false
gaps:
  - truth: "User has identified 0.966+ public notebooks from Kaggle"
    status: failed
    reason: "suman2208 LB=0.96086 below gate; rhythmghai private (403); no accessible notebook above 0.96624"
    artifacts:
      - path: "submissions/anchor_suman2208/run_metadata.txt"
        issue: "lb_score=0.96086 below 0.96624 gate; stop-loss triggered"
    missing:
      - "A public notebook with LB >= 0.966 that is accessible and reproducible"
  - truth: "Rank-average blend of eligible anchors produces a submission"
    status: failed
    reason: "Blend gate never triggered; no anchor passed Gate 1 (LB > 0.96624)"
    artifacts:
      - path: "scripts/exp30_blend_anchors.py"
        issue: "Script correct but never exercised with a qualifying anchor"
    missing:
      - "A qualifying anchor (LB > 0.96624) to trigger the blend pipeline"
  - truth: "Hyperparam grid tests n_estimators, max_features, min_samples_leaf"
    status: partial
    reason: "Grid ran 4 configs but pipeline inconsistency caused prob_48h std=0.102 vs ref 0.364; LB=0.91089/0.90860 invalid"
    artifacts:
      - path: "scripts/exp30_hyperparam_grid.py"
        issue: "Feature engineering or postprocessing diverged from reference pipeline"
      - path: "submissions/submission_exp30_grid_r1.csv"
        issue: "LB=0.91089 catastrophic failure"
      - path: "submissions/submission_exp30_grid_r2.csv"
        issue: "LB=0.90860 catastrophic failure"
    missing:
      - "Pipeline parity audit: exp30_hyperparam_grid.py must match exp17_reproduce_ref.py exactly"
      - "Re-run grid after parity fix"
  - truth: "Phase goal achieved: new anchor with LB > 0.96624"
    status: failed
    reason: "Both tracks failed. Track 1 stop-loss at LB=0.96086. Track 3 grid invalid. Best remains 0.96624."
    artifacts: []
    missing:
      - "Any submission with LB > 0.96624"
---

# Phase 4: Stronger Anchor Verification Report

**Phase Goal:** 获取或复现比 0.96624 更强的基础预测
**Verified:** 2026-02-23T06:24:26Z
**Status:** gaps_found
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User identified 0.966+ public notebooks | FAILED | suman2208 LB=0.96086 below gate; rhythmghai private (403) |
| 2 | Reproduction script ingests config and produces submission | VERIFIED | exp30_reproduce_anchor.py 161L, JSON config, engineer_features + postprocess wired |
| 3 | Reproduced submission compared vs ref 0.96624 with Spearman | VERIFIED | exp30_compare_anchors.py 40L, spearmanr per horizon, ELIGIBLE/SKIP verdict |
| 4 | p48 Spearman measured for Track 2 eligibility | VERIFIED | compare_anchors.py flags p48; blend_anchors.py gates on rho48 in [0.90, 0.99] |
| 5 | Blend of eligible anchors produces submission | FAILED | Gate never triggered - no anchor passed LB > 0.96624 |
| 6 | Hyperparam grid tests n_estimators, max_features, min_samples_leaf | PARTIAL | Grid ran but pipeline inconsistency invalidated results (std=0.102 vs 0.364) |
| 7 | Phase goal: new anchor LB > 0.96624 | FAILED | Best remains 0.96624; all attempts below gate or catastrophically failed |

**Score:** 3/7 truths verified

### Required Artifacts

| Artifact | Min Lines | Actual | Status | Details |
|----------|-----------|--------|--------|---------|
| scripts/exp30_reproduce_anchor.py | 80 | 161 | VERIFIED | engineer_features, submission_postprocess, RSF+GBSA, JSON config |
| scripts/exp30_compare_anchors.py | 40 | 40 | VERIFIED | spearmanr per horizon, ELIGIBLE/SKIP verdict |
| scripts/exp30_blend_anchors.py | 50 | 124 | VERIFIED (orphaned) | 3-gate admission + OOF weight search - correct but never triggered |
| scripts/exp30_hyperparam_grid.py | 60 | 158 | INVALID-RESULT | Script runs; outputs invalid due to pipeline inconsistency |
| submissions/submission_exp30_grid_r1.csv | - | exists | INVALID | LB=0.91089 - pipeline inconsistency |
| submissions/submission_exp30_grid_r2.csv | - | exists | INVALID | LB=0.90860 - pipeline inconsistency |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| exp30_reproduce_anchor.py | exp17 patterns | engineer_features, submission_postprocess | WIRED | Defined at L50/L69, called at L112/L113/L143 |
| exp30_compare_anchors.py | submissions/ | pd.read_csv | WIRED | L10-11: reads both CSVs, sorts by event_id |
| exp30_blend_anchors.py | exp30_compare_anchors.py | spearmanr | WIRED | L19/L84/L117: imported and used for gate + ranking |
| exp30_hyperparam_grid.py | RSF/GBSA pipeline | RandomSurvivalForest, GradientBoostingSurvivalAnalysis | WIRED | L17/L83/L85: imported and instantiated |
| exp30_hyperparam_grid.py | exp30_reproduce_anchor.py parity | feature engineering match | NOT_WIRED | Pipeline diverged - std=0.102 vs ref 0.364 |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ANCHOR-01 | 04-01-PLAN.md | Reproduction tooling + anchor comparison | SATISFIED | exp30_reproduce_anchor.py (161L) + exp30_compare_anchors.py (40L) functional |
| ANCHOR-02 | 04-02-PLAN.md | Blend: 3-gate admission + OOF weight search | SATISFIED (infra only) | exp30_blend_anchors.py correct; gate never triggered |
| ANCHOR-03 | 04-02-PLAN.md | Hyperparam grid: RSF configs tested | PARTIAL | Grid ran; results invalid due to pipeline inconsistency |

Note: ANCHOR-01/02/03 are phase-local IDs in ROADMAP.md and PLANs only. .planning/REQUIREMENTS.md uses R1-R6 for the broader project. No orphaned requirements.

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| scripts/exp30_hyperparam_grid.py | Pipeline diverges from reference (std=0.102 vs 0.364) | BLOCKER | All grid results invalid; LB=0.91 confirms |
| scripts/exp30_blend_anchors.py | Never exercised with qualifying data | WARNING | Infrastructure untested end-to-end |

### Human Verification Required

#### 1. Pipeline Parity Audit

**Test:** Diff exp30_hyperparam_grid.py feature engineering and postprocessing against exp17_reproduce_ref.py
**Expected:** Identical feature list, scaler usage, and postprocess call
**Why human:** Root cause of std=0.102 compression not identified in code review

#### 2. Blend Gate End-to-End

**Test:** Run exp30_blend_anchors.py with two submissions where one has LB > 0.96624
**Expected:** Gates evaluated, weight search runs, blend CSVs produced
**Why human:** No qualifying anchor exists yet; cannot verify end-to-end without one

### Gaps Summary

Phase 4 failed its primary goal. Infrastructure was correctly built (ANCHOR-01, ANCHOR-02 satisfied). However:

1. Track 1 (external anchor): Only suman2208 (LB=0.96086) was accessible; rhythmghai/ridge-stacker was private. Stop-loss triggered correctly per plan.

2. Track 3 (hyperparam grid): exp30_hyperparam_grid.py produced catastrophically compressed predictions (prob_48h std=0.102 vs reference 0.364). Pipeline inconsistency - feature engineering or postprocessing diverged from reference. LB=0.91089/0.90860 confirms invalid outputs. ANCHOR-03 only partially satisfied.

3. Phase goal unmet: Current best remains 0.96624 public LB. No new anchor above this threshold was obtained or reproduced.

The blend infrastructure is reusable if a qualifying anchor appears in a future phase, but requires a qualifying input to be validated end-to-end.

---

_Verified: 2026-02-23T06:24:26Z_
_Verifier: Kiro (gsd-verifier)_
