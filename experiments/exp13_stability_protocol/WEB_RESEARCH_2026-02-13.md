# Web Research Notes (2026-02-13)

## Scope
- Goal: collect **official competition info**, **public code**, and **transferable ideas** for the WiDS 2026 wildfire survival task.
- Focused on primary sources: Kaggle competition pages, official starter notebook, publicly posted competition notebooks, and survival library documentation.

## 1) Official Competition Facts (confirmed)

### 1.1 Task framing
- Predict wildfire threat timing as survival probabilities at multiple horizons.
- Public descriptions from WiDS and Watch Duty emphasize uncertainty, sparse observations, and right-censoring.

### 1.2 Metric (official evaluation page)
- Hybrid score:
  - `0.3 * C-index_12h + 0.7 * (1 - WeightedBrier)`
- Weighted Brier:
  - `0.3 * Brier@24h + 0.4 * Brier@48h + 0.3 * Brier@72h`
- This confirms practical optimization priority:
  - Brier-heavy objective (70%)
  - 48h has the highest single weight in Brier aggregation.

### 1.3 Competition timeline/status
- Timeline text on competition pages indicates final deadlines around **February 11, 2026**.
- Treat this as post-deadline research/replication context.

## 2) Public Code Landscape (Kaggle Code page)

### 2.1 Public notebook score band (visible entries)
- Top visible public scores are in roughly `0.959` to `0.965`.
- One visible notebook reaches about `0.96465` (public board).

### 2.2 Methods repeatedly appearing in stronger notebooks
- RSF-centric modeling remains dominant.
- Horizon-aware optimization is common (optimize different horizons differently).
- Repeated CV / seed averaging is widely used for stability.
- Post-processing constraints are common:
  - monotonicity across 12h/24h/48h/72h
  - clipping / floor handling for extreme low-risk cases
- Parametric + non-parametric blending appears frequently (e.g., AFT + RSF).

## 3) Most Relevant Public Notebooks (readable content)

### 3.1 `widsworldwide/starter-notebook-time-to-threat-baseline-models`
- Recommends survival-native models (Cox, RSF, GBSA, AFT).
- Emphasizes:
  - horizon-based target conversion,
  - strong CV design,
  - avoiding leakage,
  - ranking/calibration balance,
  - careful treatment of high censoring and degenerate horizons.
- Practical suggestion in starter notes: when 72h target is effectively degenerate, avoid overfitting that column and use stable handling.

### 3.2 `uygararslan/wids-2026-hybrid-rsf-est-ensemble`
- Shows RSF-only OOF around `0.9721` and an RSF+EST equal blend degrading OOF (`~0.9701`) in that setup.
- Supports the interpretation that naive EST mixing can hurt local CV even if variance reduction may help leaderboard generalization.

### 3.3 `uygararslan/wids-2026-horizon-aware-rsf-v2`
- Reports per-horizon best 2-model blend with **LogNormalAFT + RSF**:
  - 12h: `RSF 0.8 + LogNormalAFT 0.2`
  - 24h: `RSF 0.9 + LogNormalAFT 0.1`
  - 48h: `RSF 0.8 + LogNormalAFT 0.2`
- Reported OOF hybrid improvement vs RSF-only: about `+0.0004`.
- Also reports an RSF+EST observation:
  - OOF can be worse than RSF-only, but LB can still improve (interpreted as variance/generalization benefit).

## 4) Technical References Behind Model Choices

### 4.1 Censoring-aware metrics
- `scikit-survival` docs confirm Brier for right-censored data uses IPCW and requires train/test time-range consistency.
- Evaluation docs also explain Harrell C-index optimism under heavy censoring and alternatives like IPCW concordance.

### 4.2 Hazard-shape flexibility
- Weibull hazard (formula form) is shape-constrained by parameterization.
- Lognormal hazard formula (reliability/NIST references) supports more flexible shape behavior than simple monotone assumptions.
- This is consistent with why small LogNormalAFT mixing can improve calibration on certain horizons.

## 5) What This Means For Our Current Pipeline

### 5.1 High-confidence conclusions
- Per-horizon blending is not optional detail; it is structurally aligned with official metric weighting.
- RSF should remain the ranking backbone.
- AFT (especially lognormal-family) is most useful as calibration/complement, not replacement.
- EST should be treated as a **variance-control component** for final retrain/ensemble robustness, not an automatic OOF winner.

### 5.2 Priority execution order (practical)
1. Lock CV protocol and report mean/std across seeds before any model change.
2. Keep RSF as base; tune only lightweight horizon-specific blend weights with LogNormalAFT.
3. Use EST only as a bounded component in final full-retrain ensemble (not forced in OOF optimizer).
4. Keep monotonic post-processing and conservative clipping/floor policies.
5. Explicitly isolate 72h strategy (degenerate handling) from 12/24/48 optimization.

## Sources
- Kaggle competition overview: https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26
- Kaggle evaluation page (metric formula): https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/overview/evaluation
- Kaggle data page: https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/overview/data
- Kaggle code listing: https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26/code
- Official starter notebook: https://www.kaggle.com/code/widsworldwide/starter-notebook-time-to-threat-baseline-models
- Public notebook (hybrid ensemble): https://www.kaggle.com/code/uygararslan/wids-2026-hybrid-rsf-est-ensemble
- Public notebook (horizon-aware RSF v2): https://www.kaggle.com/code/uygararslan/wids-2026-horizon-aware-rsf-v2
- WiDS challenge blog: https://www.widsworldwide.org/get-inspired/blog/from-the-innovation-lab-to-global-impact-introducing-the-wids-datathon-2026-challenge/
- Watch Duty challenge context: https://www.watchduty.org/blog/inside-watch-duty-s-innovation-lab-how-we-designed-the-2026-wids-datathon-challenge
- scikit-survival metric docs: https://scikit-survival.readthedocs.io/en/latest/generated/sksurv.metrics.brier_score.html
- scikit-survival evaluation guide: https://scikit-survival.readthedocs.io/en/latest/user_guide/evaluating-survival-models.html
- reliability distribution equations: https://reliability.readthedocs.io/en/latest/Equations%20of%20supported%20distributions.html
- NIST lognormal hazard reference: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/lgnhaz.htm
