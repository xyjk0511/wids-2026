# WiDS 2026 Kaggle — Push to 0.975+

## Vision
Systematically improve competition hybrid score from PB=0.96783 to 0.975+ through dual-track optimization: model diversity for CI and distribution calibration for WBrier.

## Core Value
Every change must be validated on LB (not just CV) due to 221-sample regime. Preserve anchor framework; incremental improvements over revolutionary rewrites.

## Current State
- **PB**: 0.96783 (Exp23 rh=1.1 rl=0.7 gate=[0.012,0.018])
- **Target**: 0.975+
- **Gap**: ~0.007 hybrid score
- **Metric**: hybrid = 0.3×CI(12h) + 0.7×(1-WBrier), WBrier = 0.3×B@24h + 0.4×B@48h + 0.3×B@72h
- **72h**: Fixed to 1.0 (not optimizable)
- **Submissions remaining**: >20

## Score Decomposition (to reach 0.975)
- CI needs: ~0.955 (current ~0.941, gap +0.014)
- WBrier needs: ~0.012 (current ~0.014, gap -0.002)
- WBrier leverage is 2.3× CI's: -0.01 WBrier → +0.007 hybrid
- Must improve BOTH dimensions; single-dimension insufficient

## Key Constraints
- 221 train / 95 test samples — extreme small-data regime
- CV-LB gap ~0.01 for good experiments, up to 0.05 for overfit ones
- Anchor submission (0.96624) is foundation for all post-processing
- GBM-based per-horizon heads overfit on this sample size

## Active Requirements
1. **Model diversity**: Activate unused models (CoxPH, WeibullAFT, XGBoostCox) for CI improvement
2. **Feature selection**: Resolve 16→36 feature bloat (caused CV -0.0057)
3. **Distribution calibration**: CSD/CiPOT conformal methods for WBrier
4. **KM-sampling**: Expand effective calibration sample size using censored observations
5. **Stacking simplification**: Replace XGB heads (depth=3, 136 samples) with simpler alternatives
6. **Seed expansion**: 3→5+ seeds for variance reduction

## Out of Scope
- Deep learning models (insufficient data)
- External data sources
- Full pipeline rewrite

## Code Opportunities Found
1. Only RSF+EST used; CoxPH/WeibullAFT/XGBoostCox available but unused
2. Stacking XGB heads (200 trees, depth=3) likely overfit on ~136 samples
3. Feature engineering 16→36 features hurts generalization
4. Ensemble weight optimization target may not align with LB metric
5. Monotonicity PAVA weights hardcoded [1.0, 1.0, 10.0]
6. Only 3 seed averaging (42, 123, 456)

## SOTA Directions
1. **CSD/CiPOT**: Conformal calibration preserves CI while improving Brier
2. **KM-sampling**: Kaplan-Meier reweighting for censored data utilization
3. **Adversarial regularization**: Reduce CV-LB gap through distribution matching

## Key Decisions
- Dual-track: model diversity (CI) + calibration (WBrier)
- Preserve anchor framework (Exp22/23 logit calibration)
- Prioritize WBrier (2.3× leverage over CI)
- Validate every change on LB, not just CV
