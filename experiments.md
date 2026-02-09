# WiDS 2025 - 实验记录

## 提交历史

| # | 日期 | 模型 | Local CV Hybrid | Local CV CI | Local CV WBrier | Kaggle LB | 备注 |
|---|------|------|-----------------|-------------|-----------------|-----------|------|
| 1 | 2026-02-08 | ensemble(rankxgb+rsf+gbsa) | 0.9721 | 0.9420 | 0.0150 | 0.95054 | CV-LB gap 合理, LB 略高于 CV CI, 泛化良好 |
| 2 | 2026-02-08 | ensemble v2 (L2正则+bootstrap+per-horizon+RankXGB独立训练) | 0.9692 | 0.9420 | 0.0191 | 待提交 | L2正则lambda=0.1, bootstrap n=30, per-horizon Brier优化, stacking C=0.1, Platt Scaling禁用(恶化Brier) |
