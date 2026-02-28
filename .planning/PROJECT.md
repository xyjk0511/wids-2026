# WiDS 2026 Kaggle — 后处理优化突破 0.975

## Vision
通过系统化后处理空间搜索,从 PB=0.96783 突破到 0.975+。

## Core Value
**后处理优先**: Phase 1-4 证明 221 样本下模型侧改进无效。所有优化聚焦于后处理变换空间。

## Current State
- **PB**: 0.96783 (Exp23 gate calibration)
- **锚点**: 0.96624 (RSF baseline)
- **Target**: 0.975+
- **Gap**: ~0.007 hybrid score
- **Metric**: hybrid = 0.3×CI(12h) + 0.7×(1-WBrier), WBrier = 0.3×B@24h + 0.4×B@48h + 0.3×B@72h
- **Submissions remaining**: >20

## 已验证的失败方向 (Stop-Loss)
1. **模型侧改进**: LR/XGB head, IPCW stacking, Calibration 全部失败
2. **锚点复现**: PLE fork, RSF grid 全部失败
3. **核心教训**: 221 样本下 meta-learning 无法从 CV 迁移到 LB

## 新战略方向
1. **非线性变换**: log/sqrt/power/sigmoid/quantile
2. **排名优化**: 基于排名而非概率值的后处理
3. **集成策略**: 多后处理策略集成
4. **约束优化**: 分布匹配,单调性权重优化

## Key Constraints
- 221 train / 95 test samples — extreme small-data regime
- 锚点 0.96624 是当前 pipeline 上限
- 后处理变换空间尚未充分探索

## Active Requirements
1. **POST**: 非线性变换搜索与参数优化
2. **RANK**: 排名优化与多策略集成
3. **DIST**: 分布匹配与约束优化

## Out of Scope
- 模型侧改进 (已验证无效)
- 锚点复现/超参优化 (已验证无效)
- 深度学习模型
- 外部数据源

## Key Decisions
| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 后处理优先战略 | Phase 1-4 模型侧改进全部失败 | — Pending |
| 保留 0.96624 锚点 | 无法获得更强锚点 | ✓ Good |
| 目标 0.975+ | stretch goal,激励探索 | — Pending |
