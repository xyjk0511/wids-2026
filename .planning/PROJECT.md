# WiDS 2026 Kaggle — 复现公开高分方法突破 0.975

## Vision
复现 Public LB 0.97092/0.98088 的公开方法,从 PB=0.96783 突破到 0.975+。

## Core Value
**方法论升级**: 发现 LB 最高 0.98088,证明不是 221 样本限制,而是方法论代差。核心是 **GBSA 50-model ensemble + IPCW LightGBM**,而非后处理优化。

## Current State
- **PB**: 0.96783 (Exp23 gate calibration)
- **锚点**: 0.96624 (RSF baseline)
- **Target**: 0.975+
- **Gap**: ~0.007 hybrid score
- **Metric**: hybrid = 0.3×CI(12h) + 0.7×(1-WBrier), WBrier = 0.3×B@24h + 0.4×B@48h + 0.3×B@72h
- **Submissions remaining**: >20

## 已验证的失败方向 (Stop-Loss)
1. **模型侧改进**: LR/XGB head 实现方式错误
2. **IPCW stacking**: Exp31 失败,但 0.97092 成功 (实现差异)
3. **后处理校准**: Exp32/33 过拟合,0.97036 证明 Temperature Scaling 也过拟合
4. **锚点复现**: PLE fork, RSF grid 方向错误

## 新战略方向 (基于 0.97092/0.98088)
1. **GBSA 50-model ensemble**: 5 configs × 10 seeds (最大单项差异 +0.002-0.004)
2. **LightGBM IPCW per-horizon**: 24h/48h 独立分类器 (+0.001-0.002)
3. **5km hard cutoff**: 利用物理规律 (+0.002-0.004)
4. **PowerCal24**: Power transform (+0.003-0.005)
5. **Seeds 增加**: 10 → 15 (+0.0005-0.001)

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
