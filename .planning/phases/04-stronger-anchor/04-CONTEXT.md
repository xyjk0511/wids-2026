# Phase 4 Context: 更强锚点获取/复现

## 锁定决策

1. **三条 Track 按顺序执行**，Track 1 是前置条件
2. **不做花式改动**：精确复现优先，不加特征/不改模型结构
3. **分阶段目标**: LB > 0.9685 → LB > 0.970 → 0.975 (stretch)
4. **提交门槛**: 仅允许改变排序且有稳定性证据的方案

## Track 1: 公开高分 pipeline 精确复现
- 锁定 2-3 个 0.966+ 来源
- 严格对齐: 特征集/CV策略/seed/sksurv版本/后处理
- 目标: 拿到可迁移的新锚点

## Track 2: 多锚点融合
- 前置: Track 1 产出至少 1 个与 0.96624 排序不完全同质的锚点
- 方法: rank average + weight search
- 仅在 Spearman < 0.99 的锚点对上做 blend

## Track 3: 版本与关键超参小网格
- 前置: Track 1 复现成功
- 范围: sksurv 版本 + RSF 关键超参 (n_estimators, max_features, min_samples_leaf)
- 不做大范围盲搜

## Phase 1-3 教训
- 221 样本上模型侧改进无法超越锚点后处理
- 校准天花板已触及 (logit线性/Platt/非参数三类均失败)
- OOF 改善不等于 LB 改善 (Exp32: OOF+0.0059→LB-0.00445)
- rho=1.0 的方案不值得提交 (排序不变=CI不变)
