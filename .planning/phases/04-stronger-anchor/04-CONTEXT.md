# Phase 4: 更强锚点获取/复现 - Context

**Gathered:** 2026-02-23
**Status:** Ready for planning

<domain>
## Phase Boundary

获取比 0.96624 更强的基础预测（LB > 0.968）。
三条 Track：Track 1 = Kaggle 原始 notebook 运行，Track 2 = 多锚点 blend，Track 3 = 超参网格。
本地模型复现（exp30-33）已证明无效，不再尝试。

</domain>

<decisions>
## Implementation Decisions

### Track 1 转向策略
- 不做本地复现，直接在 Kaggle 上 fork + run 原始 notebook
- 两个目标：suman2208/ple-stacker (LB=0.96654) 和 rhythmghai/ridge-stacker (LB=0.96536)
- 仅加 OOF 输出行，其余代码不动
- 输出：submission.csv + OOF 预测（用于 blend 相关性分析）

### Blend 有效性条件
- rho 门槛：新锚点 vs 0.96624 的 Spearman rho < 0.99 即可 blend
- LB 门槛：新锚点 LB 必须 > 0.96624，否则 blend 无意义
- Blend 方法：value-space weighted blend
- 权重确定：OOF 驱动（用 OOF 相关性/质量决定权重）

### 提交预算分配
- 每天上限：5 次
- 优先级：Track 1 优先，最多用 5 次提交
- Track 1 止损：5 次后若无 LB > 0.96624 的新锚点，切换到 Track 3
- Track 3 规模：大网格 10+ 次（sksurv 版本 + RSF 超参全面扫描）

### Claude's Discretion
- Track 3 具体超参范围（n_estimators, max_features, min_samples_leaf 的网格点）
- OOF 权重的具体计算公式
- Kaggle notebook fork 的操作步骤

</decisions>

<specifics>
## Specific Ideas

- Track 1 的核心价值：拿到原始 notebook 的精确预测，而非本地近似复现
- exp30-33 教训：本地复现的模型全部是噪声（最好的 blend 也只有 0.96540 < 0.96624）
- Track 3 是 Track 1 失败后的主要备选，不是并行方向

</specifics>

<deferred>
## Deferred Ideas

- 无 — 讨论范围保持在 Phase 04 边界内

</deferred>

---

*Phase: 04-stronger-anchor*
*Context gathered: 2026-02-23*
