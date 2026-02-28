# Roadmap — WiDS 2026 (重构版)

## Milestone 1: 后处理优化突破 0.975

**战略转变**: Phase 1-4 证明 221 样本下模型侧改进无效。新战略聚焦于**系统化后处理空间搜索**。

**当前状态**:
- PB: 0.96783 (Exp23 gate calibration)
- 锚点: 0.96624 (RSF baseline)
- 目标: 0.975+ (gap ~0.007)

**核心假设**: 后处理变换空间尚未充分探索,存在未发现的有效策略。

---

### Phase 5: 非线性变换搜索
**Goal**: 探索预测值的非线性变换空间,寻找比线性 gate calibration 更优的映射

**Requirements**:
- POST-01: 实现 5+ 种非线性变换 (log/sqrt/power/sigmoid/quantile)
- POST-02: 每种变换在 24h/48h 上独立优化参数
- POST-03: 通过 Spearman rho vs 0.96624 筛选候选 (rho < 0.98)

**Success Criteria**:
- 至少 1 个变换 LB > 0.96783
- 找到 2-3 个 rho < 0.98 的多样化候选

**Estimated Submissions**: 5-8

**Plans**:
- [ ] 05-01: 实现变换库 + 参数优化框架
- [ ] 05-02: Grid search 最优变换参数
- [ ] 05-03: LB 验证 top-3 候选

---

### Phase 6: 排名优化与集成
**Goal**: 基于排名而非概率值的后处理,以及多策略集成

**Requirements**:
- RANK-01: 实现 rank-based postprocessing (保留排名,映射到目标分布)
- RANK-02: 集成 Phase 5 的多个候选 (weighted average/stacking)
- RANK-03: 时间点权重优化 (针对 WBrier 公式)

**Success Criteria**:
- Rank-based 方法 LB > 0.96783
- 集成策略 LB > max(单模型)

**Estimated Submissions**: 5-7

**Plans**:
- [ ] 06-01: Rank-based postprocessing 实现
- [ ] 06-02: 多策略集成 (Phase 5 候选)
- [ ] 06-03: 时间点权重优化

---

### Phase 7: 约束优化与分布匹配
**Goal**: 通过约束和分布匹配进一步优化

**Requirements**:
- DIST-01: 分布约束 (匹配 training set 经验分布)
- DIST-02: 优化单调性约束权重 (当前 [1.0, 1.0, 10.0])
- DIST-03: Bootstrap aggregating (221 样本 bootstrap)

**Success Criteria**:
- 分布匹配方法 LB > 0.96783
- Bootstrap aggregating 减小方差

**Estimated Submissions**: 4-6

**Plans**:
- [ ] 07-01: 分布匹配后处理
- [ ] 07-02: 单调性约束权重优化
- [ ] 07-03: Bootstrap aggregating

---

### Phase 8: 伪标签与迭代优化 (Stretch Goal)
**Goal**: 如果 Phase 5-7 有效,使用 test set 伪标签进一步优化

**Requirements**:
- PSEUDO-01: 高置信度 test 样本伪标签
- PSEUDO-02: 迭代优化后处理参数

**Success Criteria**:
- 伪标签方法 LB > Phase 7 最优

**Estimated Submissions**: 3-5

**Plans**:
- [ ] 08-01: 伪标签生成与迭代优化

---

## 止损条件

**Phase-level 止损**:
- 任何 phase 如果所有提交 LB < 0.96283 (PB - 0.005),立即关闭该 phase

**Milestone-level 止损**:
- 如果 Phase 5-7 全部失败 (无提交 > 0.96783),接受 0.96783 作为最终成绩

---

## 已关闭的 Phases (归档)

### Phase 1-2: 模型侧改进 — CLOSED
- LR/XGB head, IPCW stacking, Calibration 全部失败
- 教训: 221 样本下 meta-learning 无法迁移

### Phase 3: Conformal Calibration — CLOSED
- OOF 提升但 LB 无效,与 Phase 2 同模式

### Phase 4: 锚点复现/超参优化 — CLOSED
- PLE fork, RSF grid 全部失败
- 教训: 无法获得比 0.96624 更强的锚点

---

*Last updated: 2026-02-28 after Phase 1-4 failure analysis*
