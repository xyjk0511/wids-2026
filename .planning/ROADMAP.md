# Roadmap — WiDS 2026

## Milestone 1: Score Improvement (止损模式)

**阶段目标**: 先破 0.9685，再谈 0.975
**核心教训**: Phase 1+2 证明 221 样本上模型侧改进无法超越锚点后处理
**战略判断**: 到 0.975+ 大概率需要更强锚点，而非校准/模型改进

### Phase 1: Stacking & Feature Baseline Fix
**Goal**: Remove known overfitting sources, establish clean baseline
**Requirements**: R2 (Feature Selection), R5 (Stacking Simplification)
**Plans:** 1/2 plans executed

Plans:
- [ ] 01-01-PLAN.md — LR-head baseline (config flags only, zero code changes)
- [ ] 01-02-PLAN.md — Backward elimination for optimal feature subset

**Success**: CV stable or improved; LB >= 0.96783 (current PB)
**Estimated submissions**: 2-3

### Phase 2: Model Diversity Ensemble — CLOSED (both directions failed)
**Goal**: Improve LB through IPCW-aware stacking and new calibration methods
**Requirements**: R1 (Model Diversity) — R6 (Seed Expansion) pre-completed
**Plans:** 2/2 executed, both failed

Plans:
- [x] 02-01-PLAN.md — IPCW stacking: gate fail (OOF=0.96108 < 0.9697)
- [x] 02-02-PLAN.md — Calibration: LB=0.96338 (OOF +0.0059 but LB -0.00445)

**Result**: Phase goal NOT met. 221 samples insufficient for meta-learning and calibration transfer.
**Submissions used**: 1 (exp32)

### Phase 3: Conformal Calibration (止损式，最小实验)
**Goal**: 测试 anchor-based split-conformal 能否在 24h/48h 上提分
**约束**: 仅 2 条实验，提交上限 3 次
**硬门槛**: OOF Hybrid >= +0.0015, rho24/48 >= 0.90, CI 不下降 → 否则不提交
**止损**: 若 3 次提交后仍 < 0.9685，立即关闭，转 Phase 4

Plans:
- [ ] 03-01-PLAN.md — Split-conformal on anchor 24h/48h (2 experiments max)

**Success**: LB > 0.9685
**Estimated submissions**: 1-3
**Depends on**: Phase 2 closed

### Phase 4: 更强锚点获取/复现
**Goal**: 获取或复现比 0.96624 更强的基础预测
**方向**:
- 复现/改进参考 notebook pipeline（精确对齐 sksurv 版本、超参、后处理）
- 寻找公开高分 notebook 作为新锚点
- 多锚点融合（如有多个 0.966+ 来源）
**Success**: 新锚点 LB > 0.968
**Estimated submissions**: 5-10
