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

### Phase 3: Conformal Calibration — CLOSED (止损，不提交)
**Goal**: 测试 anchor-based split-conformal 能否在 24h/48h 上提分
**Plans:** 1/1 executed, gate passes on OOF but pattern matches Exp32 failure

Plans:
- [x] Exp33 split-conformal: OOF +0.0071 但 rho=1.0(排序不变), 阶梯校准图, 与Exp32同模式

**Result**: 止损不提交. 三类校准(logit线性/Platt参数/非参数分位数)均证明N=221校准无法迁移到LB.
**Submissions used**: 0

### Phase 4: 更强锚点获取/复现
**Goal**: 获取或复现比 0.96624 更强的基础预测
**Requirements**: ANCHOR-01 (Reproduction), ANCHOR-02 (Blending), ANCHOR-03 (Hyperparam Grid)
**Plans:** 3 plans (2 complete + 1 gap closure)

Plans:
- [x] 04-01-PLAN.md — Track 1: Reproduction tooling + suman2208 fork: LB=0.96086 (< gate, stop-loss triggered)
- [x] 04-02-PLAN.md — Track 3: RSF hyperparam grid: LB=0.91089/0.90860 (catastrophic, pipeline inconsistency, phase closed)
- [ ] 04-03-PLAN.md — Gap closure: fix pipeline parity in exp30_hyperparam_grid.py + re-run grid

**分阶段目标**: LB > 0.9685 → LB > 0.970 → 0.975 (stretch)
**Success**: 新锚点 LB > 0.968
**Estimated submissions**: 5-10
