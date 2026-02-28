# Requirements — WiDS 2026

**Defined:** 2026-02-28 (重构后)
**Core Value:** 后处理优先 — 221 样本下模型侧改进无效

---

## v1 Requirements (后处理优化)

### 非线性变换 (POST)

- [ ] **POST-01**: 实现 5+ 种非线性变换函数 (log/sqrt/power/sigmoid/quantile mapping)
- [ ] **POST-02**: 每种变换在 24h/48h 上独立优化参数 (grid search 或 Bayesian optimization)
- [ ] **POST-03**: 通过 Spearman rho vs 0.96624 筛选候选 (rho < 0.98 确保多样性)
- [ ] **POST-04**: LB 验证 top-3 变换候选

### 排名优化 (RANK)

- [ ] **RANK-01**: 实现 rank-based postprocessing (保留排名,映射到目标分布)
- [ ] **RANK-02**: 集成 Phase 5 的多个候选 (weighted average/stacking/voting)
- [ ] **RANK-03**: 时间点权重优化 (针对 WBrier 公式 0.3×B@24h + 0.4×B@48h + 0.3×B@72h)

### 分布优化 (DIST)

- [ ] **DIST-01**: 分布约束后处理 (匹配 training set 经验分布)
- [ ] **DIST-02**: 优化单调性约束权重 (当前 [1.0, 1.0, 10.0])
- [ ] **DIST-03**: Bootstrap aggregating (221 样本 bootstrap 生成多个后处理版本)

### 伪标签 (PSEUDO) — Stretch Goal

- [ ] **PSEUDO-01**: 高置信度 test 样本伪标签生成
- [ ] **PSEUDO-02**: 迭代优化后处理参数 (使用伪标签)

---

## v2 Requirements (已验证无效,归档)

### 模型侧改进 (已关闭)
- **MODEL-01**: LR/XGB head — LB < baseline
- **MODEL-02**: IPCW stacking — OOF < gate
- **MODEL-03**: Calibration — CV-LB gap 无法迁移

### 锚点优化 (已关闭)
- **ANCHOR-01**: PLE stacker fork — LB < 0.96624
- **ANCHOR-02**: RSF hyperparam grid — 分布压缩

---

## Out of Scope

| Feature | Reason |
|---------|--------|
| 模型侧改进 | Phase 1-4 验证 221 样本下无效 |
| 锚点复现/超参优化 | 无法获得比 0.96624 更强的锚点 |
| 深度学习模型 | 样本量不足 |
| 外部数据源 | 竞赛规则限制 |

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| POST-01 | Phase 5 | Pending |
| POST-02 | Phase 5 | Pending |
| POST-03 | Phase 5 | Pending |
| POST-04 | Phase 5 | Pending |
| RANK-01 | Phase 6 | Pending |
| RANK-02 | Phase 6 | Pending |
| RANK-03 | Phase 6 | Pending |
| DIST-01 | Phase 7 | Pending |
| DIST-02 | Phase 7 | Pending |
| DIST-03 | Phase 7 | Pending |
| PSEUDO-01 | Phase 8 | Pending |
| PSEUDO-02 | Phase 8 | Pending |

**Coverage:**
- v1 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0 ✓

---

*Requirements defined: 2026-02-28*
*Last updated: 2026-02-28 after Phase 1-4 failure analysis*
