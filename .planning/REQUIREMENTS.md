# Requirements — WiDS 2026

**Defined:** 2026-02-28 (v3 - GBSA 方向)
**Core Value:** 方法论升级 — 复现 0.97092/0.98088 的 GBSA ensemble + IPCW 架构

---

## v3 Requirements (GBSA 复现方向)

### GBSA Ensemble (Phase 5)

- [ ] **GBSA-01**: 实现 5 configs × 10 seeds = 50 GBSA models
- [ ] **GBSA-02**: OOF hybrid score > 0.970 (门槛)
- [ ] **GBSA-03**: LB > 0.968 (验证复现成功)

### IPCW LightGBM (Phase 6)

- [ ] **IPCW-01**: 实现 IPCW 权重计算 (Kaplan-Meier)
- [ ] **IPCW-02**: 训练 24h/48h 独立 LightGBM 分类器
- [ ] **IPCW-03**: Asymmetric blend (W_GBSA_24=0.95, W_GBSA_48=0.55)

### 增量改进 (Phase 7)

- [ ] **INCR-01**: 5km hard cutoff (利用物理规律)
- [ ] **INCR-02**: PowerCal24 grid search (power transform)
- [ ] **INCR-03**: Seeds 增加到 15

---

## v2 Requirements (后处理优化 - 已废弃)

### 非线性变换 (POST) - 已废弃

- **POST-01**: 实现 5+ 种非线性变换函数 — 被 GBSA-01 替代
- **POST-02**: 每种变换在 24h/48h 上独立优化参数 — 被 IPCW-02 替代
- **POST-03**: 通过 Spearman rho vs 0.96624 筛选候选 — 被 GBSA-02 替代
- **POST-04**: LB 验证 top-3 变换候选 — 被 GBSA-03 替代

### 排名优化 (RANK) - 已废弃

- **RANK-01**: 实现 rank-based postprocessing — 方向错误
- **RANK-02**: 集成 Phase 5 的多个候选 — 方向错误
- **RANK-03**: 时间点权重优化 — 被 IPCW-03 替代

---

## v1 Requirements (已验证无效,归档)

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
| GBSA-01 | Phase 5 | Pending |
| GBSA-02 | Phase 5 | Pending |
| GBSA-03 | Phase 5 | Pending |
| IPCW-01 | Phase 6 | Pending |
| IPCW-02 | Phase 6 | Pending |
| IPCW-03 | Phase 6 | Pending |
| INCR-01 | Phase 7 | Pending |
| INCR-02 | Phase 7 | Pending |
| INCR-03 | Phase 7 | Pending |

**Coverage:**
- v3 requirements: 9 total
- Mapped to phases: 9
- Unmapped: 0 ✓

---

*Requirements defined: 2026-02-28*
*Last updated: 2026-02-28 v3 - switched to GBSA replication strategy*
