# Roadmap — WiDS 2026 (基于 0.97092/0.98088 复现)

## Milestone 1: 复现公开高分方法突破 0.975

**战略转变 v2**: 发现 Public LB 最高 0.98088,证明**不是 221 样本限制,是方法论代差**。

**当前状态**:
- PB: 0.96783 (Exp23 gate calibration)
- LB 最高: 0.98088 (Klaus & Shirley)
- LB 公开方案: 0.97092 (10+ 队伍相同分数)
- 目标: 复现 0.97092 → 增量到 0.975+

**核心发现**: Phase 1-4 失败不是因为样本量,而是**缺少 GBSA 50-model ensemble + IPCW LightGBM**。

---

## Phase 5: GBSA 50-Model Ensemble (复现 0.97092 核心)

**Goal**: 复现 0.97092 的核心架构 — GBSA multi-config ensemble

**Requirements**:
- GBSA-01: 实现 5 configs × 10 seeds = 50 GBSA models
- GBSA-02: OOF hybrid score > 0.970 (门槛)
- GBSA-03: LB > 0.968 (验证复现成功)

**Technical Details**:
```python
# 5 个超参配置 (从 0.97092 notebook)
configs = [
    {'lr': 0.01, 'ss': 0.7,  'msl': 12, 'n': 1200},
    {'lr': 0.01, 'ss': 0.85, 'msl': 15, 'n': 1200},
    {'lr': 0.01, 'ss': 0.6,  'msl': 12, 'n': 1200},
    {'lr': 0.005,'ss': 0.85, 'msl': 12, 'n': 2000},
    {'lr': 0.01, 'ss': 0.85, 'msl': 20, 'n': 1400},
]
# 10 seeds: 42-51
```

**Success Criteria**:
- OOF hybrid ≥ 0.970
- LB ≥ 0.968
- 与 0.97092 的 gap < 0.002

**Estimated Submissions**: 2-3

**Plans**:
- [ ] 05-01: 安装 scikit-survival 0.22+, 实现 GBSA ensemble
- [ ] 05-02: OOF 验证 (目标 > 0.970)
- [ ] 05-03: LB 验证 (目标 > 0.968)

**预期提升**: +0.002-0.004 (从 0.96783 → 0.970)

---

## Phase 6: LightGBM IPCW Per-Horizon (复现 0.97092 完整)

**Goal**: 添加 LightGBM per-horizon 分类头,完整复现 0.97092

**Requirements**:
- IPCW-01: 实现 IPCW 权重计算 (Kaplan-Meier)
- IPCW-02: 训练 24h/48h 独立 LightGBM 分类器
- IPCW-03: Asymmetric blend (W_GBSA_24=0.95, W_GBSA_48=0.55)

**Technical Details**:
```python
# 24h: 深度3, 强正则化
lgb_24h = {'max_depth': 3, 'lr': 0.03, 'n': 300,
           'reg_alpha': 0.5, 'reg_lambda': 2.0}

# 48h: 深度2, 弱正则化
lgb_48h = {'max_depth': 2, 'lr': 0.05, 'n': 200,
           'reg_alpha': 0.1, 'reg_lambda': 1.0}
```

**Success Criteria**:
- LB ≥ 0.970 (完整复现 0.97092)
- Blend 权重验证: 24h 高权重 GBSA, 48h 平衡

**Estimated Submissions**: 2-3

**Plans**:
- [ ] 06-01: 实现 IPCW 权重计算
- [ ] 06-02: 训练 LightGBM 24h/48h 分类器
- [ ] 06-03: Asymmetric blend + LB 验证

**预期提升**: +0.001-0.002 (从 0.970 → 0.971)

---

## Phase 7: 增量改进到 0.975+ (基于 0.98088 策略)

**Goal**: 在 0.97092 基础上叠加 0.98088 的增量改进

**Requirements**:
- INCR-01: 5km hard cutoff (利用物理规律)
- INCR-02: PowerCal24 grid search (power transform)
- INCR-03: Seeds 增加到 15

**Technical Details**:
```python
# 1. 5km cutoff (保守版: 10km)
far_mask = dist_test >= 10000  # 先用 10km 测试
test_preds[far_mask, :] = 0.0

# 2. PowerCal24 (需 OOF 验证)
test_preds[:, 1] = test_preds[:, 1] ** 1.1

# 3. Seeds 10 → 15
N_SEEDS = 15
```

**Success Criteria**:
- 5km cutoff: LB > 0.972
- PowerCal24: LB > 0.974
- Seeds 15: LB > 0.975

**Estimated Submissions**: 5-8

**Plans**:
- [ ] 07-01: 5km cutoff (保守版 10km)
- [ ] 07-02: PowerCal24 grid search [1.05, 1.1, 1.15, 1.2]
- [ ] 07-03: Seeds 增加到 15
- [ ] 07-04: W48 权重优化 [0.4, 0.45, 0.5, 0.55]

**预期提升**: +0.005-0.008 (从 0.971 → 0.976-0.979)

---

## Phase 8: 冲刺 0.980 (Stretch Goal)

**Goal**: 如果 Phase 7 成功,尝试接近第一名 0.98088

**Requirements**:
- 组合所有有效改进
- 微调所有超参数
- 可能需要特征工程

**Success Criteria**:
- LB ≥ 0.980

**Estimated Submissions**: 5-10

**Plans**:
- [ ] 08-01: 全局超参优化
- [ ] 08-02: 特征工程 (如果需要)
- [ ] 08-03: 最终集成

**预期提升**: +0.001-0.005 (从 0.976 → 0.980)

---

## 止损条件

**Phase-level 止损**:
- Phase 5: 如果 LB < 0.965,停止 GBSA 方向
- Phase 6: 如果 LB < 0.968,停止 IPCW 方向
- Phase 7: 如果任何改进 LB < 0.970,跳过该改进

**Milestone-level 止损**:
- 如果 Phase 5-6 失败 (LB < 0.970),接受 0.96783 作为最终成绩

---

## 已关闭的 Phases (归档)

### Phase 1-4: 错误方向 — CLOSED
- **Phase 1-2**: 模型侧改进 (LR/XGB head, IPCW stacking) — 实现方式错误
- **Phase 3**: Conformal Calibration — 在 221 样本上过拟合
- **Phase 4**: 锚点复现/超参优化 — 方向错误,应该做 GBSA ensemble

**核心教训**:
- 不是 221 样本限制,是方法论落后
- IPCW stacking 我们失败了,但 0.97092 成功了
- 后处理校准可能是负贡献 (0.97036 OOF 高但 LB 低)

---

## 关键风险

### 风险 1: GBSA 训练时间
- 50 models 可能需要 2-3 小时
- 缓解: 先用 10 models 快速验证

### 风险 2: 5km cutoff 假设错误
- 如果 test 中有 dist >= 5km 但 event=1 的样本,会灾难性失败
- 缓解: 先用 10km cutoff 保守测试

### 风险 3: PowerCal24 分布漂移
- 如果 test 分布与 train 不同,power transform 可能伤分
- 缓解: 在 OOF 上验证后再提交

---

*Last updated: 2026-02-28 after discovering 0.97092/0.98088 public methods*
*Key insight: Not sample size limit, but methodology gap*
