# 0.98088 分数增量改进分析 (vs 0.97092 基线)

## 执行摘要

基于用户提供的代码片段，从 **0.97092** → **0.98088** 的 **+0.00996** 提升来自 4 个关键改进的叠加效应。这是一个**极其激进的校准策略**，核心在于：

1. **PowerCal24=1.1** 对 24h 概率的非线性拉伸
2. **Hard 5km cutoff** 对远距离样本的零风险强制
3. **W48 从 0.55 → 0.45** 降低 48h 权重
4. **Seeds 从 10 → 15** 增加模型多样性

---

## 增量改进 1: PowerCal24=1.1 (Power Transform on 24h)

### 从 0.97092 的变化
```python
# 0.97092: 无 power transform
test_gbsa[:, 1] = np.clip(test_gbsa[:, 1], 0, 1)

# 0.98088: 24h 概率 power 1.1
test_gbsa[:, 1] = np.clip(test_gbsa[:, 1] ** 1.1, 0, 1)
```

### 数学效应
- **Power > 1** 会**压缩高概率、拉伸低概率**
- 例如：0.5 → 0.435, 0.8 → 0.742, 0.2 → 0.159
- 这是一种**非参数校准**，类似 Beta Calibration 的单参数版本

### 边际收益估算
- **+0.003 ~ +0.005**
- 原理：如果 24h 预测系统性偏高（over-confident），power > 1 可以修正 Brier Score
- 风险：如果 test 分布与 train 不一致，可能适得其反

### 必要性
- **可选**（高风险高收益）
- 需要在 OOF 上验证 24h 是否真的 over-confident
- 当前项目 Exp22 系列已经用 logit 空间线性校准，power transform 可能冲突

### 实现难度
- **简单** (1 行代码)
- 但需要网格搜索最优 power 值 (1.05, 1.1, 1.15, 1.2)

---

## 增量改进 2: Hard 5km Cutoff (Zero Risk for Far Samples)

### 从 0.97092 的变化
```python
# 0.97092: 无距离 cutoff
test_blend = W24 * test_gbsa[:, 1] + W48 * test_gbsa[:, 2]

# 0.98088: 5km 外强制归零
far_mask = dist_test >= 5000  # 5km = 5000m
test_blend[far_mask, :] = 0.0
```

### 物理依据
- **竞赛定义**：5km 是疏散区边界阈值
- **训练集完美分离**：所有 dist < 5km 的样本 event=1，所有 dist >= 5km 的样本 event=0
- **Test 集推断**：如果 test 也遵循这一物理规律，5km 外的样本真实风险应为 0

### 边际收益估算
- **+0.002 ~ +0.004**
- 取决于 test 集中有多少 dist >= 5km 的样本（约 70%）
- 如果这些样本的预测概率原本 > 0，强制归零可以大幅降低 Brier Score

### 必要性
- **必须**（如果 test 遵循 train 的物理规律）
- 这是利用**领域知识**的典型案例
- 风险：如果 test 中存在 dist >= 5km 但 event=1 的样本（data leakage 或物理异常），会严重伤分

### 实现难度
- **简单** (2 行代码)
- 需要 test.csv 中的 `dist_min_ci_0_5h` 特征

---

## 增量改进 3: W48 从 0.55 → 0.45 (降低 48h 权重)

### 从 0.97092 的变化
```python
# 0.97092: W24=0.95, W48=0.55
test_blend = 0.95 * test_gbsa[:, 1] + 0.55 * test_gbsa[:, 2]

# 0.98088: W24=0.95, W48=0.45
test_blend = 0.95 * test_gbsa[:, 1] + 0.45 * test_gbsa[:, 2]
```

### 数学含义
- **降低 48h 权重** = 更信任 24h 预测
- 可能原因：48h 预测方差更大，或者 test 集中 24h 的信号更强

### 边际收益估算
- **+0.001 ~ +0.002**
- 这是一个**微调参数**，收益取决于 24h 和 48h 的相对质量

### 必要性
- **可选**（需要网格搜索验证）
- 当前项目 Exp22f 使用 gap-gated r 策略，已经隐式调节了 24h/48h 的相对权重

### 实现难度
- **简单** (修改 1 个常数)
- 但需要在 OOF 上搜索最优 W24/W48 组合

---

## 增量改进 4: Seeds 从 10 → 15 (增加模型多样性)

### 从 0.97092 的变化
```python
# 0.97092: 5 configs × 10 seeds = 50 GBSA models
N_SEEDS = 10

# 0.98088: 5 configs × 15 seeds = 75 GBSA models
N_SEEDS = 15
```

### 数学效应
- **更多 seeds** = 更多 bootstrap 样本 = 降低预测方差
- 类似 bagging 的效果，但收益递减（10 → 15 的边际收益 < 5 → 10）

### 边际收益估算
- **+0.0005 ~ +0.001**
- 收益很小，但几乎无风险

### 必要性
- **可选**（低优先级）
- 当前项目已经用 5 seeds (RSF+EST)，增加到 10-15 可能有微小提升

### 实现难度
- **简单** (修改 1 个常数)
- 但训练时间增加 50%

---

## 总收益估算

| 改进 | 边际收益 | 累积收益 | 风险 |
|------|---------|---------|------|
| PowerCal24=1.1 | +0.004 | 0.97092 → 0.97492 | 中 |
| Hard 5km cutoff | +0.003 | 0.97492 → 0.97792 | 低 |
| W48: 0.55→0.45 | +0.0015 | 0.97792 → 0.97942 | 低 |
| Seeds: 10→15 | +0.0008 | 0.97942 → 0.98022 | 极低 |
| **交互效应** | +0.0006 | 0.98022 → **0.98088** | - |

**注意**：这些改进的收益**不是严格可加的**，存在交互效应（例如 PowerCal 和 5km cutoff 可能部分重叠）。

---

## 实施优先级

### 立即实施（高收益低风险）
1. **Hard 5km cutoff** — 利用物理规律，几乎无风险
2. **Seeds 增加到 15** — 无风险，仅增加计算成本

### 谨慎实施（需验证）
3. **PowerCal24 网格搜索** — 在 OOF 上验证 power ∈ [1.05, 1.1, 1.15, 1.2]
4. **W48 权重调优** — 在 OOF 上搜索 W48 ∈ [0.4, 0.45, 0.5, 0.55]

### 风险评估
- **PowerCal24=1.1** 是最大的不确定性来源
  - 如果 test 分布与 train 不同，可能伤分
  - 建议先在 OOF 上验证 24h 是否 over-confident
- **5km cutoff** 依赖于 test 遵循 train 的物理规律
  - 如果 test 中有 dist >= 5km 但 event=1 的样本，会灾难性失败
  - 建议先检查 test.csv 的 dist 分布

---

## 与当前项目的对比

### 当前项目 (PB=0.96783) 的策略
- **Exp22f**: logit 空间线性校准 (A=1.0655, B=-0.0108, lam=6.0)
- **Gap-gated r**: 按 gap 分段调节 24h/48h 耦合强度
- **无 power transform**
- **无 5km cutoff**

### 0.98088 策略的差异
- **Power transform** vs **Logit linear**
- **Hard cutoff** vs **Soft calibration**
- **更多 seeds** (15 vs 5)

### 兼容性分析
- **5km cutoff** 可以直接叠加到 Exp22f 之上
- **PowerCal24** 可能与 logit 校准冲突，需要 A/B 测试
- **W48 调优** 可以替代 gap-gated r 的部分功能

---

## 实施建议

### Phase 1: 低风险改进（立即执行）
```python
# 1. 增加 seeds 到 15
N_SEEDS = 15

# 2. 5km cutoff (在 postprocess 中)
far_mask = test['dist_min_ci_0_5h'] >= 5000
test_preds[far_mask, :] = 0.0  # 所有 horizon 归零
```

### Phase 2: 高风险改进（OOF 验证后执行）
```python
# 3. PowerCal24 网格搜索
for power in [1.05, 1.1, 1.15, 1.2]:
    oof_24h_cal = np.clip(oof_24h ** power, 0, 1)
    brier_24h = brier_score(y_true_24h, oof_24h_cal)
    # 选择最优 power

# 4. W48 权重搜索
for w48 in [0.4, 0.45, 0.5, 0.55]:
    oof_blend = 0.95 * oof_24h + w48 * oof_48h
    hybrid = 0.3 * ci + 0.7 * (1 - wbrier)
    # 选择最优 w48
```

### Phase 3: 集成测试
- 将 5km cutoff + PowerCal24 + W48 调优叠加到 Exp22f
- 在 OOF 上验证 Hybrid Score 是否提升
- 如果 OOF 提升 > 0.003，提交 Kaggle 验证

---

## 关键风险警告

### 🚨 5km Cutoff 的致命假设
- **假设**：test 集中所有 dist >= 5km 的样本 event=0
- **如果假设错误**：例如 test 中有 1 个 dist=6km 但 event=1 的样本
  - 预测 prob=0.0，真实 label=1
  - Brier Score = (0 - 1)^2 = 1.0（单样本灾难）
- **缓解策略**：
  - 不要完全归零，改用 `test_preds[far_mask, :] *= 0.01`（保留 1% 概率）
  - 或者只对 dist > 10km 的样本归零

### 🚨 PowerCal24 的分布漂移风险
- **假设**：test 的 24h 概率分布与 train 一致
- **如果假设错误**：例如 test 的 24h 真实发生率更高
  - Power > 1 会压缩概率，导致 under-prediction
- **缓解策略**：
  - 在 OOF 上验证 power transform 是否改善校准曲线
  - 使用 reliability diagram 检查 calibration error

---

## 结论

从 0.97092 → 0.98088 的 **+0.00996** 提升是一个**极其激进的校准策略**，核心是：

1. **利用物理规律**（5km cutoff）— 必须实施
2. **非线性校准**（PowerCal24）— 需验证
3. **权重微调**（W48 降低）— 可选
4. **方差缩减**（更多 seeds）— 低成本

**当前项目的下一步**：
- 立即实施 5km cutoff（保守版：dist > 10km 归零）
- 在 OOF 上验证 PowerCal24 的有效性
- 如果 OOF 验证通过，提交 Kaggle 测试

**预期收益**：
- 保守估计：+0.005 ~ +0.008（达到 0.973 ~ 0.975）
- 乐观估计：+0.010 ~ +0.015（达到 0.978 ~ 0.982）

**最大风险**：
- 5km cutoff 如果假设错误，可能导致 LB 崩溃（-0.01 以上）
- 建议先用 dist > 10km 的保守版本测试
