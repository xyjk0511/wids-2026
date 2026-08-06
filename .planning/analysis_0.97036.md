# 0.97036 vs 0.97092 Notebook 方法论对比分析

**Date**: 2026-02-28
**Current PB**: 0.96783 (Exp23 gate calibration)
**Target A**: 0.97036 (OOF 0.97505)
**Target B**: 0.97092 (OOF 0.97491)

---

## 方法论对比

### 0.97092 核心方法
- **模型架构**: GBSA Multi-Config Ensemble (5 configs × 10 seeds = 50 models)
- **分类头**: LightGBM Per-Horizon IPCW (24h/48h 独立训练)
- **Blend 策略**: 不对称权重 (W_GBSA_24=0.95, W_GBSA_48=0.55)
- **后处理**: 无显式校准，直接输出 blend 结果
- **OOF**: 0.97491
- **LB**: 0.97092

### 0.97036 核心方法
根据你提供的代码片段：
- **模型架构**: GBSA + LightGBM 双模型系统
- **校准方法**: Temperature Scaling (logit 空间温度缩放)
- **Blend 策略**:
  - 24h: W_GBSA=0.86, W_LGB=0.14, T=0.84
  - 48h: W_GBSA=0.55, W_LGB=0.45, T=0.88
- **后处理**: 先 blend 再 temperature scaling
- **OOF**: 0.97505 (高于 0.97092)
- **LB**: 0.97036 (低于 0.97092)

### 关键差异

| 维度 | 0.97092 | 0.97036 | 影响 |
|------|---------|---------|------|
| **GBSA 规模** | 50 models (5 configs × 10 seeds) | 未知 (可能更少) | 方差缩减 |
| **Blend 权重** | W_24=0.95, W_48=0.55 | W_24=0.86, W_48=0.55 | 24h 信任度 |
| **校准方法** | 无 | Temperature Scaling (T<1) | 概率压缩 |
| **OOF Hybrid** | 0.97491 | 0.97505 | +0.00014 |
| **LB Score** | 0.97092 | 0.97036 | -0.00056 |
| **CV-LB Gap** | 0.00399 | 0.00469 | +0.00070 |

---

## Temperature Scaling 技术分析

### 原理

Temperature Scaling 是一种 logit 空间的后验校准方法：

```python
def apply_temp(p, T):
    """Logit-space temperature scaling."""
    p_safe = np.clip(p, 1e-6, 1 - 1e-6)
    logit_p = np.log(p_safe / (1 - p_safe))  # logit(p)
    scaled_logit = logit_p / T                # 除以温度
    return 1 / (1 + np.exp(-scaled_logit))    # expit(scaled_logit)
```

**温度参数 T 的作用**:
- **T < 1** (如 0.84, 0.88): 增大 logit 值 → 概率向 0/1 两端推 → **增加置信度** (sharpen)
- **T > 1**: 缩小 logit 值 → 概率向 0.5 收缩 → **降低置信度** (soften)
- **T = 1**: 无变换

**0.97036 的温度选择**:
- T24 = 0.84: 24h 预测更激进 (sharpen 更多)
- T48 = 0.88: 48h 预测相对保守 (sharpen 较少)

### 优势 (相比 PowerCal p^1.1)

1. **理论基础更强**: Temperature Scaling 是神经网络校准的标准方法 (Guo et al. 2017)
2. **单参数简洁**: 只需调 T，而 PowerCal 需要调指数
3. **Logit 空间线性**: 在 logit 空间是线性变换，保持排序不变 (Spearman rho=1.0)
4. **可解释性**: T 直接对应"模型过自信"程度

### 劣势 (相比 PowerCal p^1.1)

1. **排序不变**: 由于 logit 空间单调变换，**不改变样本排序** → 对 C-index 无贡献
2. **全局参数**: 单一 T 对所有样本施加相同变换，无法处理 subgroup 差异
3. **边界效应**: 极端概率 (接近 0/1) 的变换幅度有限
4. **需要验证集**: T 的选择需要在 OOF 上拟合，221 样本下容易过拟合

**PowerCal (p^1.1) 的对比**:
- **优势**: 非线性变换，可以改变排序 (如果指数 ≠ 1)
- **劣势**: 缺乏理论支持，指数选择更 ad-hoc

---

## CV-LB Gap 启示

### 观察: OOF 高但 LB 低

| Metric | 0.97036 | 0.97092 | 差异 |
|--------|---------|---------|------|
| OOF Hybrid | 0.97505 | 0.97491 | +0.00014 (0.97036 更高) |
| LB Score | 0.97036 | 0.97092 | -0.00056 (0.97036 更低) |
| CV-LB Gap | 0.00469 | 0.00399 | +0.00070 (0.97036 gap 更大) |

### 原因推测

#### 1. Temperature Scaling 在 221 样本上过拟合

**机制**:
- T24=0.84, T48=0.88 是在 OOF 上拟合的最优值
- 221 样本的 OOF 分布与 95 样本的 test 分布可能不同
- Temperature Scaling 是**全局参数**，对所有样本施加相同变换
- 如果 train/test 分布不匹配，全局 T 会在 test 上失效

**证据**:
- OOF +0.00014 但 LB -0.00056 → 净损失 0.00070
- 这与我们的 Exp32 (Platt calibration) 和 Exp33 (conformal) 的模式一致：
  - Exp32: OOF +0.0059, LB -0.00445
  - Exp33: OOF +0.0071, rho=1.0 (排序不变)

#### 2. 24h Blend 权重差异的影响

**0.97092**: W_GBSA_24 = 0.95 (几乎全用 GBSA)
**0.97036**: W_GBSA_24 = 0.86 (LGB 占 14%)

**推测**:
- 0.97036 在 24h 上给 LGB 更多权重 (14% vs 5%)
- 如果 LGB 在 24h 上过拟合 (221 样本下 GBM 容易过拟合)，会伤害泛化
- 0.97092 的 W=0.95 更保守，更依赖 GBSA 的稳定性

#### 3. Temperature Scaling 的边界效应

**机制**:
- T < 1 会将概率向 0/1 推
- 如果原始预测已经接近边界 (如 p > 0.9)，T=0.84 会进一步推向 1.0
- 在 test set 上，如果真实标签分布不同，这种"过度自信"会增大 Brier loss

**数值示例**:
```
原始 p=0.8, T=0.84:
  logit(0.8) = 1.386
  scaled = 1.386 / 0.84 = 1.650
  new_p = expit(1.650) = 0.839  (+0.039)

原始 p=0.95, T=0.84:
  logit(0.95) = 2.944
  scaled = 2.944 / 0.84 = 3.505
  new_p = expit(3.505) = 0.971  (+0.021)
```

高概率样本的变换幅度更小，但如果 test set 中这些样本的真实标签不是 1，Brier loss 会增大。

---

## 对我们的建议

### 1. 避免在 221 样本上做全局校准

**教训**: 0.97036 的 Temperature Scaling 和我们的 Exp32/Exp33 都显示：
- 在小样本 (221) 上拟合的校准参数无法泛化到 test (95)
- OOF 提升不等于 LB 提升

**行动**:
- **不要**在当前 pipeline 上添加 Temperature Scaling
- **不要**在 OOF 上拟合任何全局校准参数 (T, Platt A/B, 分位数映射)
- 如果必须校准，使用**无参数方法** (如 isotonic regression) 或**极简参数** (如 Exp22 的 A/B 从参考锚点借用)

### 2. 优先复现 0.97092 而非 0.97036

**原因**:
- 0.97092 的 LB 更高 (+0.00056)
- 0.97092 的 CV-LB gap 更小 (0.00399 vs 0.00469)
- 0.97092 **无后处理校准**，泛化性更好
- 0.97036 的 Temperature Scaling 是**负贡献** (OOF 涨但 LB 跌)

**行动**:
- 按 `.planning/analysis_0.97092.md` 的 Phase 1-2 执行
- 跳过 Temperature Scaling 步骤
- 如果 Phase 1-2 成功 (LB > 0.970)，再考虑是否需要后处理

### 3. 如果必须尝试 Temperature Scaling，使用极简策略

**最小风险方案**:
- 不在 OOF 上拟合 T，而是使用**固定 T=0.9** (轻微 sharpen)
- 仅对 48h 应用 (因为 WBrier 权重最高 0.4)
- 提交前计算 Spearman(T_scaled, original)，如果 rho < 0.999 则放弃 (说明改变了排序)

**止损条件**:
- 如果 LB < 0.96783 - 0.002 = 0.96583，立即回退

### 4. 关注 Blend 权重而非校准

**发现**: 0.97092 和 0.97036 的主要差异在 24h blend 权重:
- 0.97092: W_GBSA_24=0.95 (更保守)
- 0.97036: W_GBSA_24=0.86 (更激进)

**建议**:
- 如果我们复现了 GBSA + LGB 双模型系统
- 在 24h 上优先使用 W_GBSA ≥ 0.90 (接近 0.97092)
- 在 48h 上使用 W_GBSA ≈ 0.55 (两者一致)
- 通过 OOF grid search 微调权重，但**不要**在 test 上调参

---

## 总结

### 方法论对比

| 方法 | 0.97092 | 0.97036 |
|------|---------|---------|
| **核心优势** | GBSA 50-model ensemble + 无校准 | GBSA + LGB + Temperature Scaling |
| **泛化性** | 强 (CV-LB gap=0.00399) | 弱 (CV-LB gap=0.00469) |
| **复现难度** | 中 (需要 50 models) | 中 (需要 Temperature Scaling) |
| **推荐度** | **高** (LB 更高，gap 更小) | 低 (校准是负贡献) |

### Temperature Scaling vs PowerCal

| 维度 | Temperature Scaling (T<1) | PowerCal (p^1.1) |
|------|---------------------------|------------------|
| **理论基础** | 强 (神经网络校准标准) | 弱 (ad-hoc) |
| **排序影响** | 无 (logit 空间单调) | 有 (非线性) |
| **参数数量** | 1 (T) | 1 (指数) |
| **过拟合风险** | 高 (221 样本) | 高 (221 样本) |
| **适用场景** | 大样本 + 验证集 | 探索性调参 |

### CV-LB Gap 启示

**核心教训**: 在 221 样本上，任何在 OOF 上拟合的校准参数都有高风险无法泛化到 test。

**证据链**:
1. 0.97036: Temperature Scaling, OOF +0.00014, LB -0.00056 (净损失 0.00070)
2. Exp32: Platt calibration, OOF +0.0059, LB -0.00445 (净损失 0.01035)
3. Exp33: Conformal quantile, OOF +0.0071, rho=1.0 (排序不变，高风险)

**行动指南**:
- ✅ 复现 0.97092 (无校准，泛化性强)
- ❌ 复现 0.97036 的 Temperature Scaling (已证明是负贡献)
- ✅ 关注 GBSA ensemble 规模 (50 models)
- ✅ 关注 Blend 权重 (W_GBSA_24 ≥ 0.90)
- ❌ 在 OOF 上拟合任何全局校准参数

---

## 附录: Temperature Scaling 数学推导

### 定义

给定原始概率 $p \in (0, 1)$ 和温度 $T > 0$:

$$
p_{\text{scaled}} = \sigma\left(\frac{\text{logit}(p)}{T}\right)
$$

其中:
- $\text{logit}(p) = \log\frac{p}{1-p}$
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ (sigmoid/expit)

### 性质

1. **单调性**: $\frac{d p_{\text{scaled}}}{d p} > 0$ (保持排序)
2. **边界**: $p_{\text{scaled}}(0) = 0$, $p_{\text{scaled}}(1) = 1$
3. **中点**: $p_{\text{scaled}}(0.5) = 0.5$ (不变点)
4. **T < 1**: 增大 logit → 概率向 0/1 推 (sharpen)
5. **T > 1**: 缩小 logit → 概率向 0.5 收缩 (soften)

### 为什么 0.97036 选择 T < 1?

**假设**: 原始 GBSA + LGB blend 的预测**过于保守** (概率偏向 0.5)

**目标**: 通过 T < 1 将概率向 0/1 推，增加模型置信度

**风险**: 如果原始预测已经校准良好，T < 1 会导致**过度自信** (overconfident)，在 test set 上增大 Brier loss

**实际结果**: OOF +0.00014 (轻微改善), LB -0.00056 (过度自信导致泛化失败)

---

**最终建议**: 立即启动 0.97092 复现 (Phase 1: GBSA 50-model ensemble)，跳过 Temperature Scaling。
