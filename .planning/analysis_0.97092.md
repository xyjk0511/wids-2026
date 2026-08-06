# 0.97092 Notebook vs 0.96783 Current Method - Critical Gap Analysis

**Date**: 2026-02-28
**Current PB**: 0.96783 (Exp23 gate calibration)
**Target**: 0.97092 (OOF 0.97491)
**Gap**: +0.00309 LB improvement needed

---

## 关键差异 1: GBSA Multi-Config Ensemble (50 models)

**技术**:
- 5 个不同超参配置 × 10 个随机种子 = 50 个 GBSA 模型
- 配置差异维度: `learning_rate` (0.005–0.01), `subsample` (0.6–0.85), `min_samples_leaf` (12–20), `n_estimators` (1200–2000)
- 全部 `max_depth=3`, `dropout_rate=0.0` (无 dropout)
- **我们当前**: 单一 RSF 配置 × 5 seeds = 5 models (+ 5 EST in exp#9)

**原因**:
- **方差缩减**: 50 个模型的平均比 5–10 个模型更稳定，减少单次预测的随机波动
- **超参多样性**: 不同 `subsample` (0.6/0.7/0.85) 和 `min_samples_leaf` (12/15/20) 捕捉不同的数据子空间模式
- **GBSA 优势**: Gradient Boosting Survival Analysis 直接优化生存函数，比 RSF 的随机森林更精细地拟合时间依赖关系
- **OOF 证据**: 0.97491 OOF hybrid 远超我们的 0.9721 (RSF+EST)，说明 GBSA ensemble 的信号质量显著更高

**预期提升**: +0.002–0.004
**优先级**: **高** (最大单项差异)

---

## 关键差异 2: LightGBM Per-Horizon IPCW 分类头

**技术**:
- 24h 和 48h 各自独立训练 LightGBM 分类器
- 使用 IPCW (Inverse Probability of Censoring Weighting) 处理删失样本
- 24h: `max_depth=3, lr=0.03, n_est=300, subsample=0.7, reg_alpha=0.5, reg_lambda=2.0`
- 48h: `max_depth=2, lr=0.05, n_est=200, subsample=0.8, reg_alpha=0.1, reg_lambda=1.0`
- **我们当前**: 无独立分类头，直接使用 RSF 的生存函数插值

**原因**:
- **IPCW 优势**: 正确处理删失样本的信息损失，避免 naive 标签构建的偏差
- **Per-horizon 优化**: 24h 和 48h 的最优超参不同 (24h 更深/更正则化，48h 更浅/更快学习)，说明两个时间点的决策边界特性不同
- **GBM 判别力**: LightGBM 在 221 样本上比 RSF 插值更灵活，能捕捉非线性交互
- **Blend 权重差异**: W_GBSA_24=0.95 (几乎全用 GBSA), W_GBSA_48=0.55 (GBSA 和 LGB 各半)，说明 LGB 在 48h 上贡献显著

**预期提升**: +0.001–0.002
**优先级**: **高** (IPCW 是我们 Exp31 失败的根因，但他们成功了)

---

## 关键差异 3: Blend 权重策略 (24h vs 48h 不对称)

**技术**:
- 24h: `W_GBSA=0.95`, LGB 仅占 5%
- 48h: `W_GBSA=0.55`, LGB 占 45%
- **我们当前**: 无 per-horizon blend，RSF 单一模型或 uniform ensemble

**原因**:
- **时间依赖的模型优势**: GBSA 在短期 (24h) 预测上更可靠 (可能因为样本量更大)，LGB 在长期 (48h) 上补充信号
- **方差-偏差权衡**: 24h 高权重 GBSA (低方差)，48h 平衡 GBSA+LGB (降低偏差)
- **隐式校准**: 不对称权重相当于对 24h/48h 施加不同的"信任度"，可能缓解了我们在 Exp22 中发现的 p48 偏高问题

**预期提升**: +0.0005–0.001
**优先级**: **中** (需要先有 GBSA+LGB 两个模型才能测试)

---

## 关键差异 4: 无显式后处理校准

**技术**:
- Notebook 代码中**没有** logit 空间校准、gate calibration、或单调性修复
- 直接输出 blend 结果
- **我们当前**: Exp22/Exp23 复杂的 logit 校准 + gap-gated push

**原因**:
- **模型质量足够高**: OOF 0.97491 说明 GBSA ensemble 的原始输出已经接近最优分布，不需要后处理"修补"
- **避免过拟合**: 我们的 Exp32/Exp33 显示，在 221 样本上校准容易过拟合 (OOF 涨但 LB 跌)
- **简单即正义**: 无后处理意味着更少的超参搜索空间，泛化性更好

**预期提升**: 0 (我们的后处理可能是**负贡献**)
**优先级**: **低** (需要先复现 GBSA ensemble，再验证是否需要后处理)

---

## 关键差异 5: 特征集未知

**技术**:
- Notebook 未公开特征工程代码
- 可能使用与我们不同的特征集 (我们是 v96624 16 特征或 MEDIUM 36 特征)
- **我们当前**: v96624 (16) 或 MEDIUM (36)

**原因**:
- **特征质量**: 如果他们有更强的特征 (如时间序列衍生、交叉特征)，可以提升模型上限
- **不确定性高**: 无法从代码推断，可能是标准特征 + 简单工程

**预期提升**: 未知 (0–0.002)
**优先级**: **低** (先复现模型架构，特征工程是次要因素)

---

## 复现优先级建议

### Phase 1: GBSA Multi-Config Ensemble (必做)
1. 安装 `scikit-survival` 0.22+ (GBSA 支持)
2. 实现 5 configs × 10 seeds = 50 models
3. 验证 OOF hybrid > 0.970 (门槛)
4. 提交验证 LB > 0.968

**预期时间**: 2–3 小时 (训练 + 验证)
**成功标准**: OOF hybrid ≥ 0.970, LB > 0.968

### Phase 2: LightGBM IPCW Per-Horizon (高优先级)
1. 实现 IPCW 权重计算 (Kaplan-Meier censoring estimator)
2. 训练 24h/48h 独立 LGB 分类器
3. Blend GBSA (from Phase 1) + LGB with W_24=0.95, W_48=0.55
4. 提交验证 LB > 0.970

**预期时间**: 1–2 小时
**成功标准**: LB ≥ 0.970

### Phase 3: 移除后处理 (验证性)
1. 使用 Phase 2 的 blend 结果，**不做** Exp22/Exp23 校准
2. 对比 LB 分数
3. 如果无后处理 > 有后处理，确认我们的校准是负贡献

**预期时间**: 30 分钟
**成功标准**: 确认后处理的净收益 (可能为负)

---

## 风险与止损

### 风险 1: GBSA 训练时间过长
- **缓解**: 先用 2 configs × 5 seeds = 10 models 快速验证
- **止损**: 如果 10 models OOF < 0.965，停止 GBSA 方向

### 风险 2: IPCW 实现错误
- **缓解**: 使用 `lifelines.KaplanMeierFitter` 标准库
- **止损**: 如果 IPCW LGB OOF < naive LGB，回退 naive 标签

### 风险 3: 复现失败 (LB < 0.968)
- **止损**: 如果 Phase 1 LB < 0.968，说明 notebook 有隐藏技巧 (特征/数据增强)，切换到 Phase 4 (更强锚点搜索)

---

## 总结

**最关键的 3 个差异** (按重要性):
1. **GBSA 50-model ensemble** (+0.002–0.004) ← 最大收益
2. **LightGBM IPCW per-horizon** (+0.001–0.002) ← 技术正确性
3. **Per-horizon blend 权重** (+0.0005–0.001) ← 精细调优

**累计预期提升**: +0.0035–0.007 (理论上可达 0.970–0.973)

**行动**: 立即启动 Phase 1 (GBSA ensemble)，这是唯一可能突破 0.970 的路线。
