# Phase 5 - Claude x Codex Dialogue

## Round 1
### Claude's Questions
Q1: 0.97092 notebook 的后处理用的是 isotonic calibration 还是其他方法？
Q2: GBSA 和 RSF 在 prob_12h 分布上有什么理论差异？
Q3: exp32 用 10-fold KFold，0.97092 用什么 CV 策略？
Q4: 0.97092 notebook 的特征工程和 exp32 augment_features 一致吗？
Q5: 剩余 18+ 次提交，Phase 5 GBSA 方向应该分配多少次提交预算？
Q6: 是否应该先做 GBSA vs RSF 的分布对比诊断？
Q7: Exp34 GBSA OOF 0.97257 但 LB 0.96600，LB 下降的主要原因？
Q8: ROADMAP 中的 5 个 GBSA 超参配置是否合理？
Q9: 鉴于 Exp34 LB 不佳，Phase 5 应该怎么走？
Q10: (用户已回答) 能否访问 0.97092 完整 notebook 代码？

### User's Direct Answers
- Q10: 能访问完整代码
- Q1-Q9: 交给 Codex 回答

### Codex's Answers
A1: [LOW][NO_EVIDENCE] 无法确认 0.97092 是否用 isotonic calibration，需检查 notebook 代码
A2: [MED][NO_EVIDENCE] GBSA prob_12h 更连续平滑，RSF 更阶梯化；12h 对 CI 敏感时影响排序稳定性
A3: [LOW][NO_EVIDENCE] 无法确认 0.97092 的 CV 策略，KFold vs StratifiedKFold 差异足以改变 OOF 偏差
A4: [LOW][NO_EVIDENCE] 无法确认特征工程是否一致，需逐列对比
A5: [MED] 建议 6 次诊断 + 8 次融合扫描 + 4 次冲刺
A6: [HIGH] 应该先做分布对比诊断，OOF-LB gap=0.016 是验证失配信号
A7: [MED] 主要是验证估计偏乐观 + 分布/校准失配，不是模型完全无信号
A8: [LOW][NO_EVIDENCE] 建议改为结构化覆盖：低LR高树数、基准组、高LR低树数、强正则、随机子采样
A9: [MED] 不放弃 GBSA，走 GBSA 作为弱权重多样性分量 + RSF/EST 主体的融合路线

### Codex's Counter-Questions
CQ1: 能否贴出 0.97092 notebook 的完整后处理代码（isotonic/logit/clip/rank）？
CQ2: 0.97092 的 CV 是 KFold/StratifiedKFold/RepeatedKFold 还是自定义？random_state 和 shuffle 怎么设？
CQ3: 0.97092 是否对 event/censor 比例做了分层？
CQ4: exp32 的 6 个 engineered features 在 0.97092 里是否一一存在？
CQ5: Exp34 后处理是否与 RSF+EST PB 方案完全一致？
CQ6: 能否提供 GBSA 与 RSF 的 prob_12h 在 OOF 和 test 的分位数表？
CQ7: 是否做过按 horizon 拆分的指标归因？
CQ8: 50 模型 ensemble 是等权平均吗？有没有按 fold 稳定性加权？
CQ9: exp34 与 PB 的特征预处理是否完全对齐？
CQ10: 愿不愿意先用 3 次提交做诊断型小实验来定位主因？

### User's Direct Answers to Codex
- CQ1: 能提供 0.97092 后处理代码
- CQ10: 同意用 3 次提交做诊断型小实验
- CQ2-CQ9: 交给 Claude 回答

## Round 2 (Ground Truth — 用户提供 0.97092 完整代码)

### 用户提供的 0.97092 Notebook 关键发现

**特征工程** (回答 CQ4):
- 0.97092 用 create_features() 生成 30+ 距离导向特征
- 包括: log_distance, inv_distance, inv_distance_sq, sqrt_distance, dist_km, dist_km_sq, dist_rank, radius_to_dist, area_to_dist_ratio, eta_hours, log_eta, effective_closing_speed, eta_effective, threat_score, fire_urgency, growth_intensity, zone_critical/warning/safe, is_summer, is_afternoon
- 删除 6 列: relative_growth_0_5h, projected_advance_m, centroid_displacement_m, centroid_speed_m_per_h, closing_speed_abs_m_per_h, area_growth_abs_0_5h
- 与 exp32 的 augment_features 完全不同

**CV 策略** (回答 CQ2, CQ3):
- StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
- 按 event 分层，每个 seed 不同的 random_state
- exp32 用 KFold(10) 不分层 — 关键差异

**后处理** (回答 CQ1, CQ5):
- 无 isotonic calibration，无 Platt scaling
- 原始 GBSA 输出直接使用
- prob_72h = sigmoid(dist, threshold=5450, scale=50)
- Hard 5km cutoff: dist >= 5000 → 全部概率设为 0

**架构**:
- CV-bag: 每个 fold 模型预测 test，跨 fold 平均
- 5 configs × 15 seeds × 5 folds = 375 GBSA fold models
- LightGBM IPCW: 24h/48h 独立，20 seeds × 5 folds = 200 LGB models
- Blend: W24=0.97 (GBSA主导), W48=0.48 (平衡)

**GBSA 超参** (回答 Q8):
- 5 configs 与 ROADMAP 完全一致
- dropout_rate=0.0, max_depth=3, min_samples_split=3

## Convergence
Stopped after 2 rounds. Reason: 用户提供 0.97092 完整 notebook 代码，所有 [NO_EVIDENCE] 问题获得 ground truth 答案，对话自然收敛。
