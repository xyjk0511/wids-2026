# Phase 5: GBSA 50-Model Ensemble — Context & Post-Mortem

## Claude-Codex 自主对话结论 (2026-02-28)

### 实验结果

| 指标 | 值 | 对比 |
|------|-----|------|
| OOF Hybrid | 0.97257 | +0.00474 vs PB |
| Kaggle LB | 0.96600 | -0.00183 vs PB |
| OOF-LB Gap | 0.01657 | +60% vs RSF典型gap |
| Gate | PASS (≥0.970) | OOF通过但LB失败 |

### 失败根因分析（按概率排序）

1. **GBSA boosting 结构性过拟合 (~70%)**
   - 1200-2000棵sequential trees在N=221上有效参数量远超样本量
   - RSF(bagging)天然抗过拟合，GBSA(boosting)不行
   - exp32(LB 0.96338)和exp34(LB 0.96600)两次验证同一模式

2. **特征集膨胀加剧过拟合 (~60%，与#1叠加)**
   - medium=36 features，GBSA每棵树看全部特征
   - RSF有max_features=sqrt随机选子集，GBSA没有
   - 0.97092原始方案的特征集未公开，可能更精简

3. **min_samples_leaf monkey-patch语义不足 (~30%)**
   - 技术上生效，但min_samples_split=16 hardcode与msl=20冲突
   - 不是致命问题，但导致树结构与原始方案有差异

4. **缺少LightGBM IPCW per-horizon头 (~25%)**
   - 0.97092还有LGB分类器(W_GBSA_48=0.55)
   - 但这解释的是0.97092 vs 0.96783的gap，不是exp34 < PB

5. **无后处理对比缺失 (~15%)**
   - exp34是裸提交(无Exp22校准)
   - 0.96600 vs 0.96624(锚点)仅差-0.00024
   - 说明GBSA排序能力≈锚点，但未超越

### 决策：有条件放弃 GBSA

**理由**:
- 三次实验(exp28/exp32/exp34)验证：自训练模型在N=221上无法超越锚点
- OOF高估严重且不可修复（boosting的结构性问题）
- 盲目复现0.97092的ROI极低（缺少关键信息：特征集、sksurv版本行为）

**唯一继续条件**: 获取0.97092的完整特征工程代码

### 替代路线（推荐）

1. **锚点搜索** — 寻找LB>0.968的公开submission作为新锚点
2. **Exp22最后一英里** — gate参数细粒度grid search（预期+0.00001~0.00003）
3. **12h排序增强** — tie-breaking改善CI分量（上限~0.0005）
4. **等待赛末公开方案** — 从高分方案提取可复现技术

### 关键教训

> 在N=221的生存分析竞赛中，bagging(RSF) >> boosting(GBSA)。
> OOF高分是陷阱，LB才是真相。
> 高质量锚点 + 精细后处理 > 自训练复杂模型。
