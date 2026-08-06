# 实验结果分析与路径重规划

## 已验证的失败方向 (Stop-Loss Triggered)

### 1. 模型侧改进 (Phase 1-2)
- **LR head**: LB=0.96274 (< 0.96331 baseline)
- **XGB head**: LB=0.95511 (灾难性过拟合)
- **IPCW stacking**: OOF=0.96108 (< gate 0.9697)
- **Calibration (Exp32)**: LB=0.96338 (OOF +0.0059 但 LB -0.00445)
- **Conformal (Exp33)**: OOF +0.0071 但 rho=1.0,与 Exp32 同模式

**教训**: 221 样本下,meta-learning 和 calibration 无法从 CV 迁移到 LB

### 2. 锚点复现/超参优化 (Phase 4)
- **PLE stacker fork**: LB=0.96086 (< 0.96624 gate)
- **RSF hyperparam grid**: LB=0.91089/0.90860 (分布压缩,pipeline 不一致)

**教训**:
- 无法通过 Kaggle fork 获得可靠的高质量锚点
- 超参网格需要 pipeline 完全一致才有效

## 当前最优配置

- **PB**: 0.96783 (Exp23 rh=1.1 rl=0.7 gate=[0.012,0.018])
- **锚点**: 0.96624 (RSF-only baseline)
- **Gap**: 0.00159 (锚点 → PB)

## 未充分探索的方向

### A. 后处理变换空间
- **非线性变换**: log/sqrt/power/sigmoid 变换预测值
- **排名优化**: 基于 Spearman rho 而非概率值的后处理
- **分位数映射**: 将预测分布映射到目标分布

### B. 集成策略
- **多后处理集成**: 集成不同后处理策略的结果
- **时间点权重优化**: 针对 WBrier 公式 (0.3×B@24h + 0.4×B@48h + 0.3×B@72h) 优化

### C. 数据增强
- **Bootstrap aggregating**: 在 221 样本上 bootstrap 生成多个后处理版本
- **Pseudo-labeling**: 使用 test set 的高置信度预测

### D. 约束优化
- **单调性约束**: 强制 12h ≤ 24h ≤ 48h ≤ 72h (已有,但可优化)
- **分布约束**: 匹配 training set 的经验分布

## 关键假设需要验证

1. **锚点质量假设**: 0.96624 是否已经是我们 pipeline 的上限?
2. **后处理空间假设**: 当前后处理 (Exp23 gate calibration) 是否已经是最优?
3. **CV-LB gap 假设**: 是否存在某种后处理策略能减小 gap?

## 建议的新 Phase 结构

### Phase 5: 后处理变换空间搜索
- 非线性变换 (log/sqrt/power)
- 排名优化 (rank-based postprocessing)
- 分位数映射

### Phase 6: 集成与约束优化
- 多后处理集成
- 时间点权重优化
- 分布约束优化

### Phase 7: 数据增强与伪标签
- Bootstrap aggregating
- Pseudo-labeling (如果 Phase 5-6 有效)

## 止损条件

- 任何新方向如果 LB < 0.96783 - 0.005 = 0.96283,立即止损
- 如果 Phase 5-6 全部失败,接受 0.96783 作为最终成绩
