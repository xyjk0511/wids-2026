# WiDS 2026 野火生存分析竞赛改进路线研究报告

## Executive Summary
本项目的“卡点”不是模型在训练集上的拟合能力，而是**泛化差距（CV-LB gap）异常偏大**与**12h 概率被系统性压缩导致 CI 受损**的叠加：Hybrid 里 CI 占 30%，而 CI 只看 prob_12h 排序；同时 WBrier 占 70%，但有效主要在 24h/48h（你们已观测到 72h 近似恒为 1）。竞赛方 Hybrid 公式与权重明确如此（0.3×C-index + 0.7×(1−Weighted Brier)）。citeturn23search0  
在仅 221 训练样本的设定里，**校准器/stacking/权重搜索**很容易“把 CV 当作真相”，从而在 Kaggle 公榜（仅用测试子集计分）上回撤。WiDS 赛程与依托组织信息也表明竞赛仍在进行阶段，公榜/私榜划分会显著放大选择偏差。citeturn26view0turn25search2  

优先行动（按收益/风险比排序）：
1) **版本回溯复现参考 0.96624**：把“sksurv 版本差异→StepFunction 边界/网格→12h 分布变化”从猜测变为可验证事实；  
2) **统一生存函数→概率的确定性转换**（避免 ValueError/域外处理差异引入系统偏移）；  
3) **定量分解 floor(0.01) 对 12h ties 与 CI 的贡献**，并将 floor 改为“数值 ε + 排序专用头”而非硬裁剪；  
4) **最小化 floor_recovery 重构 MVP**：排序层输出 + 极简 per-horizon 概率头（强正则）+ 单调投影（最小二乘），“先证伪再扩展”。  

## 问题诊断与关键假设
竞赛数据属于**右删失生存数据**：若 72 小时内到达疏散区则 event=1 且 time_to_hit_hours 观测到，否则 event=0 且时间在最后观测处删失。citeturn23search4turn24search1  
你们总结的核心症状可归并为四类：
- **CV-LB gap 偏大**：本质是模型选择偏差与分布不匹配（公榜只用测试子集计分更易放大）。citeturn25search2  
- **floor 吸附**（clip 到 0.01 形成大量 ties）→ 12h 排序信息被抹平；  
- **12h 概率压缩**：RSF/EST 输出是阶梯生存函数，S(t) 的取值规则与实现细节能导致 P(T≤12)=1−S(12) 大量接近 0；scikit-survival 的 StepFunction 定义为分段常数并按 \(x_i \le z < x_{i+1}\) 取值。citeturn0search1  
- **样本极小**：任何额外自由度（更多特征、复杂校准、stacking、权重调参）都可能只是在 CV 上“降噪过拟合”。此外，生存评估/校准一般隐含“独立删失”假设，而你们怀疑存在信息性删失，进一步降低泛化可靠性。citeturn13search3turn12search4  

关键未指定假设（不阻塞建议，但影响“复现参考”与“偏差归因”）：
- entity["organization","scikit-survival","python survival library"] 版本：未指定；  
- entity["organization","scikit-learn","python ml library"] / numpy / scipy 版本：未指定；  
- 是否可多环境并行（conda/docker）：未知（建议可选启用）。  
（可选输入：reference 环境 pip freeze、conda env.yml、原始 notebook 运行日志。）

## 优先验证项：复现与归因
复现与归因建议按“**先锁版本→再锁边界→最后锁分布**”执行，避免在 221 样本上做盲目大改。

**参考 0.96624 的复现策略（版本回溯矩阵 + 行为指纹）**  
动机：scikit-survival 在 0.21.0 有明确变更：`event_times_`→`unique_times_`，并且 `predict_survival_function` 返回的函数定义域/时间网格从“仅事件时刻”扩展为“训练中出现的全部唯一时间点（含删失）”，且对函数评价的异常行为做了修复。citeturn21view0  
步骤（最小可行）：  
- 建 5 个环境：sksurv {0.17.0, 0.20.0, 0.21.0, 0.23.0, 0.27.0}（其依赖约束见 release notes；0.27.0 依赖较新且含缺失值分裂修复）。citeturn17view0turn17view0  
- 固定同一组随机种子与同一折划分，输出“行为指纹”：  
  - `fn.x[-1]` 分布；  
  - `P12/P24/P48` 的 (min/median/max)、floor 占比、ties 数；  
  - 与参考提交分布的 KS 距离/分位差（不用很精细，先能区分“系统偏移”即可）。  
风险：环境构建耗时；回滚：保留当前环境与 pipeline，仅新增“对照跑”脚本。

**StepFunction 边界处理对比测试（定位 12h 概率偏移来源）**  
依据：旧版 StepFunction 在 `x < x[0] or x > x[-1]` 会抛 ValueError；新版引入 domain 并对域下界做 clip 处理。citeturn10view0turn19view0 同时，RSF 预测在某些版本/场景会出现“看似在区间内评价却报域错误”的已知案例。citeturn11view0  
测试：对每个样本的 `surv_fn`，评估 `fn(12)` 是否触发异常、触发比例是否随版本变化；再对比你们使用的 `fn.x[-1]` 兜底与 try/except 兜底的差异。

**量化 floor 对 12h median 与 CI 的贡献（消融设计）**  
实验：仅改后处理，不改模型：floor ∈ {0, 1e−6, 1e−4, 1e−3, 1e−2, 1e−2(仅24/48), 0.01}；每种都记录：  
- `ties@12h`（唯一值个数/样本数）、`median(prob_12h)`、CI、WBrier、Hybrid。  
判定：若 CI 对 ties 高敏感，则应看到“floor 下降→ties↓→CI↑”的单调趋势；若无趋势，说明根因更多在“prob_12h 排序信号”而非裁剪。

## 优先改进方向与实验清单
下面给出至少 8 项、按优先级排序的可执行路线（每项都给 MVP、风险与回滚、以及小样本稳定性检验）。

1) **版本回溯复现参考（最高优先）**  
动机：你们的关键差异指向“StepFunction 评价/网格”而非模型结构；0.21 以后 scikit-survival 明确改变了时间网格与函数评价行为。citeturn21view0  
实现：`scripts/env_matrix_run.py`，循环 conda env → 复跑同一 RSF 配置 → 导出分布与 OOF 指标。  
依赖：conda/mamba，可选 docker。  
预期：CI↑、Hybrid↑（若 12h 分布回到参考双峰）。  
风险：时间成本；回滚：仅保留“匹配到的版本”用于最终训练。  
验证：以“分布指纹 + OOF Hybrid 匹配度”作为通过标准（见 §度量）。

2) **生存函数→horizon 概率的“确定性转换层”（锁定边界与域外规则）**  
动机：StepFunction 在不同版本对域上下界处理不同（旧版域外抛错，新版有 domain 并 clip）。citeturn10view0turn19view0turn0search1  
实现步骤：新增 `src/surv_post.py::sf_to_cdf(fn, h, policy)`：  
- 若 `hasattr(fn,"domain")`：`h_clip = clip(h, fn.domain[0], fn.domain[1])`；否则用 `[fn.x[0], fn.x[-1]]`；  
- 统一返回 `p=1-fn(h_clip)`；禁止 try/except 分支散落。  
依赖：无新增。  
预期：缩小“系统性 12h 偏移”，CI/WBrier 均可能更稳。  
风险：若规则选错会整体偏移；回滚：保留旧实现并 A/B。  
验证：同一模型下，比较 `policy` 的分布差异与 Hybrid（bootstrap CI）。

3) **把 floor 从“硬裁剪”改为“数值 ε + 显式 ties 管控”**  
动机：硬 floor=0.01 会制造大量 12h ties，直接伤 CI；而 CI 只看排序。citeturn23search0  
实现：  
- `monotonic.py`: `floor=1e-6`（仅防数值）；  
- 12h：不再 `clip` 到 0.01；而是用排序头产生“可分辨概率”（见下一条）。  
依赖：无。  
预期：CI↑；WBrier ≈（若 24/48 仍维持合理 floor）。  
风险：极小概率导致数值/提交格式问题；回滚：对 24/48 保留 0.01，12h 用 `min(p12, p24)` 保单调。  
验证：消融（§优先验证项第三条）+ 折内 ties 与 CI 的相关系数。

4) **12h 排序专用头：用风险分数替代 1−S(12) 做排序信号**  
动机：RSF/EST 的 `predict()` 风险分数来自累积风险估计，天然更“连续”，可作为 tie-break；而 RSF 生存函数在 t=12 前事件稀少时会导致 1−S(12) 接近 0。RSF 的生存函数在叶节点由 Kaplan–Meier 估计并跨树取平均。citeturn22view0turn0search1  
实现（保证单调）：  
- 生成 `risk = model.predict(X)`；  
- `p12 = p24 * sigmoid(a*(risk-b))`（天然 ≤ p24）；a,b 用 OOF 只最大化 CI（不看 Brier）。  
依赖：`scipy.optimize` 或闭式网格搜索。  
预期：CI↑，Hybrid↑；WBrier 基本不变（p24 由原管线决定）。  
风险：若 p24 太小会压缩 p12；回滚：改为 `p12=min(p12_raw, p24)`。  
验证：MVP 只在 OOF 上做（不改 full retrain）；用 bootstrap 评估 ∆CI 的 95% 置信区间。

5) **用“最小二乘单调投影”替代 `max-accumulate` 单调化**  
动机：`max()` 的单调化会单边抬高后续 horizon，可能伤 24/48 校准；而等长序列的 isotonic/PAVA 投影能在“最小改动”下满足单调（更利于 Brier）。  
实现：`src/monotonic.py::project_monotone_l2(p12,p24,p48,1.0, weights)`（4 点 PAVA）。  
依赖：可自写 PAVA，无需额外库。  
预期：WBrier↓（尤其 24/48），Hybrid↑。  
风险：实现 bug；回滚：保留现有两种策略并 A/B。  
验证：只替换后处理，其他不动；对每折记录 WBrier 分解（B24/B48）。

6) **特征集“稳定性筛选”以缩小 gap（少即是多）**  
动机：0.21 以后很多库更严格；但根本仍是 221 样本下维度过高导致选择偏差。  
实现：`src/features.py` 增加 `stability_filter(features, oof_importance)`：  
- 在 50-fold OOF 上计算 permutation importance 分布；  
- 仅保留“重要性为正且跨折符号一致率≥0.8”的特征；工程特征加 L2 shrink（可做线性头）。  
依赖：sklearn permutation_importance。  
预期：CV-LB gap↓、LB↑（更稳）。  
风险：误删有效特征；回滚：回到 MEDIUM 或 v96624。  
验证：Pseudo-LB（下一条）优先用来做模型选择，而不是纯 OOF。

7) **Pseudo-LB 验证：用训练集模拟“公榜子集计分”以抑制选择偏差**  
动机：该竞赛公榜仅用测试集约 54% 计分，模型在 CV 上的小优势很可能是噪声。citeturn25search2  
实现：`experiments/exp_public_lb_emulation.py`：  
- 重复 R 次：随机抽训练样本的 54% 作为 pseudo-public，剩余作为 pseudo-private；  
- 对每个候选方案输出 `Hybrid_public` 与 `Hybrid_private` 的均值/方差与差值。  
依赖：无。  
预期：筛掉“public 冒尖、private 崩”的方案 → 最终真实 LB 更稳。  
风险：训练集分布≠测试集；回滚：仍以 OOF 为主，只把 pseudo-LB 作为“拒绝规则”。  
验证：选择标准改为“在 ≥70% 重复中 public 与 private 都不降”。

8) **floor_recovery_reform_plan 的 MVP：排序层 + 极简 per-horizon heads + 单调投影**  
动机：竞赛的 Brier 是按 horizon 的二分类误差衡量（且对删失样本有资格规则），从“离散风险/发生概率”角度建模更贴合；time-dependent Brier 在文献里通常需处理删失（如 IPCW），说明“评估-训练对齐”很关键。citeturn12search16turn12search4turn23search0  
MVP 实现（尽量小模型）：  
- 排序层：只用 RSF+EST（你们当前最佳组合）；输出 `risk` 与 `p24_base,p48_base`；  
- 概率头：只用 LogisticRegression(L2, C=0.1) 三个头预测条件概率 `q12,q24,q48`，输入仅 `[log_dist, eta_hours, risk]`（≤3~5维）；  
- 组合：`p12=q12; p24=p12+(1-p12)*q24; p48=p24+(1-p24)*q48; p72=1`；  
- 校准：只允许 `sigmoid`（严格单调、保排序；且相对 isotonic 更不易在小样本过拟合）。citeturn0search1turn0search6  
预期：WBrier↓（24/48），同时 12h 分布更可控；Hybrid↑。  
风险：仍可能过拟合；回滚：只保留“排序层 + 12h 头”（不动 24/48）。  
验证：双层交叉拟合（cross-fitting）+ bootstrap 置信区间；若 ∆Hybrid 的 95%CI 不为正则不进 full retrain。

9) **缺失值/稀疏观测处理一致化（版本敏感项）**  
动机：scikit-survival 0.27 明确修复了 RSF/SurvivalTree 在 log-rank 分裂下对缺失值的处理逻辑。若你们特征存在 NaN/稀疏模式，版本差异会直接改变分裂与生存函数。citeturn17view0  
实现：统一在 `train.py` 前置 `SimpleImputer(strategy="median")` + 缺失指示特征；并冻结该流程。  
预期：gap↓、复现性↑。  
风险：对某些模型退化；回滚：仅对树模型启用缺失指示，保持原值。  
验证：在版本矩阵下比较“同版本不同机器”的方差。

## 推荐系统重构架构与图表设计
推荐重构的核心是把“**排序信号**”与“**概率/校准**”解耦，并让单调性成为“投影层”而不是到处的 if/max。

```mermaid
flowchart LR
  A[原始特征X] --> B[排序层: RSF/EST/XGB-Cox 输出 risk & base probs]
  B --> C[概率头: per-horizon heads<br/>q12,q24,q48 (强正则)]
  C --> D[校准层: sigmoid/温度缩放(可选)]
  D --> E[单调投影: PAVA/L2 projection<br/>p12<=p24<=p48<=p72]
  E --> F[submission: prob_12h/24h/48h/72h]
```

图表建议（用于定位而非“好看”）：
- **Probability distribution 对比图**：分别画 12/24/48 的直方图或 KDE，标注 `median`、`floor占比`、`ties率`；并在“参考 vs 当前 vs 改动”三者间对齐同一 y 轴。StepFunction 的取值规则与边界会改变分布形状，需可视化验证。citeturn0search1turn21view0  
- **CV-LB gap 时间线**：x=实验编号/日期；y1=Local CV Hybrid（带 bootstrap 误差条），y2=Kaggle LB；同时画 `gap=y1−y2`（你们已记录到 gap 量级差异）。公榜只用测试子集计分这一事实要作为解释背景。citeturn25search2  

## 对照表：当前 vs 参考 vs 建议改动
| 方案 | 模型 | 特征数 | sksurv版本假设 | full retrain树数 | floor策略 | CV配置 | 后处理策略 | 预期对LB影响 |
|---|---|---:|---|---:|---|---|---|---|
| 当前最佳（#9） | RSF+EST(10模型)+权重/可选头 | 36 | 未指定 | 200×5seed×2 | clip至0.01 | 5×10重复分层 | max链/分裂链 | 主要风险在gap与12h压缩 |
| 参考目标（0.96624） | RSF单模型 | 16 | 未指定（重点回溯≤0.20 vs ≥0.21）citeturn21view0 | 200 | clip至0.01 | 5×10重复分层 | 单调+72h=1 | 目标：gap更小、12h分布更分离 |
| 建议主线（优先） | 版本锁定RSF/EST + 12h排序头 + L2单调投影 + MVP heads | 16~20 | 显式固定/可复现矩阵 | 200~400（需验证） | 12h用ε，24/48按需floor | OOF + Pseudo-LB筛选citeturn25search2 | PAVA投影+72h=1 | 方向：CI↑且WBrier不恶化，优先缩小gap |

## 度量、稳定性检验与回滚准则
度量与检验建议统一成“低样本可承受”的标准流程：
- **主度量**：严格复刻竞赛 Hybrid（CI 用 prob_12h；WBrier 用 24/48/72 权重）。citeturn23search0  
- **稳定性**：对任意改动，报告三件事：  
  1) 50-fold OOF 的 ∆Hybrid 均值；  
  2) bootstrap(≥200) 的 ∆Hybrid 95%CI（若 CI 跨 0 直接拒绝）；  
  3) Pseudo-LB（54%/46%）重复分割下 public 与 private 的双稳健性（至少 70% 重复不下降）。citeturn25search2  
- **回滚准则**：若出现“OOF↑但 pseudo-private↓”或“ties@12h↑”或“p12>p24 纠正幅度大（>5%样本需强制修正）”，优先回滚到上一个稳定版本。  
- **校准慎用**：若必须校准，优先 `sigmoid`；isotonic 在样本≪1000 时不推荐且容易引入 ties，影响排序度量。citeturn0search1turn0search6  

## 结论
在 221 样本、Hybrid 指标、且 12h 仅影响排序的结构下，最有效的路径是：**先把“参考实现的系统性差异”用版本/边界测试锁死**（避免凭感觉调参），再用**12h 排序专用头**解决 ties 与排序退化，同时用**最小二乘单调投影**守住 24/48 的 Brier；最后才上 **floor_recovery 的 MVP**，以极低自由度验证“解耦排序与概率”是否真能缩小 CV-LB gap 并逼近参考 LB。citeturn21view0turn22view0turn23search0