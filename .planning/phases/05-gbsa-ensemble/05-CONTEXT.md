# Phase 5: GBSA 50-Model Ensemble - Context

**Gathered:** 2026-02-28  **Status:** Closed (failed)
**Source:** Claude-Codex dialogue (2 rounds, autonomous mode)

<assumptions>
## Confirmed Assumptions
- GBSA boosting在N=221上结构性过拟合 — confirmed by: 3次实验(exp28/exp32/exp34)
- monkey-patch min_samples_leaf在fit前生效 — confirmed by: 代码审查(line 156, CV loop内)
- 50模型等权平均无二次选择偏差 — confirmed by: 代码审查
- sksurv 0.21 vs 0.22 GBSA核心逻辑无差异 — confirmed by: Codex查阅release notes
- dropout_rate=0.0在两个版本等价 — confirmed by: Codex [HIGH]

## Rejected Assumptions
- "复现0.97092只需GBSA 50-model ensemble" — reality: 特征工程与CV协议差异可能是关键
- "OOF≥0.970即可预期LB≥0.968" — reality: OOF-LB gap从0.010扩大到0.016
- "更多模型=更好泛化" — reality: 10→50模型仅+0.00025 OOF，LB未验证
</assumptions>

<domain>
## Phase Boundary
Phase 5 尝试复现0.97092公开方案的核心架构(GBSA 50-model ensemble)。实验证明GBSA boosting在N=221上结构性过拟合，OOF虚高(0.97257)但LB退化(0.96600)。三次独立实验(exp28/exp32/exp34)验证了同一结论：自训练boosting模型无法超越高质量锚点。Phase 5方向关闭，后续转向RSF中心化路线。
</domain>

<decisions>
## Implementation Decisions

### GBSA方向去留
| Option | Complexity | Performance | Maintainability | Risk |
|--------|-----------|-------------|-----------------|------|
| 继续优化GBSA | H | L (LB退化) | L | OOF-LB gap不可控 |
| 放弃GBSA回归RSF | L | M (稳定) | H | 已验证路线 |

- Decision: 放弃GBSA [HIGH共识]
- Claude: boosting结构性过拟合，三次验证失败
- Codex: 同意，补充sksurv版本不对齐可能加剧问题
- Evidence: exp28/exp32/exp34三次LB均低于锚点
- Resolution: Round 1即达成共识

### Phase 6 路线调整
| Option | Complexity | Performance | Maintainability | Risk |
|--------|-----------|-------------|-----------------|------|
| GBSA-based LGB IPCW | H | 未知 | L | 依赖失败的GBSA |
| RSF-based LGB IPCW | M | M | M | RSF已验证稳定 |

- Decision: Phase 6改为RSF-based LGB IPCW [HIGH共识]
- Claude: RSF survival function作为特征输入LGB
- Codex: 同意，建议X+{S_RSF, H_RSF, risk_RSF}作为LGB输入
- Resolution: Round 2达成共识

### 主线策略
- Decision: 回到锚点后处理优化 [HIGH共识]
- 校准优先级高(Brier权重0.7 > CI权重0.3)
- 可探索: isotonic/单调样条/temperature scaling
- 止损门槛: 固定预算内无稳定OOF提升即收敛回锚点

### Disagreements (Unresolved)
- 无重大分歧

### Claude's Discretion
- 具体校准方法选择(isotonic vs 单调样条 vs temperature)留给实现阶段决定
</decisions>

<risks>
## Pre-mortem Analysis

### Claude's Failure Predictions
1. RSF-based LGB IPCW同样过拟合 — likelihood: M, impact: H
   - N=221对任何per-horizon分类器都是挑战
2. 校准方法在hidden test分布偏移下失效 — likelihood: M, impact: M
   - Exp22已证明校准有天花板(lam=6.0峰值)
3. 0.96783已接近当前方法论极限 — likelihood: H, impact: H
   - 需要方法论代差(如0.98088的未知技术)才能突破

### Codex's Failure Predictions
1. sksurv版本不对齐导致隐性行为差异 — likelihood: M, impact: M
2. 特征工程差异是0.97092 gap的真正根因 — likelihood: H, impact: H
3. Phase 6 LGB IPCW在Brier分量上无增益 — likelihood: M, impact: M

### Blind Spots (only one side predicted)
- sksurv版本问题 — predicted by: Codex only (Claude未关注)
- 0.96783是方法论极限 — predicted by: Claude only (Codex更乐观)
</risks>

<specifics>
## Specific Ideas
- RSF survival function作为LGB特征: X + {S_RSF(t_k), H_RSF(t_k), risk_RSF}
- 低方差模型池: RSF保守参数、EST、Ridge-Cox、IPCRidge、参数化AFT(Weibull)
- 校准方法: isotonic、单调样条(PCHIP/I-spline)、temperature scaling
- 竞赛指标Hybrid=0.3*CI+0.7*WBrier，校准路线优先级高于排序路线
</specifics>

<deferred>
## Deferred Ideas
- sksurv 0.22.2环境下GBSA复现实验(仅在获取0.97092完整特征代码后考虑)
- GBSA+RSF混合ensemble(需先验证GBSA在子群体上的优势)
- 数据增强/半监督方法(超出当前竞赛框架)
</deferred>
