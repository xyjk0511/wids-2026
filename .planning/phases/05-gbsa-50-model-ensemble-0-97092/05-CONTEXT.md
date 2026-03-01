# Phase 5: GBSA 50-Model Ensemble (复现 0.97092 核心) - Context

**Gathered:** 2026-03-01  **Status:** Ready for planning
**Source:** Claude-Codex dialogue (2 rounds) + 0.97092 notebook ground truth

<assumptions>
## Confirmed Assumptions
- GBSA 5 configs 超参与 0.97092 完全一致 — confirmed by: notebook 代码
- dropout_rate=0.0 是正确的 — confirmed by: notebook 代码
- 用户能访问完整 notebook 代码 — confirmed by: 用户直接提供

## Rejected Assumptions
- exp34 特征集与 0.97092 一致 — reality: 0.97092 用 30+ 距离导向特征，exp34 用 medium 36+6 augmented，完全不同
- exp34 CV 策略与 0.97092 一致 — reality: 0.97092 用 5-fold StratifiedKFold，exp34 用 10-fold KFold
- isotonic calibration 是正确的后处理 — reality: 0.97092 不做任何校准，直接用原始输出
- GBSA 单独就能达到 0.97092 — reality: 需要 GBSA + LightGBM IPCW blend + 5km cutoff + sigmoid 72h
</assumptions>

<domain>
## Phase Boundary

Phase 5 的目标从"GBSA 50-model ensemble"扩展为"精确复现 0.97092 的完整架构"。0.97092 不是单纯的 GBSA ensemble，而是一个多组件系统：GBSA CV-bag (375 fold models) + LightGBM IPCW per-horizon (200 fold models) + asymmetric blend + hard 5km cutoff + sigmoid 72h。Exp34 的 LB 0.96600 低于 PB 的根因不是 GBSA 本身，而是缺少 LGB IPCW 组件、特征工程不匹配、CV 策略不同、以及错误地使用了 isotonic calibration。
</domain>

<decisions>
## Implementation Decisions

### 特征工程
| Option | Complexity | Performance | Maintainability | Risk |
|--------|-----------|-------------|-----------------|------|
| 精确复制 0.97092 create_features | L | H | H | 低 — 已有完整代码 |
| 保留 exp34 medium+augmented | L | L | M | 高 — 已证明 LB 不佳 |

- Decision: 精确复制 0.97092 的 create_features() [HIGH]
- Claude: 必须对齐特征集，这是最大差异源
- Codex: 无法确认（Round 1 无代码）
- Evidence: 0.97092 notebook 完整代码
- Resolution: Ground truth 确认

### CV 策略
| Option | Complexity | Performance | Maintainability | Risk |
|--------|-----------|-------------|-----------------|------|
| 5-fold StratifiedKFold (0.97092) | L | H | H | 低 |
| 10-fold KFold (exp34) | L | L | M | 高 — OOF 偏差 |

- Decision: 使用 5-fold StratifiedKFold，random_state=seed [HIGH]
- Evidence: 0.97092 notebook 代码

### 后处理
| Option | Complexity | Performance | Maintainability | Risk |
|--------|-----------|-------------|-----------------|------|
| 无校准 + 5km cutoff + sigmoid 72h | M | H | H | 低 |
| Isotonic calibration | L | L | M | 高 — 221样本过拟合 |

- Decision: 不做校准，直接用原始输出 + hard 5km cutoff + sigmoid 72h [HIGH]
- Evidence: 0.97092 不做任何校准，isotonic 在 exp34 导致 gap 扩大
</decisions>

<risks>
## Pre-mortem Analysis

### Claude's Early Assessment (Round 1)
1. OOF-LB gap 无法修复 — likelihood: H, impact: H
   → Mitigation: 精确复现 0.97092 消除实现差异
2. 特征集/预处理不对齐导致复现失败 — likelihood: M, impact: H
   → Mitigation: 逐行复制 create_features()
3. 提交预算耗尽但方向未收敛 — likelihood: M, impact: M
   → Mitigation: 用户同意 3 次诊断提交

### Ground Truth 风险更新 (Round 2)
1. GBSA 训练时间长 (375 models) — likelihood: H, impact: M
   → Mitigation: 先用 2 configs × 5 seeds 快速验证
2. LGB IPCW 实现复杂度 — likelihood: M, impact: M
   → Mitigation: 直接复制 notebook 的 IPCW 代码
3. 5km cutoff 过于激进 — likelihood: L, impact: H
   → Mitigation: 训练集 100% 验证 (dist>=5km 全部 event=0)

### Codex Pre-mortem
Not run (dialogue converged after用户提供 ground truth)
</risks>

<specifics>
## Specific Ideas

### 0.97092 完整架构复现清单
1. **create_features()** — 精确复制 30+ 距离导向特征工程
2. **GBSA CV-bag** — 5 configs × 15 seeds × 5-fold StratifiedKFold = 375 fold models
3. **LGB IPCW** — 24h/48h 独立分类器，20 seeds × 5-fold = 200 fold models
4. **Blend** — W24=0.97, W48=0.48
5. **Hard 5km cutoff** — dist >= 5000 → 全0
6. **Sigmoid 72h** — sigmoid(dist, 5450, 50)
7. **无校准** — 原始输出直接提交

### 快速验证策略 (3 次诊断提交)
- 提交 1: 精确复现 0.97092 (全量) → 验证是否达到 0.970+
- 提交 2: 去掉 LGB IPCW (纯 GBSA + cutoff) → 量化 LGB 贡献
- 提交 3: GBSA + RSF blend → 测试多样性增益
</specifics>

<deferred>
## Deferred Ideas
- Phase 6 IPCW LightGBM 已包含在 0.97092 复现中，可合并到 Phase 5
- Phase 7 增量改进 (PowerCal24, 15 seeds) 在复现成功后再考虑
- GBSA + RSF blend 作为超越 0.97092 的增量方向
</deferred>
