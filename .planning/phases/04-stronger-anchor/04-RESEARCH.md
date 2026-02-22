# Phase 4: 更强锚点获取/复现 - Research

**Researched:** 2026-02-22
**Domain:** Survival analysis pipeline reproducibility, sksurv version behavior, rank-based ensemble blending
**Confidence:** MEDIUM

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
1. 三条 Track 按顺序执行，Track 1 是前置条件
2. 不做花式改动：精确复现优先，不加特征/不改模型结构
3. 分阶段目标: LB > 0.9685 → LB > 0.970 → 0.975 (stretch)
4. 提交门槛: 仅允许改变排序且有稳定性证据的方案

### Claude's Discretion
- Track 1 内的具体复现策略（sksurv 版本矩阵、超参网格范围）
- Track 2 的 blend weight 搜索范围
- Track 3 的超参网格设计

### Deferred Ideas (OUT OF SCOPE)
- 新特征工程
- 新模型结构（DL、IPCW stacking 等）
- 校准方法（已证明 N=221 上无法迁移）
</user_constraints>

---

## Summary

Phase 4 的核心任务是获取比 0.96624 更强的基础预测。经过深入调查，发现以下关键事实：

**公开 notebook 可访问性**：Kaggle WiDS 2026 竞赛页面在研究时无法通过 WebFetch 访问（404），无法自动枚举公开 notebook 列表及其 LB 分数。这是 Phase 4 最大的信息缺口——Track 1 的"锁定 2-3 个 0.966+ 来源"需要用户手动从 Kaggle 竞赛 Code 页面获取。

**sksurv 版本行为（已验证）**：sksurv 0.21.0 是关键分水岭。0.21 之前 `unique_times_` = 仅事件时间；0.21+ 包含所有唯一时间点（含删失）。在本竞赛数据中，由于存在 t < 12h 的事件（49 个事件 < 12h，最小事件时间 = 0.001h），sksurv 0.22.2 的 RSF 可以正确计算 p12（不全为 0）。当前环境（sksurv 0.22.2）的 p12 分布（med=0.015）低于参考 0.96624（med=0.041），差距来源于模型配置而非版本差异。版本矩阵测试优先级低。

**参考 0.96624 的分布对比（已验证）**：
- p12: ref med=0.041 vs PB med=0.032，Spearman=0.985
- p24: ref med=0.078 vs PB med=0.032，Spearman=0.988
- p48: Spearman=1.000（完全相同排序）

p48 排序完全相同意味着：若新锚点 p48 仍与 0.96624 rho=1.0，Track 2 融合对 LB 无贡献。

**Primary recommendation:** 用户必须手动从 Kaggle 获取公开 notebook URL。获取后，精确对齐特征/CV/seed/后处理是复现关键，sksurv 版本差异在本数据集上影响有限。

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sksurv | 0.22.2 (current) | RSF/EST/GBSA | 已验证，unique_times_ 包含所有时间点，p12 可正确计算 |
| scikit-learn | current | StandardScaler, CV | 依赖 |
| scipy | current | spearmanr, rank ops | 融合评估 |
| numpy | current | 数值计算 | 基础 |
| lifelines | current | C-index 计算 | OOF 评估 |

---

## Architecture Patterns

### Track 1: 精确复现模式

复现的关键对齐点（按重要性排序）：

1. **特征集**：v96624 = 13 base + 3 engineered = 16 features（已知，见 exp17_reproduce_ref.py）
2. **CV 策略**：5×10 重复分层 KFold（已知）
3. **后处理**：两轮单调链 + 72h=1.0 + clip[0.01,0.99]（已知，见 exp17 submission_postprocess）
4. **seed**：42（已知）
5. **模型配置**：RSF n_estimators=200, max_depth=5, min_samples_leaf=5, min_samples_split=10（已知）
6. **sksurv 版本**：0.22.2 在本数据集上行为与 0.21+ 一致，无需版本回溯

**关键发现**：sksurv 0.21.0 的变更（unique_times_ 包含所有时间点）在本竞赛数据上不造成 p12 差异，因为本数据有 49 个 t < 12h 的事件，unique_times_ 在 0.21+ 中已包含这些时间点，sf(12) 可正确评估。版本矩阵测试（0.17→0.22）的优先级低于特征/CV/后处理对齐。

### Track 2: Rank Average 融合

**前置条件**：Track 1 产出锚点与 0.96624 的 Spearman(p48) < 0.99。

当前 PB 与 0.96624 的 Spearman：p48=1.000（融合无意义）。若新锚点 p48 仍 rho=1.0，跳过 Track 2。

```python
def rank_average_blend(sub_a, sub_b, weight_a=0.5):
    result = sub_a[['event_id']].copy()
    for col in ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']:
        ra = sub_a[col].rank(pct=True)
        rb = sub_b[col].rank(pct=True)
        result[col] = weight_a * ra + (1 - weight_a) * rb
    return result
```

### Track 3: 超参小网格

仅在 Track 1 复现成功后执行。关键超参：
- `n_estimators`: [200, 500, 1000]
- `max_features`: ['sqrt', 0.5]
- `min_samples_leaf`: [3, 5, 8]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Rank correlation | 自写 rank 相关 | `scipy.stats.spearmanr` | 标准实现 |
| Rank normalization | 自写 percentile rank | `pd.Series.rank(pct=True)` | 处理 ties |
| Survival function eval | 自写边界处理 | `fn(float(t))` 直接调用 | sksurv 0.22.2 已处理边界 |

---

## Common Pitfalls

### Pitfall 1: 无法自动获取 Kaggle 公开 notebook
**What goes wrong:** 研究者无法通过 WebFetch 访问 Kaggle 竞赛页面（需登录或 404）
**Why it happens:** Kaggle 需要认证
**How to avoid:** 用户必须手动从 Kaggle 竞赛 Code 页面（按 Score 降序）获取 LB >= 0.966 的 notebook URL
**Warning signs:** 任何自动化爬取尝试均失败

### Pitfall 2: sksurv 版本矩阵测试耗时但收益有限
**What goes wrong:** 花大量时间安装多个 sksurv 版本，发现差异微小
**Why it happens:** 本数据集有 49 个 t < 12h 的事件，0.21+ 行为一致
**How to avoid:** 优先对齐特征/CV/后处理，版本测试作为最后手段
**Warning signs:** 版本差异测试显示 p12 分布几乎相同

### Pitfall 3: p48 Spearman=1.0 导致融合无效
**What goes wrong:** 新锚点与 0.96624 的 p48 排序完全相同，rank blend 不改变 LB
**Why it happens:** 相同特征集的 RSF 模型在 p48 上收敛到相同排序
**How to avoid:** 提交前计算 Spearman(new_p48, ref_p48)，若 > 0.99 则跳过 Track 2
**Warning signs:** Spearman(p48) > 0.99

### Pitfall 4: 复现时忽略后处理细节
**What goes wrong:** 特征/模型对齐但后处理不同，导致 p12 分布偏移
**Why it happens:** 参考代码有两轮后处理（05_ensemble.py + 06_submit.py），容易漏掉第二轮
**How to avoid:** 严格按 exp17_reproduce_ref.py 的 `submission_postprocess` 函数执行
**Warning signs:** p12 min < 0.01 或出现大量 0.01 值

---

## Code Examples

### Spearman 相似度检查（融合前置条件）
```python
from scipy.stats import spearmanr
import pandas as pd

def check_blend_eligibility(sub_a_path, sub_b_path):
    a = pd.read_csv(sub_a_path)
    b = pd.read_csv(sub_b_path)
    for col in ['prob_12h', 'prob_24h', 'prob_48h']:
        rho, _ = spearmanr(a[col], b[col])
        print(f'{col}: rho={rho:.4f} blend_eligible={rho < 0.99}')
```

### sksurv 0.22.2 StepFunction 边界行为（已验证）
```python
# domain = (0.0, x[-1])
# sf(t) for t < x[0] returns y[0] (survival at first time point)
# sf(t) for t > x[-1] raises ValueError
# Safe evaluation:
def safe_p(fn, t):
    return 1.0 - fn(min(float(t), fn.x[-1]))
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| event_times_ (only events) | unique_times_ (all times) | sksurv 0.21.0 | p12 可正确计算（本数据有早期事件） |
| ValueError on out-of-domain | Returns boundary value | sksurv 0.21.0 | 不再需要 try/except |
| max_features='auto' | max_features='sqrt' | sksurv 0.22 | 需显式指定 |

---

## Open Questions

1. **公开 notebook 的具体 URL 和代码**
   - What we know: 竞赛存在公开 notebook，0.96624 来自某个公开 notebook
   - What's unclear: 是否有 LB > 0.966 的其他公开 notebook，其特征集/模型配置
   - Recommendation: 用户手动访问 Kaggle 竞赛 Code 页面，按 Score 降序，提供 URL 给规划者

2. **新锚点 p48 是否与 0.96624 独立**
   - What we know: 当前 PB p48 Spearman=1.0 vs 0.96624
   - What's unclear: 不同特征集/模型的 p48 是否能产生独立排序
   - Recommendation: 获取新 notebook 后立即计算 Spearman(p48)，< 0.99 才值得 blend

---

## Sources

### Primary (HIGH confidence)
- sksurv 0.21.0 release notes (WebFetch verified) — unique_times_ 变更，StepFunction 边界行为
- sksurv 0.22.0 release notes (WebFetch verified) — low_memory 选项，max_features='auto' 移除
- 本地实验验证 (Bash) — sksurv 0.22.2 StepFunction 行为，RSF p12 分布，参考 submission 统计

### Secondary (MEDIUM confidence)
- experiments.md — 所有历史 LB 分数和实验结论
- exp17_reproduce_ref.py — 参考 pipeline 复现代码

### Tertiary (LOW confidence)
- Kaggle 竞赛公开 notebook 列表 — 无法访问，需用户手动获取

---

## Metadata

**Confidence breakdown:**
- sksurv 版本行为: HIGH — 本地实验直接验证
- 参考 submission 分布: HIGH — 直接读取文件计算
- 公开 notebook 信息: LOW — Kaggle 无法访问，信息缺口
- rank blend 方法: HIGH — 标准方法

**Research date:** 2026-02-22
**Valid until:** 2026-03-22

---

## 行动指南（给 Planner）

### 最高优先级：用户需要手动完成的前置步骤

在 Track 1 计划执行前，**必须**由用户完成：
1. 访问 Kaggle 竞赛 Code 页面（按 Score 降序）
2. 找到 LB >= 0.966 的公开 notebook（除 0.96624 外）
3. 提供 notebook URL 或代码给规划者

**如果用户无法提供新 notebook**，Track 1 退化为：
- 精确复现 0.96624（已有 exp17_reproduce_ref.py）
- 在 0.96624 基础上做 Track 3 超参网格

### Track 1 复现检查清单
- [ ] 特征集完全对齐（16 features）
- [ ] StandardScaler fit on train only
- [ ] RSF 超参完全对齐
- [ ] 后处理两轮完全对齐
- [ ] Spearman(reproduced_p48, ref_p48) > 0.999

### Track 2 融合前置条件
- [ ] 新锚点 LB > 0.966（独立验证）
- [ ] Spearman(new_p48, ref_p48) < 0.99
- [ ] 若 p48 rho >= 0.99，跳过 Track 2
