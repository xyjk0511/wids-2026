# exp13_stability_protocol

目标：在不改动主训练流程（`src/train.py`）的前提下，建立一套更稳健的离线评估协议，减少“小数点后四位幻觉”。

## 包含内容

- `cv_protocol.py`
  - 复用型 CV 工具。
  - 新增 `event_time` 复合分层：`event/censor + time_bin(<=12,12-24,24-48,>48)`。
  - 自动合并稀有分层，避免 `RepeatedStratifiedKFold` 因样本过少失效。
  - 统一 OOF 打分、模型融合、权重搜索。

- `stability_benchmark.py`
  - 多 CV 随机种子稳定性评估。
  - 比较三种策略：
    - `RSF-only`
    - 固定权重融合（默认 `RSF=0.90 + EST=0.10`）
    - OOF最优融合（可再按 `max_rsf_weight` 做 cap）
  - 输出 `mean/std` 和 `delta/noise`（相对基线波动的信噪比）。

- `isomorphic_oof_eval.py`
  - 按“提交策略同构”的方式评估 OOF：
    - 先按 raw OOF 搜索权重
    - 再应用 cap（默认 0.90）
    - 再执行 `submission_postprocess + monotonic enforcement`
  - 可对比 `event` 与 `event_time` 分层策略。

## 运行示例

从项目根目录执行：

```bash
python experiments/exp13_stability_protocol/stability_benchmark.py --strat-mode event_time --cv-seeds 1,7,21,42,2026
```

```bash
python experiments/exp13_stability_protocol/isomorphic_oof_eval.py --compare-strat --cv-seed 42
```

## 建议使用顺序

1. 先跑 `stability_benchmark.py`，确认候选策略是否超过噪声门槛。
2. 再跑 `isomorphic_oof_eval.py`，检查“OOF->提交策略”是否一致。
3. 只有当改进在多种子下稳定为正，再考虑迁移到 `src/train.py` 主流程。

