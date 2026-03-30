"""
Exp17 Step 1: 精确复现参考 0.96624 Pipeline
=============================================
目标: 逐元素对齐参考 submission_0.96624.csv

参考 pipeline (from SCORE_LOG):
- 15 特征 (12 selected + 3 engineered)
- 全局 StandardScaler (fit on train, transform both)
- RSF(0.2) + GBSA(0.8) 加权集成
- RSF: n_estimators=200, max_depth=5, min_samples_leaf=5, min_samples_split=10, seed=42
- GBSA: n_estimators=300, lr=0.02, max_depth=3, leaf=8, split=16, subsample=0.8, dropout=0.1
- SF 评估: fn(t) if t <= fn.x[-1] else fn(fn.x[-1])
- 后处理: 12h clip[0.01,0.99], 24/48/72 单调链, 72h=1.0, clip[0.01,0.99]
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

# ── 路径 ──
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_SUB_PATH = os.path.join(PROJECT_DIR, "submission_0.96624.csv")
TRAIN_PATH = os.path.join(PROJECT_DIR, "train.csv")
TEST_PATH = os.path.join(PROJECT_DIR, "test.csv")

EVAL_TIMES = [12, 24, 48, 72]


# ── 1. 特征定义 (与参考 02_preprocessing.py 完全一致) ──
SELECTED_FEATURES = [
    "low_temporal_resolution_0_5h",
    "log1p_area_first",
    "log1p_growth",
    "centroid_speed_m_per_h",
    "dist_min_ci_0_5h",
    "dist_slope_ci_0_5h",
    "dist_fit_r2_0_5h",
    "closing_speed_abs_m_per_h",
    "spread_bearing_sin",
    "spread_bearing_cos",
    "event_start_hour",
    "event_start_dayofweek",
    "event_start_month",
]

ENGINEERED_FEATURES = ["has_growth", "is_approaching", "log_dist_min"]
FINAL_FEATURES = SELECTED_FEATURES + ENGINEERED_FEATURES  # 15 total


def engineer_features(df):
    """与参考 02_preprocessing.py 完全一致的特征工程"""
    out = df.copy()
    out["has_growth"] = (out["log1p_growth"] > 0).astype(int)
    out["is_approaching"] = (out["dist_slope_ci_0_5h"] < 0).astype(int)
    out["log_dist_min"] = np.log1p(out["dist_min_ci_0_5h"])
    return out


# ── 2. SF 评估函数 (与参考 04_models.py 完全一致) ──
def eval_survival_functions(surv_fns, n_samples):
    """fn(t) if t <= fn.x[-1] else fn(fn.x[-1])"""
    probs = {}
    for t in EVAL_TIMES:
        p_arr = np.zeros(n_samples)
        for i, fn in enumerate(surv_fns):
            s_t = fn(t) if t <= fn.x[-1] else fn(fn.x[-1])
            p_arr[i] = 1 - s_t
        probs[t] = np.clip(p_arr, 0, 1)
    return probs


# ── 3. 后处理 (与参考 03_evaluation.py submission_postprocess 完全一致) ──
def submission_postprocess(pred_probs):
    """
    两轮后处理, 与参考 05_ensemble.py + 06_submit.py 完全一致:
    Round 1 (05_ensemble.py): 12h clip, 24/48/72 mono chain, 72h=1.0, clip
    Round 2 (06_submit.py): 行级 12<=24<=48<=72 强制, 再 clip
    """
    # Round 1: submission_postprocess from 03_evaluation.py
    result = {}
    result[12] = np.clip(pred_probs[12], 0.01, 0.99)
    prev = pred_probs[24]
    result[24] = prev
    for t in [48, 72]:
        current = np.maximum(pred_probs[t], prev)
        result[t] = current
        prev = current
    result[72] = np.ones(len(result[72]))
    for t in [24, 48]:
        result[t] = np.clip(result[t], 0.01, 0.99)

    # Round 2: 06_submit.py 行级单调性 (确保 12h <= 24h <= 48h <= 72h)
    cols = [12, 24, 48, 72]
    n = len(result[12])
    for i in range(n):
        prev_val = 0.0
        for t in cols:
            val = max(result[t][i], prev_val)
            result[t][i] = val
            prev_val = val
    # Round 2 clip
    for t in cols:
        if t == 72:
            result[t] = np.clip(result[t], 0.01, 1.0)
        else:
            result[t] = np.clip(result[t], 0.01, 0.99)
    return result


# ── 4. 对比分析工具 ──
def compare_submissions(ours, ref, label=""):
    """对比两个 submission 的逐 horizon 差异"""
    print(f"\n{'='*60}")
    print(f"对比: {label}")
    print(f"{'='*60}")

    all_pass = True
    for col, t in zip(
        ["prob_12h", "prob_24h", "prob_48h", "prob_72h"], EVAL_TIMES
    ):
        a = ours[col].values
        b = ref[col].values
        diff = np.abs(a - b)
        max_diff = diff.max()
        mean_diff = diff.mean()
        corr, _ = spearmanr(a, b)

        status = "PASS" if max_diff < 1e-3 and corr > 0.999 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"\n  {col} ({t}h)  [{status}]")
        print(f"    max_abs_diff = {max_diff:.6e}")
        print(f"    mean_abs_diff = {mean_diff:.6e}")
        print(f"    Spearman = {corr:.6f}")
        print(f"    ours:  min={a.min():.4f} p10={np.percentile(a,10):.4f} "
              f"med={np.median(a):.4f} p90={np.percentile(a,90):.4f} "
              f"max={a.max():.4f} unique={len(np.unique(a))}")
        print(f"    ref:   min={b.min():.4f} p10={np.percentile(b,10):.4f} "
              f"med={np.median(b):.4f} p90={np.percentile(b,90):.4f} "
              f"max={b.max():.4f} unique={len(np.unique(b))}")

    return all_pass


# ── 5. 主流程 ──
def main():
    print("=" * 60)
    print("Exp17 Step 1: 精确复现参考 0.96624 Pipeline")
    print("=" * 60)

    # 加载数据
    print("\n[1/6] 加载数据...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    ref_sub = pd.read_csv(REF_SUB_PATH)
    print(f"  train: {train.shape}, test: {test.shape}")

    # 特征工程
    print("\n[2/6] 特征工程 (15 features)...")
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)

    y_time = train_fe["time_to_hit_hours"].values
    y_event = train_fe["event"].values.astype(bool)

    X_train = train_fe[FINAL_FEATURES].values.astype(np.float64)
    X_test = test_fe[FINAL_FEATURES].values.astype(np.float64)
    print(f"  features: {FINAL_FEATURES}")
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # 全局 scaler
    print("\n[3/6] 全局 StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 构造 y_struct
    y_struct = np.array(
        [(e, t) for e, t in zip(y_event, y_time)],
        dtype=[("event", bool), ("time", float)],
    )

    # ── 方案 A: RSF 单模型 (05_ensemble.py 当前代码) ──
    print("\n[4/6] 训练模型...")
    print("  训练 RSF (n_estimators=200, max_depth=5)...")
    rsf = RandomSurvivalForest(
        n_estimators=200, max_depth=5,
        min_samples_leaf=5, min_samples_split=10,
        random_state=42, n_jobs=-1,
    )
    rsf.fit(X_train, y_struct)

    print("  训练 GBSA (n_estimators=300, lr=0.02)...")
    gbsa = GradientBoostingSurvivalAnalysis(
        n_estimators=300, learning_rate=0.02,
        max_depth=3, min_samples_leaf=8, min_samples_split=16,
        subsample=0.8, dropout_rate=0.1, random_state=42,
    )
    gbsa.fit(X_train, y_struct)

    # ── 预测 ──
    print("\n[5/6] 生成预测...")
    rsf_surv = rsf.predict_survival_function(X_test)
    gbsa_surv = gbsa.predict_survival_function(X_test)
    n = X_test.shape[0]

    rsf_probs = eval_survival_functions(rsf_surv, n)
    gbsa_probs = eval_survival_functions(gbsa_surv, n)

    # 方案 A: RSF 单模型
    rsf_only = submission_postprocess(rsf_probs)

    # 方案 B: RSF(0.2) + GBSA(0.8) (SCORE_LOG 记录)
    blended = {}
    for t in EVAL_TIMES:
        blended[t] = 0.2 * rsf_probs[t] + 0.8 * gbsa_probs[t]
    blend_post = submission_postprocess(blended)

    # 方案 C: RSF(0.8) + GBSA(0.2) (反向权重)
    blended_rev = {}
    for t in EVAL_TIMES:
        blended_rev[t] = 0.8 * rsf_probs[t] + 0.2 * gbsa_probs[t]
    blend_rev_post = submission_postprocess(blended_rev)

    # ── 构造 submission DataFrames ──
    test_ids = test["event_id"].values

    def make_sub(preds):
        return pd.DataFrame({
            "event_id": test_ids,
            "prob_12h": preds[12],
            "prob_24h": preds[24],
            "prob_48h": preds[48],
            "prob_72h": preds[72],
        })

    sub_rsf = make_sub(rsf_only)
    sub_blend = make_sub(blend_post)
    sub_blend_rev = make_sub(blend_rev_post)

    # 确保 event_id 顺序与参考一致
    ref_order = ref_sub["event_id"].values
    sub_rsf = sub_rsf.set_index("event_id").loc[ref_order].reset_index()
    sub_blend = sub_blend.set_index("event_id").loc[ref_order].reset_index()
    sub_blend_rev = sub_blend_rev.set_index("event_id").loc[ref_order].reset_index()

    # ── 对比 ──
    print("\n[6/6] 对比参考 submission...")
    pass_a = compare_submissions(sub_rsf, ref_sub, "方案A: RSF 单模型")
    pass_b = compare_submissions(sub_blend, ref_sub, "方案B: RSF(0.2)+GBSA(0.8)")
    pass_c = compare_submissions(sub_blend_rev, ref_sub, "方案C: RSF(0.8)+GBSA(0.2)")

    # ── 单调性检查 ──
    print(f"\n{'='*60}")
    print("单调性检查")
    print(f"{'='*60}")
    for name, sub in [("RSF", sub_rsf), ("Blend", sub_blend), ("BlendRev", sub_blend_rev)]:
        violations = 0
        for i in range(len(sub)):
            row = [sub.iloc[i][c] for c in ["prob_12h","prob_24h","prob_48h","prob_72h"]]
            for j in range(len(row)-1):
                if row[j] > row[j+1] + 1e-8:
                    violations += 1
                    break
        print(f"  {name}: {violations} violations")

    # ── 结论 ──
    print(f"\n{'='*60}")
    print("结论")
    print(f"{'='*60}")
    best = None
    if pass_a:
        print("  方案A (RSF单模型) 通过所有验收标准!")
        best = ("rsf_only", sub_rsf)
    if pass_b:
        print("  方案B (RSF(0.2)+GBSA(0.8)) 通过所有验收标准!")
        best = ("blend_0.2_0.8", sub_blend)
    if pass_c:
        print("  方案C (RSF(0.8)+GBSA(0.2)) 通过所有验收标准!")
        best = ("blend_0.8_0.2", sub_blend_rev)
    if not any([pass_a, pass_b, pass_c]):
        print("  所有方案均未通过验收标准, 需要进一步排查")
        # 找最接近的方案
        best_name, best_max_diff = None, float("inf")
        for name, sub in [("rsf_only", sub_rsf), ("blend", sub_blend), ("blend_rev", sub_blend_rev)]:
            max_d = max(
                np.abs(sub[c].values - ref_sub[c].values).max()
                for c in ["prob_12h","prob_24h","prob_48h","prob_72h"]
            )
            if max_d < best_max_diff:
                best_max_diff = max_d
                best_name = name
                best = (name, sub)
        print(f"  最接近的方案: {best_name} (max_diff={best_max_diff:.6e})")

    # 保存最佳方案
    if best:
        out_path = os.path.join(PROJECT_DIR, "scripts", "exp17_reproduced.csv")
        best[1].to_csv(out_path, index=False)
        print(f"\n  [saved] {out_path}")


if __name__ == "__main__":
    main()
