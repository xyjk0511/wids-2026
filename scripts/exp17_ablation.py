"""
Exp17 Step 2: 逐项消融实验
===========================
基线: RSF 单模型 (15 features, 200 trees, seed=42, 全局 scaler)
消融 A~F: 每次只改一个维度, 测量 CV 指标和 test 侧分布变化

消融项:
A. 特征膨胀 (15→36)
B. 树数增加 (200→1000)
C. 模型混合 (加 EST 50/50)
D. Scaler 方式 (全局→折内)
E. 后处理 (参考 clip→当前 floor=1e-6 + split mono)
F. Test 预测方式 (单次 full retrain→多折 CV ensemble 平均)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.ensemble import RandomSurvivalForest, ExtraSurvivalTrees
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index as lifelines_cindex

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(PROJECT_DIR, "train.csv")
TEST_PATH = os.path.join(PROJECT_DIR, "test.csv")
REF_SUB_PATH = os.path.join(PROJECT_DIR, "submission_0.96624.csv")

EVAL_TIMES = [12, 24, 48, 72]
BRIER_WEIGHTS = {24: 0.3, 48: 0.4, 72: 0.3}


# ── 特征定义 ──
FEATURES_15 = [
    "low_temporal_resolution_0_5h", "log1p_area_first", "log1p_growth",
    "centroid_speed_m_per_h", "dist_min_ci_0_5h", "dist_slope_ci_0_5h",
    "dist_fit_r2_0_5h", "closing_speed_abs_m_per_h", "spread_bearing_sin",
    "spread_bearing_cos", "event_start_hour", "event_start_dayofweek",
    "event_start_month", "has_growth", "is_approaching", "log_dist_min",
]

# 36 features = all original columns + engineered
FEATURES_36 = None  # will be set after loading data


def engineer_features(df):
    out = df.copy()
    out["has_growth"] = (out["log1p_growth"] > 0).astype(int)
    out["is_approaching"] = (out["dist_slope_ci_0_5h"] < 0).astype(int)
    out["log_dist_min"] = np.log1p(out["dist_min_ci_0_5h"])
    return out


def eval_sf(surv_fns, n):
    """参考 SF 评估: fn(t) if t<=fn.x[-1] else fn(fn.x[-1])"""
    probs = {}
    for t in EVAL_TIMES:
        p = np.zeros(n)
        for i, fn in enumerate(surv_fns):
            s = fn(t) if t <= fn.x[-1] else fn(fn.x[-1])
            p[i] = 1 - s
        probs[t] = np.clip(p, 0, 1)
    return probs


# ── 后处理方案 ──
def postprocess_ref(pred):
    """参考后处理: 12h clip[0.01,0.99], 24/48/72 mono, 72h=1.0, clip"""
    r = {}
    r[12] = np.clip(pred[12], 0.01, 0.99)
    prev = pred[24]
    r[24] = prev
    for t in [48, 72]:
        cur = np.maximum(pred[t], prev)
        r[t] = cur
        prev = cur
    r[72] = np.ones(len(r[72]))
    for t in [24, 48]:
        r[t] = np.clip(r[t], 0.01, 0.99)
    # 06_submit.py 行级单调性
    n = len(r[12])
    for i in range(n):
        pv = 0.0
        for t in EVAL_TIMES:
            v = max(r[t][i], pv)
            r[t][i] = v
            pv = v
    for t in EVAL_TIMES:
        hi = 1.0 if t == 72 else 0.99
        r[t] = np.clip(r[t], 0.01, hi)
    return r


def postprocess_current(pred):
    """当前后处理: floor=1e-6, split mono (12h独立, 24/48/72链), clip"""
    r = {}
    floor = 1e-6
    r[12] = np.clip(pred[12], floor, 1.0)
    prev = np.clip(pred[24], floor, 1.0)
    r[24] = prev
    for t in [48, 72]:
        cur = np.maximum(np.clip(pred[t], floor, 1.0), prev)
        r[t] = cur
        prev = cur
    r[72] = np.ones(len(r[72]))
    return r


# ── 评估函数 ──
def brier_censored(y_time, y_event, pred, t):
    hit = y_event & (y_time <= t)
    after = y_time > t
    eligible = hit | after
    if eligible.sum() == 0:
        return 0.0
    y_true = hit[eligible].astype(float)
    return np.mean((y_true - pred[eligible]) ** 2)


def hybrid_score(y_time, y_event, preds):
    wb = sum(
        w * brier_censored(y_time, y_event, preds[t], t)
        for t, w in BRIER_WEIGHTS.items()
    )
    ci = lifelines_cindex(y_time, -preds[12], y_event)
    score = 0.3 * ci + 0.7 * (1 - wb)
    return score, {"hybrid": score, "c_index": ci, "w_brier": wb}


def cv_evaluate(X, y_time, y_event, model_factory, postprocess_fn,
                fit_scaler=False, n_splits=5, n_repeats=10):
    """5x10 重复分层 CV"""
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=42)
    scores, details_list = [], []
    for tr_idx, va_idx in rkf.split(X, y_event.astype(int)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        yt_tr, yt_va = y_time[tr_idx], y_time[va_idx]
        ye_tr, ye_va = y_event[tr_idx], y_event[va_idx]
        if fit_scaler:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_va = sc.transform(X_va)
        pred_fn = model_factory(X_tr, yt_tr, ye_tr)
        preds = pred_fn(X_va)
        preds = postprocess_fn(preds)
        s, d = hybrid_score(yt_va, ye_va, preds)
        scores.append(s)
        details_list.append(d)
    return scores, details_list


# ── 模型工厂 ──
def rsf_factory(n_est=200, seed=42):
    def factory(X_tr, yt_tr, ye_tr):
        y_s = np.array([(e,t) for e,t in zip(ye_tr, yt_tr)],
                       dtype=[("event",bool),("time",float)])
        rsf = RandomSurvivalForest(
            n_estimators=n_est, max_depth=5,
            min_samples_leaf=5, min_samples_split=10,
            random_state=seed, n_jobs=-1)
        rsf.fit(X_tr, y_s)
        def predict(X):
            return eval_sf(rsf.predict_survival_function(X), X.shape[0])
        return predict
    return factory


def rsf_est_factory(n_est=200, seed=42):
    """RSF + EST 50/50 混合"""
    def factory(X_tr, yt_tr, ye_tr):
        y_s = np.array([(e,t) for e,t in zip(ye_tr, yt_tr)],
                       dtype=[("event",bool),("time",float)])
        rsf = RandomSurvivalForest(
            n_estimators=n_est, max_depth=5,
            min_samples_leaf=5, min_samples_split=10,
            random_state=seed, n_jobs=-1)
        est = ExtraSurvivalTrees(
            n_estimators=n_est, max_depth=5,
            min_samples_leaf=5, min_samples_split=10,
            random_state=seed, n_jobs=-1)
        rsf.fit(X_tr, y_s)
        est.fit(X_tr, y_s)
        def predict(X):
            n = X.shape[0]
            rp = eval_sf(rsf.predict_survival_function(X), n)
            ep = eval_sf(est.predict_survival_function(X), n)
            return {t: 0.5*rp[t] + 0.5*ep[t] for t in EVAL_TIMES}
        return predict
    return factory


# ── 结果打印 ──
def print_result(name, scores, details, baseline_test_probs=None,
                 test_probs=None, ref_sub=None):
    s = np.array(scores)
    ci = np.mean([d["c_index"] for d in details])
    wb = np.mean([d["w_brier"] for d in details])
    print(f"\n  [{name}]")
    print(f"    CV Hybrid: {s.mean():.4f} +/- {s.std():.4f}")
    print(f"    CV CI:     {ci:.4f}")
    print(f"    CV WBrier: {wb:.4f}")

    if test_probs is not None:
        u12 = len(np.unique(test_probs[12]))
        floor_count = np.sum(test_probs[12] <= 0.011)
        print(f"    Test 12h: unique={u12}, floor_count={floor_count}")

    if baseline_test_probs is not None and test_probs is not None:
        for t in [12, 24, 48]:
            a, b = test_probs[t], baseline_test_probs[t]
            corr, _ = spearmanr(a, b)
            md = np.abs(a - b).max()
            print(f"    vs baseline {t}h: spearman={corr:.4f}, max_diff={md:.4f}")

    if ref_sub is not None and test_probs is not None:
        for col, t in zip(["prob_12h","prob_24h","prob_48h"], [12,24,48]):
            a, b = test_probs[t], ref_sub[col].values
            corr, _ = spearmanr(a, b)
            print(f"    vs ref 0.96624 {t}h: spearman={corr:.4f}")


# ── 主流程 ──
def main():
    print("=" * 60)
    print("Exp17 Step 2: 逐项消融实验")
    print("=" * 60)

    # 加载数据
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    ref_sub = pd.read_csv(REF_SUB_PATH)

    train_fe = engineer_features(train)
    test_fe = engineer_features(test)

    y_time = train_fe["time_to_hit_hours"].values
    y_event = train_fe["event"].values.astype(bool)

    # 15 features (基线)
    X_train_15 = train_fe[FEATURES_15].values.astype(np.float64)
    X_test_15 = test_fe[FEATURES_15].values.astype(np.float64)

    # 36 features (消融 A)
    global FEATURES_36
    exclude = {"event_id", "time_to_hit_hours", "event",
               "has_growth", "is_approaching", "log_dist_min"}
    base_cols = [c for c in train.columns if c not in exclude]
    FEATURES_36 = base_cols + ["has_growth", "is_approaching", "log_dist_min"]
    X_train_36 = train_fe[FEATURES_36].values.astype(np.float64)
    X_test_36 = test_fe[FEATURES_36].values.astype(np.float64)

    print(f"  15 features: {len(FEATURES_15)}")
    print(f"  36 features: {len(FEATURES_36)}")

    # 全局 scaler (基线用)
    scaler_15 = StandardScaler()
    X_train_15s = scaler_15.fit_transform(X_train_15)
    X_test_15s = scaler_15.transform(X_test_15)

    scaler_36 = StandardScaler()
    X_train_36s = scaler_36.fit_transform(X_train_36)
    X_test_36s = scaler_36.transform(X_test_36)

    # 对齐 event_id 顺序
    ref_ids = ref_sub["event_id"].values
    test_ids = test["event_id"].values
    idx_map = {eid: i for i, eid in enumerate(test_ids)}
    order = [idx_map[eid] for eid in ref_ids]

    def get_test_probs(factory, X_tr, X_te, postprocess_fn):
        pred_fn = factory(X_tr, y_time, y_event)
        raw = pred_fn(X_te)
        post = postprocess_fn(raw)
        return {t: post[t][order] for t in EVAL_TIMES}

    # ══════════════════════════════════════════
    # 基线: RSF(200, seed=42), 15 features, 全局 scaler, 参考后处理
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("基线: RSF(200), 15 features, 全局 scaler, 参考后处理")
    print("=" * 60)

    base_scores, base_details = cv_evaluate(
        X_train_15s, y_time, y_event,
        rsf_factory(200, 42), postprocess_ref)
    base_test = get_test_probs(
        rsf_factory(200, 42), X_train_15s, X_test_15s, postprocess_ref)
    print_result("BASELINE", base_scores, base_details,
                 test_probs=base_test, ref_sub=ref_sub)

    # ══════════════════════════════════════════
    # 消融 A: 特征膨胀 15→36
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("消融 A: 特征膨胀 15→36")
    print("=" * 60)

    a_scores, a_details = cv_evaluate(
        X_train_36s, y_time, y_event,
        rsf_factory(200, 42), postprocess_ref)
    a_test = get_test_probs(
        rsf_factory(200, 42), X_train_36s, X_test_36s, postprocess_ref)
    print_result("A: 36 features", a_scores, a_details,
                 baseline_test_probs=base_test, test_probs=a_test,
                 ref_sub=ref_sub)

    # ══════════════════════════════════════════
    # 消融 B: 树数 200→1000
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("消融 B: 树数 200→1000")
    print("=" * 60)

    b_scores, b_details = cv_evaluate(
        X_train_15s, y_time, y_event,
        rsf_factory(1000, 42), postprocess_ref)
    b_test = get_test_probs(
        rsf_factory(1000, 42), X_train_15s, X_test_15s, postprocess_ref)
    print_result("B: 1000 trees", b_scores, b_details,
                 baseline_test_probs=base_test, test_probs=b_test,
                 ref_sub=ref_sub)

    # ══════════════════════════════════════════
    # 消融 C: 加 EST 50/50 混合
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("消融 C: RSF+EST 50/50 混合")
    print("=" * 60)

    c_scores, c_details = cv_evaluate(
        X_train_15s, y_time, y_event,
        rsf_est_factory(200, 42), postprocess_ref)
    c_test = get_test_probs(
        rsf_est_factory(200, 42), X_train_15s, X_test_15s, postprocess_ref)
    print_result("C: RSF+EST", c_scores, c_details,
                 baseline_test_probs=base_test, test_probs=c_test,
                 ref_sub=ref_sub)

    # ══════════════════════════════════════════
    # 消融 D: Scaler 全局→折内
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("消融 D: Scaler 全局→折内")
    print("=" * 60)

    # CV 用折内 scaler (传入未缩放数据)
    d_scores, d_details = cv_evaluate(
        X_train_15, y_time, y_event,
        rsf_factory(200, 42), postprocess_ref,
        fit_scaler=True)
    # Test 仍用全局 scaler (因为 test 只有一种方式)
    d_test = base_test  # test 侧不变
    print_result("D: fold scaler", d_scores, d_details,
                 baseline_test_probs=base_test, test_probs=d_test,
                 ref_sub=ref_sub)

    # ══════════════════════════════════════════
    # 消融 E: 后处理 参考→当前 (floor=1e-6, split mono)
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("消融 E: 后处理 参考→当前 (floor=1e-6, split mono)")
    print("=" * 60)

    e_scores, e_details = cv_evaluate(
        X_train_15s, y_time, y_event,
        rsf_factory(200, 42), postprocess_current)
    e_test = get_test_probs(
        rsf_factory(200, 42), X_train_15s, X_test_15s, postprocess_current)
    print_result("E: current postproc", e_scores, e_details,
                 baseline_test_probs=base_test, test_probs=e_test,
                 ref_sub=ref_sub)

    # ══════════════════════════════════════════
    # 消融 F: Test 预测 单次→多折 CV ensemble
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("消融 F: Test 预测 单次→多折 CV ensemble")
    print("=" * 60)

    # CV 指标不变 (CV 评估方式相同)
    f_scores, f_details = base_scores, base_details

    # Test 侧: 50 折 CV ensemble 平均
    from sklearn.model_selection import StratifiedKFold
    n_folds_test = 50
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ensemble_probs = {t: np.zeros(X_test_15s.shape[0]) for t in EVAL_TIMES}
    n_models = 0
    for repeat in range(10):
        for tr_idx, _ in skf.split(X_train_15s, y_event.astype(int)):
            pred_fn = rsf_factory(200, 42)(
                X_train_15s[tr_idx], y_time[tr_idx], y_event[tr_idx])
            raw = pred_fn(X_test_15s)
            for t in EVAL_TIMES:
                ensemble_probs[t] += raw[t]
            n_models += 1
    for t in EVAL_TIMES:
        ensemble_probs[t] /= n_models
    f_test_raw = ensemble_probs
    f_test = postprocess_ref(f_test_raw)
    f_test = {t: f_test[t][order] for t in EVAL_TIMES}
    print_result("F: CV ensemble test", f_scores, f_details,
                 baseline_test_probs=base_test, test_probs=f_test,
                 ref_sub=ref_sub)

    # ══════════════════════════════════════════
    # 汇总
    # ══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("汇总: CV Hybrid 对比")
    print("=" * 60)
    results = [
        ("BASELINE", base_scores),
        ("A: 36 features", a_scores),
        ("B: 1000 trees", b_scores),
        ("C: RSF+EST", c_scores),
        ("D: fold scaler", d_scores),
        ("E: current postproc", e_scores),
        ("F: CV ensemble test", f_scores),
    ]
    base_mean = np.mean(base_scores)
    for name, scores in sorted(results, key=lambda x: -np.mean(x[1])):
        m = np.mean(scores)
        diff = m - base_mean
        print(f"  {name:25s}: {m:.4f} ({diff:+.4f})")


if __name__ == "__main__":
    main()
