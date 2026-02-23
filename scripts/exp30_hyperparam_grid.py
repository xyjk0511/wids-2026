"""
Exp30: RSF hyperparam small grid on reproduced pipeline.

Budget: 5 runs (4 RSF configs + 1 sksurv version check).
Configs prioritize differences from reference (n_estimators=200, max_features=0.5).
Uses full train set (no CV) for speed — OOF proxy via train-set Spearman vs ref submission.

Usage:
  .venv_sksurv22/Scripts/python scripts/exp30_hyperparam_grid.py
  .venv_sksurv22/Scripts/python scripts/exp30_hyperparam_grid.py --version_check
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(PROJECT, "train.csv")
TEST_PATH  = os.path.join(PROJECT, "test.csv")
REF_PATH   = os.path.join(PROJECT, "submissions", "submission_0.96624.csv")
EVAL_TIMES = [12, 24, 48, 72]

FEATURES = [
    "low_temporal_resolution_0_5h", "log1p_area_first", "log1p_growth",
    "centroid_speed_m_per_h", "dist_min_ci_0_5h", "dist_slope_ci_0_5h",
    "dist_fit_r2_0_5h", "closing_speed_abs_m_per_h", "spread_bearing_sin",
    "spread_bearing_cos", "event_start_hour", "event_start_dayofweek",
    "event_start_month", "has_growth", "is_approaching", "log_dist_min",
]

GBSA_CFG = dict(n_estimators=300, learning_rate=0.02, max_depth=3,
                min_samples_leaf=8, min_samples_split=16,
                subsample=0.8, dropout_rate=0.1, random_state=42)

# 4 RSF configs — most different from reference first
GRID = [
    {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 3},
    {"n_estimators": 500, "max_features": 0.5,    "min_samples_leaf": 3},
    {"n_estimators": 200, "max_features": "sqrt",  "min_samples_leaf": 3},
    {"n_estimators": 500, "max_features": 0.5,    "min_samples_leaf": 5},
]
RSF_BASE = dict(max_depth=5, min_samples_split=10, random_state=42, n_jobs=-1)


def engineer(df):
    out = df.copy()
    out["has_growth"]    = (out["log1p_growth"] > 0).astype(int)
    out["is_approaching"] = (out["dist_slope_ci_0_5h"] < 0).astype(int)
    out["log_dist_min"]  = np.log1p(out["dist_min_ci_0_5h"])
    return out


def eval_sf(surv_fns, n):
    probs = {}
    for t in EVAL_TIMES:
        arr = np.zeros(n)
        for i, fn in enumerate(surv_fns):
            s = fn(t) if t <= fn.x[-1] else fn(fn.x[-1])
            arr[i] = 1 - s
        probs[t] = np.clip(arr, 0, 1)
    return probs


def postprocess(p):
    r = {12: np.clip(p[12], 0.01, 0.99), 24: p[24].copy()}
    r[48] = np.maximum(p[48], r[24])
    r[72] = np.ones(len(p[72]))
    for t in [24, 48]:
        r[t] = np.clip(r[t], 0.01, 0.99)
    n = len(r[12])
    for i in range(n):
        prev = 0.0
        for t in EVAL_TIMES:
            r[t][i] = max(r[t][i], prev); prev = r[t][i]
    for t in EVAL_TIMES:
        r[t] = np.clip(r[t], 0.01, 1.0 if t == 72 else 0.99)
    return r


def run_config(rsf_cfg, X_train, y_struct, X_test, test_ids, ref_sub, label, out_path=None):
    rsf = RandomSurvivalForest(**{**RSF_BASE, **rsf_cfg})
    rsf.fit(X_train, y_struct)
    gbsa = GradientBoostingSurvivalAnalysis(**GBSA_CFG)
    gbsa.fit(X_train, y_struct)

    n = X_test.shape[0]
    rp = eval_sf(rsf.predict_survival_function(X_test), n)
    gp = eval_sf(gbsa.predict_survival_function(X_test), n)
    blended = {t: 0.2 * rp[t] + 0.8 * gp[t] for t in EVAL_TIMES}
    pp = postprocess(blended)

    sub = pd.DataFrame({"event_id": test_ids,
                        "prob_12h": pp[12], "prob_24h": pp[24],
                        "prob_48h": pp[48], "prob_72h": pp[72]})

    rho48, _ = spearmanr(sub["prob_48h"].values,
                         ref_sub.set_index("event_id").loc[test_ids]["prob_48h"].values)
    print(f"  {label:45s}  rho_p48={rho48:.4f}")

    if out_path:
        sub.to_csv(out_path, index=False)
        print(f"    -> saved {out_path}")
    return sub, rho48


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version_check", action="store_true",
                        help="Run sksurv version comparison (Run 5)")
    args = parser.parse_args()

    import sksurv
    print(f"[exp30_grid] sksurv={sksurv.__version__}")

    train = engineer(pd.read_csv(TRAIN_PATH))
    test  = engineer(pd.read_csv(TEST_PATH))
    ref   = pd.read_csv(REF_PATH)

    y_struct = np.array([(bool(e), float(t)) for e, t in
                         zip(train["event"], train["time_to_hit_hours"])],
                        dtype=[("event", bool), ("time", float)])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES].values.astype(np.float64))
    X_test  = scaler.transform(test[FEATURES].values.astype(np.float64))
    test_ids = test["event_id"].values

    print(f"\n{'Config':45s}  {'rho_p48':>8}")
    print("-" * 60)

    results = []
    for i, cfg in enumerate(GRID):
        label = (f"n={cfg['n_estimators']} mf={cfg['max_features']} "
                 f"msl={cfg['min_samples_leaf']}")
        sub, rho = run_config(cfg, X_train, y_struct, X_test, test_ids, ref, label)
        results.append((rho, label, sub, cfg))

    # Sort by rho deviation from 1.0 (most different = most interesting)
    results.sort(key=lambda x: abs(1.0 - x[0]))

    print("\nTop-2 configs (most different p48 ranking from ref):")
    for rank, (rho, label, sub, cfg) in enumerate(results[:2]):
        out = os.path.join(PROJECT, "submissions",
                           f"submission_exp30_grid_r{rank+1}.csv")
        sub.to_csv(out, index=False)
        print(f"  [{rank+1}] {label}  rho_p48={rho:.4f}  -> {out}")

    if args.version_check:
        print("\n[Run 5] sksurv version check — re-run ref config with current env")
        ref_cfg = {"n_estimators": 200, "max_features": 0.5, "min_samples_leaf": 5}
        label = f"REF n=200 mf=0.5 msl=5 (sksurv={sksurv.__version__})"
        run_config(ref_cfg, X_train, y_struct, X_test, test_ids, ref, label)


if __name__ == "__main__":
    main()
