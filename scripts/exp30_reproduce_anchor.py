"""
Exp30: Configurable Anchor Reproduction Script
===============================================
Reproduces any RSF+GBSA pipeline via JSON config.
Default config = exact 0.96624 settings (from exp17).

Usage:
  python scripts/exp30_reproduce_anchor.py [--config path/to/config.json] [--out path/to/output.csv]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(PROJECT_DIR, "train.csv")
TEST_PATH = os.path.join(PROJECT_DIR, "test.csv")
EVAL_TIMES = [12, 24, 48, 72]

# ── Default config = exact 0.96624 settings ──
DEFAULT_CONFIG = {
    "features": [
        "low_temporal_resolution_0_5h", "log1p_area_first", "log1p_growth",
        "centroid_speed_m_per_h", "dist_min_ci_0_5h", "dist_slope_ci_0_5h",
        "dist_fit_r2_0_5h", "closing_speed_abs_m_per_h", "spread_bearing_sin",
        "spread_bearing_cos", "event_start_hour", "event_start_dayofweek",
        "event_start_month", "has_growth", "is_approaching", "log_dist_min",
    ],
    "rsf": {
        "n_estimators": 200, "max_depth": 5,
        "min_samples_leaf": 5, "min_samples_split": 10,
        "random_state": 42,
    },
    "gbsa": {
        "n_estimators": 300, "learning_rate": 0.02,
        "max_depth": 3, "min_samples_leaf": 8, "min_samples_split": 16,
        "subsample": 0.8, "dropout_rate": 0.1, "random_state": 42,
    },
    "weights": {"rsf": 0.2, "gbsa": 0.8},
    "postprocess": True,
}


def engineer_features(df):
    out = df.copy()
    out["has_growth"] = (out["log1p_growth"] > 0).astype(int)
    out["is_approaching"] = (out["dist_slope_ci_0_5h"] < 0).astype(int)
    out["log_dist_min"] = np.log1p(out["dist_min_ci_0_5h"])
    return out


def eval_survival_functions(surv_fns, n_samples):
    probs = {}
    for t in EVAL_TIMES:
        p_arr = np.zeros(n_samples)
        for i, fn in enumerate(surv_fns):
            s_t = fn(t) if t <= fn.x[-1] else fn(fn.x[-1])
            p_arr[i] = 1 - s_t
        probs[t] = np.clip(p_arr, 0, 1)
    return probs


def submission_postprocess(pred_probs):
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
    # Row-level monotonicity
    n = len(result[12])
    for i in range(n):
        prev_val = 0.0
        for t in EVAL_TIMES:
            val = max(result[t][i], prev_val)
            result[t][i] = val
            prev_val = val
    for t in EVAL_TIMES:
        result[t] = np.clip(result[t], 0.01, 1.0 if t == 72 else 0.99)
    return result


def main():
    parser = argparse.ArgumentParser(description="Configurable RSF+GBSA anchor reproduction")
    parser.add_argument("--config", default=None, help="Path to JSON config (default: 0.96624 settings)")
    parser.add_argument("--out", default=None, help="Output CSV path (default: scripts/exp30_output.csv)")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))

    out_path = args.out or os.path.join(PROJECT_DIR, "scripts", "exp30_output.csv")

    print(f"[exp30] Config: rsf_w={cfg['weights']['rsf']}, gbsa_w={cfg['weights']['gbsa']}, "
          f"rsf_seed={cfg['rsf']['random_state']}, gbsa_seed={cfg['gbsa']['random_state']}")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)

    y_time = train_fe["time_to_hit_hours"].values
    y_event = train_fe["event"].values.astype(bool)
    y_struct = np.array([(e, t) for e, t in zip(y_event, y_time)],
                        dtype=[("event", bool), ("time", float)])

    feats = cfg["features"]
    X_train = train_fe[feats].values.astype(np.float64)
    X_test = test_fe[feats].values.astype(np.float64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("[exp30] Training RSF...")
    rsf = RandomSurvivalForest(n_jobs=-1, **cfg["rsf"])
    rsf.fit(X_train, y_struct)

    print("[exp30] Training GBSA...")
    gbsa = GradientBoostingSurvivalAnalysis(**cfg["gbsa"])
    gbsa.fit(X_train, y_struct)

    n = X_test.shape[0]
    rsf_probs = eval_survival_functions(rsf.predict_survival_function(X_test), n)
    gbsa_probs = eval_survival_functions(gbsa.predict_survival_function(X_test), n)

    w_rsf, w_gbsa = cfg["weights"]["rsf"], cfg["weights"]["gbsa"]
    blended = {t: w_rsf * rsf_probs[t] + w_gbsa * gbsa_probs[t] for t in EVAL_TIMES}

    preds = submission_postprocess(blended) if cfg["postprocess"] else blended

    sub = pd.DataFrame({
        "event_id": test["event_id"].values,
        "prob_12h": preds[12], "prob_24h": preds[24],
        "prob_48h": preds[48], "prob_72h": preds[72],
    })
    sub.to_csv(out_path, index=False)

    # Distribution stats
    print(f"\n[exp30] Distribution stats:")
    for col, t in zip(["prob_12h", "prob_24h", "prob_48h", "prob_72h"], EVAL_TIMES):
        v = sub[col].values
        print(f"  {col}: min={v.min():.4f} med={np.median(v):.4f} max={v.max():.4f}")
    print(f"\n[exp30] Saved: {out_path}")


if __name__ == "__main__":
    main()
