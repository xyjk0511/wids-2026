"""Exp32: Compare isotonic, Platt, piecewise-linear calibration on 24h/48h OOF.

Two tracks:
  Track A (anchor-incremental): new_p = anchor_p + alpha*(cal_p - anchor_p)
  Track B (independent): calibration applied directly

Best method selected by OOF hybrid score without CI degradation.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit as sp_logit

from src.config import (
    TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH, TIME_COL, EVENT_COL,
    HORIZONS, PROB_COLS, RANDOM_STATE,
    FEATURES_V96624_BASE, FEATURES_V96624_ENGINEERED,
)
from src.features import add_engineered
from src.models import RSF, EST
from src.labels import build_horizon_labels
from src.evaluation import hybrid_score, horizon_brier_score, c_index
from src.monotonic import submission_postprocess
from src.train import _strat_labels

FEATURES = list(FEATURES_V96624_BASE) + list(FEATURES_V96624_ENGINEERED)
SEEDS = [42, 123, 456, 789, 2026]
N_SPLITS = 5
ANCHOR_PATH = "submissions/submission_0.96624.csv"
EPS = 1e-7


def safe_logit(p):
    return sp_logit(np.clip(np.asarray(p, float), EPS, 1.0 - EPS))


# ---------------------------------------------------------------------------
# OOF generation: RSF+EST 50/50 blend, 5-fold x 5-seed
# ---------------------------------------------------------------------------

def run_oof_blend(train, feature_cols):
    """5-fold x 5-seed OOF for RSF+EST 50/50 blend."""
    X = train[feature_cols]
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values
    n = len(train)

    oof = {h: np.zeros(n) for h in HORIZONS}
    counts = np.zeros(n)

    strat = _strat_labels(y_time, y_event)

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(X, strat):
            scaler = StandardScaler()
            X_tr = pd.DataFrame(
                scaler.fit_transform(X.iloc[tr_idx]),
                columns=feature_cols, index=X.iloc[tr_idx].index,
            )
            X_va = pd.DataFrame(
                scaler.transform(X.iloc[va_idx]),
                columns=feature_cols, index=X.iloc[va_idx].index,
            )
            yt_tr, ye_tr = y_time[tr_idx], y_event[tr_idx]

            rsf = RSF(random_state=seed)
            rsf.fit(X_tr, yt_tr, ye_tr)
            p_rsf = rsf.predict_proba(X_va)

            est = EST(random_state=seed)
            est.fit(X_tr, yt_tr, ye_tr)
            p_est = est.predict_proba(X_va)

            for h in HORIZONS:
                oof[h][va_idx] += 0.5 * p_rsf[h] + 0.5 * p_est[h]
            counts[va_idx] += 1

    mask = counts > 0
    for h in HORIZONS:
        oof[h][mask] /= counts[mask]

    print(f"  OOF done. Counts: min={counts.min():.0f} max={counts.max():.0f}")
    return oof, y_time, y_event


# ---------------------------------------------------------------------------
# Calibration methods
# ---------------------------------------------------------------------------

def fit_isotonic(p_train, y_train):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_train, y_train)
    return iso

def apply_isotonic(iso, p):
    return iso.predict(p)


def fit_platt(p_train, y_train):
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(safe_logit(p_train).reshape(-1, 1), y_train)
    return lr

def apply_platt(lr, p):
    return lr.predict_proba(safe_logit(p).reshape(-1, 1))[:, 1]


def fit_piecewise(p_train, y_train):
    """3-knot piecewise linear calibration at p25/p50/p75."""
    knots = np.quantile(p_train, [0.25, 0.50, 0.75])
    # Build feature matrix: [p, (p-k1)+, (p-k2)+, (p-k3)+]
    def _features(p):
        p = np.asarray(p, float)
        cols = [p] + [np.maximum(p - k, 0.0) for k in knots]
        return np.column_stack(cols)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(_features(p_train), y_train)
    return lr, knots

def apply_piecewise(model_knots, p):
    lr, knots = model_knots
    p = np.asarray(p, float)
    cols = [p] + [np.maximum(p - k, 0.0) for k in knots]
    X = np.column_stack(cols)
    return lr.predict_proba(X)[:, 1]


METHODS = {
    "isotonic": (fit_isotonic, apply_isotonic),
    "platt": (fit_platt, apply_platt),
    "piecewise": (fit_piecewise, apply_piecewise),
}


# ---------------------------------------------------------------------------
# 5-fold calibration CV on OOF
# ---------------------------------------------------------------------------

def cv_calibrate(oof_h, y_time, y_event, horizon, method_name):
    """5-fold CV calibration on OOF predictions for one horizon."""
    labels, eligible = build_horizon_labels(y_time, y_event, horizon)
    p_elig = oof_h[eligible]
    y_elig = labels[eligible]
    n_elig = eligible.sum()

    fit_fn, apply_fn = METHODS[method_name]
    cal_probs = np.zeros(n_elig)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf.split(p_elig, y_elig.astype(int)):
        model = fit_fn(p_elig[tr_idx], y_elig[tr_idx])
        cal_probs[va_idx] = apply_fn(model, p_elig[va_idx])

    # Reconstruct full-length array (non-eligible stays as original)
    result = oof_h.copy()
    result[eligible] = cal_probs
    return result


# ---------------------------------------------------------------------------
# Track evaluation
# ---------------------------------------------------------------------------

def eval_track_a(oof_cal_h, oof_base_h, anchor_h, alpha, y_time, y_event, horizon, prob_dict_base):
    """Track A: anchor-incremental blend."""
    # anchor_h is test-set anchor; for OOF eval we use oof_base as proxy for anchor
    new_h = anchor_h + alpha * (oof_cal_h - oof_base_h)
    new_h = np.clip(new_h, EPS, 1.0 - EPS)
    pd_new = dict(prob_dict_base)
    pd_new[horizon] = new_h
    return new_h, pd_new


def eval_track_b(oof_cal_h, y_time, y_event, horizon, prob_dict_base):
    """Track B: independent calibration."""
    pd_new = dict(prob_dict_base)
    pd_new[horizon] = oof_cal_h
    return oof_cal_h, pd_new


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Loading data ===")
    train = add_engineered(pd.read_csv(TRAIN_PATH))
    test = add_engineered(pd.read_csv(TEST_PATH))
    feature_cols = [c for c in FEATURES if c in train.columns]
    print(f"  Features: {len(feature_cols)}")

    print("\n=== Generating OOF (RSF+EST 50/50, 5-fold x 5-seed) ===")
    oof, y_time, y_event = run_oof_blend(train, feature_cols)

    # Baseline OOF score
    score_base, det_base = hybrid_score(y_time, y_event, oof)
    ci_base = det_base["c_index"]
    wb_base = det_base["weighted_brier"]
    print(f"\n  Baseline OOF: Hybrid={score_base:.4f}  CI={ci_base:.4f}  WBrier={wb_base:.4f}")

    # ---------------------------------------------------------------------------
    # Calibration comparison
    # ---------------------------------------------------------------------------
    print("\n=== Calibration Comparison ===")
    print(f"{'Method':12s} {'Track':6s} {'Horizon':8s} {'WBrier':8s} {'CI':8s} {'Hybrid':8s} {'dHybrid':8s}")
    print("-" * 65)

    results = []
    alphas_a = [0.1, 0.2, 0.3]

    for horizon in [24, 48]:
        for method_name in METHODS:
            # CV-calibrated OOF for this horizon
            oof_cal_h = cv_calibrate(oof[horizon], y_time, y_event, horizon, method_name)

            # Track B: independent
            pd_b = dict(oof)
            pd_b[horizon] = oof_cal_h
            score_b, det_b = hybrid_score(y_time, y_event, pd_b)
            row = {
                "method": method_name, "track": "B", "horizon": horizon,
                "hybrid": score_b, "ci": det_b["c_index"],
                "wbrier": det_b["weighted_brier"],
                "dhybrid": score_b - score_base,
                "oof_cal_h": oof_cal_h,
            }
            results.append(row)
            print(f"{method_name:12s} {'B':6s} {horizon:8d} "
                  f"{det_b['weighted_brier']:8.5f} {det_b['c_index']:8.4f} "
                  f"{score_b:8.4f} {score_b - score_base:+8.4f}")

            # Track A: anchor-incremental
            for alpha in alphas_a:
                pd_a = dict(oof)
                new_h = oof[horizon] + alpha * (oof_cal_h - oof[horizon])
                new_h = np.clip(new_h, EPS, 1.0 - EPS)
                pd_a[horizon] = new_h
                score_a, det_a = hybrid_score(y_time, y_event, pd_a)
                row_a = {
                    "method": method_name, "track": f"A{alpha:.1f}", "horizon": horizon,
                    "hybrid": score_a, "ci": det_a["c_index"],
                    "wbrier": det_a["weighted_brier"],
                    "dhybrid": score_a - score_base,
                    "oof_cal_h": new_h,
                    "alpha": alpha,
                }
                results.append(row_a)
                print(f"{method_name:12s} {'A'+str(alpha):6s} {horizon:8d} "
                      f"{det_a['weighted_brier']:8.5f} {det_a['c_index']:8.4f} "
                      f"{score_a:8.4f} {score_a - score_base:+8.4f}")

    # ---------------------------------------------------------------------------
    # Select best: highest hybrid, CI not degraded vs baseline
    # ---------------------------------------------------------------------------
    valid = [r for r in results if r["ci"] >= ci_base - 0.001]
    if not valid:
        valid = results  # fallback: no CI constraint
        print("\n  WARNING: No result passes CI constraint; using all results.")

    best = max(valid, key=lambda r: r["hybrid"])
    print(f"\n=== Best: method={best['method']} track={best['track']} "
          f"horizon={best['horizon']} hybrid={best['hybrid']:.4f} "
          f"dHybrid={best['dhybrid']:+.4f} ===")

    # ---------------------------------------------------------------------------
    # Task 2: Generate best calibration submission
    # ---------------------------------------------------------------------------
    print("\n=== Generating best calibration submission ===")
    anchor = pd.read_csv(ANCHOR_PATH)

    # Retrain calibrator on full OOF for best horizon
    h_best = best["horizon"]
    method_best = best["method"]
    track_best = best["track"]

    labels_full, eligible_full = build_horizon_labels(y_time, y_event, h_best)
    p_elig = oof[h_best][eligible_full]
    y_elig = labels_full[eligible_full]
    fit_fn, apply_fn = METHODS[method_best]
    cal_model = fit_fn(p_elig, y_elig)

    # Retrain RSF+EST on full data → test predictions in same distribution as OOF
    print("  Retraining RSF+EST for test predictions (same distribution as OOF)...")
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(train[feature_cols]), columns=feature_cols)
    X_te = pd.DataFrame(scaler.transform(test[feature_cols]), columns=feature_cols)
    test_blend = {h: np.zeros(len(test)) for h in HORIZONS}
    for seed in SEEDS:
        rsf = RSF(random_state=seed); rsf.fit(X_tr, y_time, y_event)
        est = EST(random_state=seed); est.fit(X_tr, y_time, y_event)
        for h in HORIZONS:
            test_blend[h] += 0.5 * rsf.predict_proba(X_te)[h] + 0.5 * est.predict_proba(X_te)[h]
    for h in HORIZONS:
        test_blend[h] /= len(SEEDS)

    # Apply calibrator to RSF+EST test predictions (same distribution as OOF)
    cal_test_h = apply_fn(cal_model, test_blend[h_best])

    if track_best.startswith("A"):
        alpha_best = best.get("alpha", 0.2)
        anchor_h = anchor[f"prob_{h_best}h"].values
        test_h_new = anchor_h + alpha_best * (cal_test_h - anchor_h)
    else:
        test_h_new = cal_test_h

    test_h_new = np.clip(test_h_new, EPS, 1.0 - EPS)

    # Build submission: calibrated horizon from RSF+EST, others from anchor
    test_preds = {h: anchor[f"prob_{h}h"].values.copy() for h in HORIZONS}
    test_preds[h_best] = test_h_new

    # Enforce monotonicity
    pp = submission_postprocess(test_preds)
    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = pp[h]

    out_path = "submissions/submission_exp32_cal.csv"
    sub.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  Rows: {len(sub)}  Cols: {list(sub.columns)}")

    # Summary comparison
    print("\n=== OOF Summary: baseline vs best calibration ===")
    print(f"  Baseline:  Hybrid={score_base:.4f}  CI={ci_base:.4f}  WBrier={wb_base:.4f}")
    print(f"  Best cal:  Hybrid={best['hybrid']:.4f}  CI={best['ci']:.4f}  WBrier={best['wbrier']:.4f}")
    print(f"  Delta:     dHybrid={best['dhybrid']:+.4f}  dCI={best['ci']-ci_base:+.4f}  dWBrier={best['wbrier']-wb_base:+.5f}")
    print(f"  Config:    method={method_best}  track={track_best}  horizon={h_best}h")

    print("\n=== Exp32 complete ===")


if __name__ == "__main__":
    main()
