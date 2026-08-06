"""Exp18b: 2-parameter monotone calibration (a*logit+b) with bootstrap median.

Core idea: odds_scale (1-param) lacks intercept. This adds it while keeping
monotonicity (a>0). Strong regularization + bootstrap median tames variance
on n=166 eligible samples.

Closed-loop evaluation: calibrate -> postprocess -> Brier.

Usage:
    python -m scripts.exp18b_2param_cal
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit as sp_logit
from scipy.stats import spearmanr

from src.config import (
    HORIZONS, PROB_COLS, TIME_COL, EVENT_COL, SAMPLE_SUB_PATH,
)
from src.labels import build_horizon_labels
from src.evaluation import (
    hybrid_score, horizon_brier_score, weighted_brier_score,
)
from src.monotonic import submission_postprocess
from src.train import load_data, run_cv, _strat_labels, _print_score


# ---------------------------------------------------------------------------
# 2-parameter monotone calibrator: p' = sigmoid(a * logit(p) + b), a > 0
# ---------------------------------------------------------------------------
def logit_ab_transform(probs, a, b, eps=1e-7):
    """Apply p' = sigmoid(a * logit(p) + b). Monotone when a > 0."""
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    return expit(a * sp_logit(p) + b)


def fit_logit_ab_brier(probs, labels, lam_a=0.1, lam_b=0.1):
    """Fit (a, b) minimizing regularized Brier on logit scale.

    Loss = mean((sigmoid(a*logit(p)+b) - y)^2)
           + lam_a*(a-1)^2 + lam_b*b^2

    Regularization pulls toward identity (a=1, b=0).
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.size == 0:
        return 1.0, 0.0

    def objective(params):
        a, b = params[0], params[1]
        if a <= 0:
            return 1e6
        p2 = logit_ab_transform(p, a, b)
        brier = float(np.mean((p2 - y) ** 2))
        reg = lam_a * (a - 1.0) ** 2 + lam_b * b ** 2
        return brier + reg

    res = minimize(
        objective, x0=[1.0, 0.0],
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-10},
    )
    a, b = res.x
    if a <= 0:
        return 1.0, 0.0
    return float(a), float(b)


def bootstrap_calibration(probs, labels, n_boot=1000, lam_a=0.1, lam_b=0.1,
                          seed=42):
    """Bootstrap (a, b) and return median estimates + full distribution."""
    rng = np.random.default_rng(seed)
    n = len(probs)
    ab_samples = np.empty((n_boot, 2))

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        a, b = fit_logit_ab_brier(probs[idx], labels[idx],
                                  lam_a=lam_a, lam_b=lam_b)
        ab_samples[i] = [a, b]

    a_med = float(np.median(ab_samples[:, 0]))
    b_med = float(np.median(ab_samples[:, 1]))
    return a_med, b_med, ab_samples


# ---------------------------------------------------------------------------
# Closed-loop evaluation: calibrate -> postprocess -> score
# ---------------------------------------------------------------------------
def closed_loop_eval(blend, y_time, y_event, a, b, horizon=48):
    """Calibrate p_{horizon}, run postprocess, return full metrics."""
    cal = {h: blend[h].copy() for h in HORIZONS}
    cal[horizon] = logit_ab_transform(blend[horizon], a, b)

    pp_pre = submission_postprocess(blend)
    pp_post = submission_postprocess(cal)

    b48_pre = horizon_brier_score(y_time, y_event, pp_pre[48], 48)
    b48_post = horizon_brier_score(y_time, y_event, pp_post[48], 48)
    b24_pre = horizon_brier_score(y_time, y_event, pp_pre[24], 24)
    b24_post = horizon_brier_score(y_time, y_event, pp_post[24], 24)
    wb_pre = weighted_brier_score(y_time, y_event, pp_pre)
    wb_post = weighted_brier_score(y_time, y_event, pp_post)

    return {
        "b48_pre": b48_pre, "b48_post": b48_post,
        "b24_pre": b24_pre, "b24_post": b24_post,
        "wb_pre": wb_pre, "wb_post": wb_post,
        "pp_post": pp_post, "pp_pre": pp_pre,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== Exp18b: 2-param monotone calibration (a*logit+b) ===\n")

    train, test = load_data(feature_level="medium")
    from src.features import get_feature_set
    feature_cols = get_feature_set(train, level="medium")

    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    print("=== Running CV for OOF predictions ===")
    oof = run_cv(train, feature_cols)

    blend = {h: 0.5 * oof["RSF"][h] + 0.5 * oof["EST"][h] for h in HORIZONS}
    pp_base = submission_postprocess(blend)
    _print_score("Baseline (proj)", pp_base, y_time, y_event)

    # Get eligible 48h samples
    labels48, elig48 = build_horizon_labels(y_time, y_event, 48)
    p48_elig = blend[48][elig48]
    y48_elig = labels48[elig48]
    print(f"\n  48h eligible: n={elig48.sum()}, "
          f"pos_rate={y48_elig.mean():.3f}")

    # ----- Lambda sweep: find best regularization strength -----
    print("\n=== Lambda sweep (on full eligible, diagnostic only) ===")
    for lam in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        a, b = fit_logit_ab_brier(p48_elig, y48_elig, lam_a=lam, lam_b=lam)
        p_cal = logit_ab_transform(p48_elig, a, b)
        brier_raw = float(np.mean((p48_elig - y48_elig) ** 2))
        brier_cal = float(np.mean((p_cal - y48_elig) ** 2))
        print(f"  lam={lam:.2f}: a={a:.4f} b={b:.4f} "
              f"Brier {brier_raw:.6f}->{brier_cal:.6f} "
              f"(delta={brier_cal - brier_raw:+.6f})")

    # ----- Bootstrap calibration with selected lambda -----
    # Use moderate regularization: strong enough to tame variance,
    # weak enough to let signal through
    LAM = 0.1
    print(f"\n=== Bootstrap calibration (lam={LAM}, n_boot=1000) ===")
    a_med, b_med, ab_dist = bootstrap_calibration(
        p48_elig, y48_elig, n_boot=1000, lam_a=LAM, lam_b=LAM,
    )

    # Distribution summary
    a_vals, b_vals = ab_dist[:, 0], ab_dist[:, 1]
    print(f"  a: median={a_med:.4f}  "
          f"p5={np.percentile(a_vals, 5):.4f}  "
          f"p95={np.percentile(a_vals, 95):.4f}  "
          f"std={a_vals.std():.4f}")
    print(f"  b: median={b_med:.4f}  "
          f"p5={np.percentile(b_vals, 5):.4f}  "
          f"p95={np.percentile(b_vals, 95):.4f}  "
          f"std={b_vals.std():.4f}")

    # ----- Closed-loop evaluation -----
    print("\n=== Closed-loop evaluation (calibrate->postprocess->Brier) ===")
    res = closed_loop_eval(blend, y_time, y_event, a_med, b_med, horizon=48)

    print(f"  B48:    {res['b48_pre']:.6f} -> {res['b48_post']:.6f} "
          f"(delta={res['b48_post'] - res['b48_pre']:+.6f})")
    print(f"  B24:    {res['b24_pre']:.6f} -> {res['b24_post']:.6f} "
          f"(delta={res['b24_post'] - res['b24_pre']:+.6f})")
    print(f"  WBrier: {res['wb_pre']:.6f} -> {res['wb_post']:.6f} "
          f"(delta={res['wb_post'] - res['wb_pre']:+.6f})")

    if res['b24_post'] > res['b24_pre'] + 0.0005:
        print("  WARNING: B24 degraded > 0.0005")

    _print_score("After cal (proj)", res["pp_post"], y_time, y_event)

    # ----- Bootstrap Brier improvement distribution -----
    print("\n=== Bootstrap Brier@48 improvement distribution ===")
    brier_deltas = np.empty(1000)
    for i in range(1000):
        a_i, b_i = ab_dist[i]
        r = closed_loop_eval(blend, y_time, y_event, a_i, b_i, horizon=48)
        brier_deltas[i] = r["b48_post"] - r["b48_pre"]

    pct_improve = (brier_deltas < 0).mean() * 100
    print(f"  median delta: {np.median(brier_deltas):+.6f}")
    print(f"  mean delta:   {brier_deltas.mean():+.6f}")
    print(f"  p5={np.percentile(brier_deltas, 5):+.6f}  "
          f"p95={np.percentile(brier_deltas, 95):+.6f}")
    print(f"  % improving:  {pct_improve:.1f}%")

    if np.median(brier_deltas) >= -0.0002:
        print("\n  STOP: median Brier@48 improvement < 0.0002. "
              "Calibration not worth pursuing.")
        print("  Recommendation: move to submission bagging.")
        return

    # ----- Generate test submission -----
    print("\n=== Generating test submission ===")
    from src.stacking import _train_predict_base, HEAD_HORIZONS

    X_train = train[feature_cols]
    X_test = test[feature_cols]

    SEEDS = [42, 123, 456, 789, 2026]
    print(f"  Full retrain ({len(SEEDS)} seeds)...")
    base_all = []
    for seed in SEEDS:
        bp = _train_predict_base(X_train, y_time, y_event, X_test, seed=seed)
        base_all.append(bp)
        print(f"    Seed {seed} done")

    base_avg = {}
    for name in base_all[0]:
        base_avg[name] = {
            h: np.mean([bp[name][h] for bp in base_all], axis=0)
            for h in HEAD_HORIZONS
        }

    test_blend = {}
    for h in HORIZONS:
        if h in HEAD_HORIZONS and "RSF" in base_avg and "EST" in base_avg:
            test_blend[h] = 0.5 * base_avg["RSF"][h] + 0.5 * base_avg["EST"][h]
        else:
            test_blend[h] = np.ones(len(test))

    # Apply median (a, b) to test p48
    test_blend[48] = logit_ab_transform(test_blend[48], a_med, b_med)
    pp_test = submission_postprocess(test_blend)

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = pp_test[h]

    h72_ok = (sub["prob_72h"] == 1.0).all()
    print(f"  72h all-ones: {'PASS' if h72_ok else 'FAIL'}  "
          f"Shape: {sub.shape}")

    path = "submission_exp18b_2param.csv"
    sub.to_csv(path, index=False)
    print(f"  Saved: {path}")

    try:
        ref = pd.read_csv("submission_0.96624.csv")
        print(f"  Spearman vs 0.96624:")
        for col in PROB_COLS[:-1]:
            sr, _ = spearmanr(sub[col], ref[col])
            print(f"    {col}: rho={sr:.6f}")
    except FileNotFoundError:
        pass

    print("\n=== Exp18b complete ===")


if __name__ == "__main__":
    main()
