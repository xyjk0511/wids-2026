"""Exp22: Two-track calibration.

Route 1: Ultra-conservative logit-space shrink toward identity.
  logit(p48_new) = (1-lam)*logit(p48_anchor) + lam*(a*logit(p48_anchor)+b)

Route 2: Anchor distillation - train student on test to mimic anchor p48,
  predict on train to get calibration pairs, then calibrate anchor.

Usage:
    python -m scripts.exp22_two_track
"""

import numpy as np
import pandas as pd
from scipy.special import expit, logit as sp_logit
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

from src.config import (
    HORIZONS, PROB_COLS, TIME_COL, EVENT_COL, SAMPLE_SUB_PATH,
)
from src.labels import build_horizon_labels
from src.evaluation import hybrid_score, horizon_brier_score
from src.monotonic import submission_postprocess
from src.train import load_data, run_cv, _print_score


ANCHOR_PATH = "submission_0.96624.csv"
EPS = 1e-7


# ===================================================================
# Shared helpers
# ===================================================================
def safe_logit(p, eps=EPS):
    """Logit with clipping to avoid inf."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return sp_logit(p)


def safety_check(p_new, p_anchor, label):
    """Print safety metrics; return Spearman."""
    sr, _ = spearmanr(p_new, p_anchor)
    mad = np.mean(np.abs(p_new - p_anchor))
    med_old = np.median(p_anchor)
    med_new = np.median(p_new)
    print(f"  [{label}] Spearman={sr:.6f}  MAD={mad:.6f}  "
          f"median: {med_old:.4f}->{med_new:.4f}")
    if sr < 0.999:
        print(f"    WARNING: Spearman {sr:.6f} < 0.999")
    if med_new < 0.03:
        print(f"    WARNING: median {med_new:.4f} < 0.03 floor")
    return sr


def make_submission(anchor, p48_new, path):
    """Replace p48 in anchor, enforce monotonicity, save."""
    prob_dict = {
        12: anchor["prob_12h"].values.copy(),
        24: anchor["prob_24h"].values.copy(),
        48: np.clip(p48_new.copy(), 1e-6, 1.0),
        72: np.ones(len(anchor)),
    }
    prob_dict[48] = np.maximum(prob_dict[48], prob_dict[24] + 1e-7)
    pp = submission_postprocess(prob_dict)

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = pp[h]

    mono_v = (sub["prob_24h"] > sub["prob_48h"] + 1e-9).sum()
    print(f"  72h={'PASS' if (sub['prob_72h']==1.0).all() else 'FAIL'}  "
          f"mono_viol={mono_v}")
    sub.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # Post-process Spearman vs anchor
    for col in PROB_COLS[:-1]:
        sr, _ = spearmanr(sub[col], anchor[col])
        print(f"    {col}: rho={sr:.6f}")
    return sub


# ===================================================================
# Route 1: Logit-space shrink with bootstrap (a,b)
# ===================================================================
def bootstrap_ab(p_elig, y_elig, n_boot=1000, lam_reg=0.1, seed=42):
    """Bootstrap (a,b) logit-linear calibration on eligible samples.

    Minimizes: mean((sigmoid(a*logit(p)+b) - y)^2) + lam*(a-1)^2 + lam*b^2
    Returns median (a, b) and per-bootstrap arrays.
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    n = len(p_elig)
    logit_p = safe_logit(p_elig)
    a_list, b_list = [], []

    def objective(params, lp, y):
        a, b = params
        pred = expit(a * lp + b)
        brier = np.mean((pred - y) ** 2)
        reg = lam_reg * ((a - 1.0) ** 2 + b ** 2)
        return brier + reg

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        res = minimize(objective, x0=[1.0, 0.0],
                       args=(logit_p[idx], y_elig[idx]),
                       method="Nelder-Mead",
                       options={"maxiter": 500, "xatol": 1e-6})
        a_list.append(res.x[0])
        b_list.append(res.x[1])

    a_arr = np.array(a_list)
    b_arr = np.array(b_list)
    a_med = np.median(a_arr)
    b_med = np.median(b_arr)
    print(f"  Bootstrap (a,b): a={a_med:.4f} (std={a_arr.std():.4f})  "
          f"b={b_med:.4f} (std={b_arr.std():.4f})")
    return a_med, b_med, a_arr, b_arr


def apply_logit_shrink(p_anchor, a, b, lam):
    """Apply logit-space shrink: logit(p_new) = (1-lam)*logit(p) + lam*(a*logit(p)+b)."""
    lp = safe_logit(p_anchor)
    lp_cal = a * lp + b
    lp_new = (1.0 - lam) * lp + lam * lp_cal
    return expit(lp_new)


def route1(anchor_p48, p48_elig, y48_elig):
    """Route 1: Ultra-conservative logit-space shrink."""
    print("\n" + "=" * 60)
    print("  ROUTE 1: Logit-Space Shrink (near-identity)")
    print("=" * 60)

    a_med, b_med, _, _ = bootstrap_ab(p48_elig, y48_elig)

    lambdas = [0.05, 0.10, 0.20]
    results = {}
    for lam in lambdas:
        print(f"\n--- lambda={lam:.2f} ---")
        p48_new = apply_logit_shrink(anchor_p48, a_med, b_med, lam)
        p48_new = np.clip(p48_new, 1e-6, 1.0)
        sr = safety_check(p48_new, anchor_p48, f"lam={lam}")
        results[lam] = (p48_new, sr)

    return a_med, b_med, results


# ===================================================================
# Route 2: Anchor distillation
# ===================================================================
def train_student_a(X_test, anchor_p48, feature_cols):
    """Student A: Monotone single-score model.

    Uses IsotonicRegression on log(dist_min) to fit anchor p48.
    Fallback to linear if isotonic fails.
    """
    from sklearn.isotonic import IsotonicRegression

    # Primary feature: log(dist_min) — strongest predictor, monotone with p48
    dist_col = "dist_min_ci_0_5h"
    if dist_col not in X_test.columns:
        print("  [WARN] dist_min_ci_0_5h not in test features")
        return None, None

    log_dist = np.log1p(X_test[dist_col].values)
    # p48 should decrease as distance increases → use negative log_dist
    score = -log_dist

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(score, anchor_p48)
    p48_fit = iso.predict(score)

    sr, _ = spearmanr(p48_fit, anchor_p48)
    print(f"  Student A (isotonic on -log_dist): Spearman={sr:.6f}  "
          f"median={np.median(p48_fit):.4f}")
    return iso, score


def train_student_b(X_test, anchor_p48):
    """Student B: 3-feature linear model in logit space.

    logit(p48) ~ w1*log_dist + w2*log_area + w3*month
    Ultra-low capacity to avoid overfitting on 96 test samples.
    """
    from sklearn.linear_model import Ridge

    dist = np.log1p(X_test["dist_min_ci_0_5h"].values)
    area = np.log1p(X_test["area_first_ha"].values)
    month = X_test["event_start_month"].values / 12.0

    feats = np.column_stack([dist, area, month])
    target = safe_logit(anchor_p48)

    # Very strong regularization (alpha=10) for 96 samples, 3 features
    model = Ridge(alpha=10.0)
    model.fit(feats, target)
    pred_logit = model.predict(feats)
    pred_p = expit(pred_logit)

    sr, _ = spearmanr(pred_p, anchor_p48)
    print(f"  Student B (Ridge logit, 3 feat): Spearman={sr:.6f}  "
          f"median={np.median(pred_p):.4f}")
    print(f"    coefs: dist={model.coef_[0]:.4f}  area={model.coef_[1]:.4f}  "
          f"month={model.coef_[2]:.4f}  intercept={model.intercept_:.4f}")
    return model, feats


def predict_student_b_on_train(model, X_train):
    """Apply Student B to train data."""
    dist = np.log1p(X_train["dist_min_ci_0_5h"].values)
    area = np.log1p(X_train["area_first_ha"].values)
    month = X_train["event_start_month"].values / 12.0
    feats = np.column_stack([dist, area, month])
    return expit(model.predict(feats))


def calibrate_from_student(p_student_train, y_time, y_event, anchor_p48,
                           n_boot=1000, seed=42):
    """Learn calibration on (p_student_train, y) pairs, apply to anchor.

    Uses bootstrap (a,b) logit-linear with strong regularization.
    """
    labels48, elig48 = build_horizon_labels(y_time, y_event, 48)
    p_elig = p_student_train[elig48]
    y_elig = labels48[elig48]
    n_elig = elig48.sum()
    print(f"  Student-train 48h eligible: n={n_elig}, "
          f"pos_rate={y_elig.mean():.3f}")
    print(f"  Student-train p48: min={p_elig.min():.4f}  "
          f"med={np.median(p_elig):.4f}  max={p_elig.max():.4f}")

    # Distribution comparison
    sr_dist, _ = spearmanr(p_student_train, np.arange(len(p_student_train)))
    print(f"  Student-train full: med={np.median(p_student_train):.4f}  "
          f"anchor med={np.median(anchor_p48):.4f}")

    # Bootstrap (a,b) on student's train predictions
    a_med, b_med, _, _ = bootstrap_ab(p_elig, y_elig, n_boot=n_boot,
                                       lam_reg=0.2, seed=seed)

    # Apply calibration to anchor p48 with shrinkage
    results = {}
    for lam in [0.10, 0.20, 0.30]:
        print(f"\n  --- Route 2 lambda={lam:.2f} ---")
        p48_new = apply_logit_shrink(anchor_p48, a_med, b_med, lam)
        p48_new = np.clip(p48_new, 1e-6, 1.0)
        sr = safety_check(p48_new, anchor_p48, f"R2_lam={lam}")
        results[lam] = (p48_new, sr)

    return a_med, b_med, results


def route2(anchor_p48, X_test, X_train, y_time, y_event, feature_cols):
    """Route 2: Anchor distillation pipeline."""
    print("\n" + "=" * 60)
    print("  ROUTE 2: Anchor Distillation")
    print("=" * 60)

    # --- Student A: Isotonic on -log(dist) ---
    print("\n--- Student A: Isotonic Regression ---")
    iso_model, test_score_a = train_student_a(X_test, anchor_p48, feature_cols)

    if iso_model is not None:
        # Predict on train
        log_dist_train = np.log1p(X_train["dist_min_ci_0_5h"].values)
        train_score_a = -log_dist_train
        p48_student_a_train = iso_model.predict(train_score_a)
        print(f"  Student A on train: med={np.median(p48_student_a_train):.4f}  "
              f"min={p48_student_a_train.min():.4f}  max={p48_student_a_train.max():.4f}")

        print("\n  Calibrating via Student A predictions...")
        a_a, b_a, results_a = calibrate_from_student(
            p48_student_a_train, y_time, y_event, anchor_p48,
        )
    else:
        results_a = {}

    # --- Student B: Ridge logit (3 features) ---
    print("\n--- Student B: Ridge Logit (3 features) ---")
    ridge_model, test_feats_b = train_student_b(X_test, anchor_p48)

    p48_student_b_train = predict_student_b_on_train(ridge_model, X_train)
    print(f"  Student B on train: med={np.median(p48_student_b_train):.4f}  "
          f"min={p48_student_b_train.min():.4f}  max={p48_student_b_train.max():.4f}")

    print("\n  Calibrating via Student B predictions...")
    a_b, b_b, results_b = calibrate_from_student(
        p48_student_b_train, y_time, y_event, anchor_p48,
    )

    return results_a, results_b


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print("  Exp22: Two-Track Calibration")
    print("=" * 60)

    anchor = pd.read_csv(ANCHOR_PATH)
    anchor_p48 = anchor["prob_48h"].values.copy()
    print(f"\n  Anchor: {ANCHOR_PATH} ({len(anchor)} rows)")
    print(f"  Anchor p48: min={anchor_p48.min():.4f}  "
          f"med={np.median(anchor_p48):.4f}  max={anchor_p48.max():.4f}")

    # Load data
    train, test = load_data(feature_level="medium")
    from src.features import get_feature_set
    feature_cols = get_feature_set(train, level="medium")
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    # OOF for Route 1 calibration fitting
    print("\n=== Running CV for OOF ===")
    oof = run_cv(train, feature_cols)
    blend = {h: 0.5 * oof["RSF"][h] + 0.5 * oof["EST"][h] for h in HORIZONS}
    pp_base = submission_postprocess(blend)
    _print_score("Baseline (proj)", pp_base, y_time, y_event)

    # 48h eligible for Route 1
    labels48, elig48 = build_horizon_labels(y_time, y_event, 48)
    p48_elig = blend[48][elig48]
    y48_elig = labels48[elig48]
    print(f"\n  48h eligible: n={elig48.sum()}, pos_rate={y48_elig.mean():.3f}")

    # ==================================================================
    # Route 1
    # ==================================================================
    a_med, b_med, r1_results = route1(anchor_p48, p48_elig, y48_elig)

    # Generate Route 1 submissions
    print("\n=== Route 1 Submissions ===")
    for lam in [0.05, 0.10]:
        if lam in r1_results:
            p48_new, sr = r1_results[lam]
            if sr >= 0.998:
                tag = f"lam{int(lam*100):02d}"
                make_submission(anchor, p48_new,
                                f"submission_exp22_r1_{tag}.csv")
            else:
                print(f"  SKIP lam={lam}: Spearman {sr:.6f} below 0.998")

    # ==================================================================
    # Route 2
    # ==================================================================
    X_train = train[feature_cols]
    X_test = test[feature_cols]
    results_a, results_b = route2(
        anchor_p48, X_test, X_train, y_time, y_event, feature_cols,
    )

    # Generate Route 2 submissions (best safe option from each student)
    print("\n=== Route 2 Submissions ===")
    for student_label, results in [("A", results_a), ("B", results_b)]:
        # Pick the largest lambda that passes safety
        for lam in sorted(results.keys()):
            p48_new, sr = results[lam]
            if sr >= 0.998:
                tag = f"s{student_label}_lam{int(lam*100):02d}"
                make_submission(anchor, p48_new,
                                f"submission_exp22_r2_{tag}.csv")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Route 1 (a,b): a={a_med:.4f}, b={b_med:.4f}")
    for lam, (p, sr) in sorted(r1_results.items()):
        print(f"    lam={lam:.2f}: Spearman={sr:.6f}  "
              f"med={np.median(p):.4f}")
    print(f"  Route 2A (isotonic student):")
    for lam, (p, sr) in sorted(results_a.items()):
        print(f"    lam={lam:.2f}: Spearman={sr:.6f}  "
              f"med={np.median(p):.4f}")
    print(f"  Route 2B (ridge student):")
    for lam, (p, sr) in sorted(results_b.items()):
        print(f"    lam={lam:.2f}: Spearman={sr:.6f}  "
              f"med={np.median(p):.4f}")

    print("\n=== Exp22 complete ===")


if __name__ == "__main__":
    main()
