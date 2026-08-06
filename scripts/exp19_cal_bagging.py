"""Exp19/20: p48 calibration bagging (lock p12/p24/p72 from anchor).

Exp19: (a,b) logit-linear bagging - bootstrap 1000x, average calibrated p48
Exp20: bin-PAVA bagging - quantile bins + isotonic + interpolation, bootstrap avg
Final: 50/50 blend of Exp19 and Exp20 on p48

Usage:
    python -m scripts.exp19_cal_bagging
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
from src.train import load_data, run_cv, _print_score


ANCHOR_PATH = "submission_0.96624.csv"


# ---------------------------------------------------------------------------
# Exp19: (a,b) logit-linear calibrator
# ---------------------------------------------------------------------------
def logit_ab(probs, a, b, eps=1e-7):
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    return expit(a * sp_logit(p) + b)


def fit_ab(probs, labels, lam=0.1):
    """Fit (a,b) with L2 regularization toward identity."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.size == 0:
        return 1.0, 0.0

    def obj(params):
        a, b = params
        if a <= 0:
            return 1e6
        p2 = logit_ab(p, a, b)
        return float(np.mean((p2 - y) ** 2)) + lam * ((a - 1) ** 2 + b ** 2)

    res = minimize(obj, [1.0, 0.0], method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-10})
    a, b = res.x
    return (float(a), float(b)) if a > 0 else (1.0, 0.0)


def bootstrap_ab_bagging(p_train, y_train, p_test, n_boot=1000, lam=0.1,
                         seed=42):
    """Bootstrap (a,b), apply each to test, return averaged p48."""
    rng = np.random.default_rng(seed)
    n = len(p_train)
    p_test_sum = np.zeros_like(p_test, dtype=float)
    ab_list = []

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        a, b = fit_ab(p_train[idx], y_train[idx], lam=lam)
        ab_list.append((a, b))
        p_test_sum += logit_ab(p_test, a, b)

    p_test_bag = p_test_sum / n_boot
    ab_arr = np.array(ab_list)
    return p_test_bag, ab_arr


# ---------------------------------------------------------------------------
# Exp20: bin-PAVA calibrator
# ---------------------------------------------------------------------------
def pava_increasing(values):
    """Pool Adjacent Violators for non-decreasing sequence."""
    v = list(values)
    w = [1.0] * len(v)
    i = 0
    while i < len(v) - 1:
        if v[i] > v[i + 1]:
            wt = w[i] + w[i + 1]
            v[i] = (w[i] * v[i] + w[i + 1] * v[i + 1]) / wt
            w[i] = wt
            v.pop(i + 1)
            w.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1
    return v, w


def fit_bin_pava(probs, labels, n_bins=10):
    """Fit quantile-binned PAVA calibrator. Returns (bin_centers, cal_values)."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)

    # Quantile bin edges (avoid duplicate edges)
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(p, quantiles))
    if len(edges) < 3:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    bin_idx = np.digitize(p, edges[1:-1])  # 0..n_actual_bins-1
    centers = []
    means = []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        centers.append(p[mask].mean())
        means.append(y[mask].mean())

    if len(centers) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    # PAVA to enforce monotonicity
    pava_vals, _ = pava_increasing(means)

    # Expand back if PAVA merged bins
    cal_vals = []
    j = 0
    for i in range(len(centers)):
        if j < len(pava_vals):
            cal_vals.append(pava_vals[j])
        if i < len(centers) - 1 and j < len(pava_vals) - 1:
            j += 1
    # Ensure lengths match
    while len(cal_vals) < len(centers):
        cal_vals.append(cal_vals[-1])
    cal_vals = cal_vals[:len(centers)]

    return np.array(centers), np.array(cal_vals)


def apply_bin_pava(probs, centers, cal_vals):
    """Apply bin-PAVA mapping via linear interpolation."""
    return np.interp(probs, centers, cal_vals, left=cal_vals[0],
                     right=cal_vals[-1])


def bootstrap_pava_bagging(p_train, y_train, p_test, n_boot=1000,
                           n_bins=10, seed=42):
    """Bootstrap bin-PAVA, apply each to test, return averaged p48."""
    rng = np.random.default_rng(seed)
    n = len(p_train)
    p_test_sum = np.zeros_like(p_test, dtype=float)

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        centers, cal_vals = fit_bin_pava(p_train[idx], y_train[idx],
                                        n_bins=n_bins)
        p_test_sum += apply_bin_pava(p_test, centers, cal_vals)

    return p_test_sum / n_boot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_submission(anchor, p48_new, path):
    """Replace p48 in anchor, postprocess, save."""
    prob_dict = {
        12: anchor["prob_12h"].values.copy(),
        24: anchor["prob_24h"].values.copy(),
        48: p48_new.copy(),
        72: np.ones(len(anchor)),
    }
    pp = submission_postprocess(prob_dict)

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = pp[h]

    h72_ok = (sub["prob_72h"] == 1.0).all()
    print(f"  72h all-ones: {'PASS' if h72_ok else 'FAIL'}  "
          f"Shape: {sub.shape}")
    sub.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # Spearman vs anchor
    print(f"  Spearman vs anchor:")
    for col in PROB_COLS[:-1]:
        sr, _ = spearmanr(sub[col], anchor[col])
        print(f"    {col}: rho={sr:.6f}")
    return sub


def oof_closed_loop(blend, p48_cal, y_time, y_event, label):
    """Evaluate calibrated p48 in closed loop (postprocess then score)."""
    cal = {h: blend[h].copy() for h in HORIZONS}
    cal[48] = p48_cal
    pp = submission_postprocess(cal)
    _print_score(label, pp, y_time, y_event)
    return pp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== Exp19/20: p48 Calibration Bagging ===\n")

    # Load anchor
    anchor = pd.read_csv(ANCHOR_PATH)
    print(f"  Anchor loaded: {ANCHOR_PATH} ({len(anchor)} rows)")

    # Load data + OOF
    train, test = load_data(feature_level="medium")
    from src.features import get_feature_set
    feature_cols = get_feature_set(train, level="medium")
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    print("\n=== Running CV for OOF predictions ===")
    oof = run_cv(train, feature_cols)
    blend = {h: 0.5 * oof["RSF"][h] + 0.5 * oof["EST"][h] for h in HORIZONS}

    pp_base = submission_postprocess(blend)
    _print_score("Baseline (proj)", pp_base, y_time, y_event)

    # 48h eligible
    labels48, elig48 = build_horizon_labels(y_time, y_event, 48)
    p48_elig = blend[48][elig48]
    y48_elig = labels48[elig48]
    n_elig = elig48.sum()
    print(f"\n  48h eligible: n={n_elig}, pos_rate={y48_elig.mean():.3f}")

    # Test p48 from our model (for calibration application)
    from src.stacking import _train_predict_base, HEAD_HORIZONS
    X_train, X_test = train[feature_cols], test[feature_cols]

    SEEDS = [42, 123, 456, 789, 2026]
    print(f"\n=== Full retrain ({len(SEEDS)} seeds) ===")
    base_all = []
    for seed in SEEDS:
        bp = _train_predict_base(X_train, y_time, y_event, X_test, seed=seed)
        base_all.append(bp)
        print(f"    Seed {seed} done")

    # Test blend: RSF:EST 50:50
    test_p48 = np.mean([
        0.5 * bp["RSF"][48] + 0.5 * bp["EST"][48] for bp in base_all
    ], axis=0)
    anchor_p48 = anchor["prob_48h"].values

    print(f"\n  Test p48 stats: min={test_p48.min():.4f} "
          f"median={np.median(test_p48):.4f} max={test_p48.max():.4f}")
    print(f"  Anchor p48 stats: min={anchor_p48.min():.4f} "
          f"median={np.median(anchor_p48):.4f} max={anchor_p48.max():.4f}")
    sr, _ = spearmanr(test_p48, anchor_p48)
    print(f"  Spearman(our_p48, anchor_p48): {sr:.4f}")

    # ==================================================================
    # Exp19: (a,b) bagging
    # ==================================================================
    print("\n" + "=" * 60)
    print("=== Exp19: (a,b) logit-linear bagging (1000 bootstrap) ===")
    print("=" * 60)

    # Bagging on our model's test p48
    p48_ab_bag, ab_dist = bootstrap_ab_bagging(
        p48_elig, y48_elig, test_p48, n_boot=1000, lam=0.1,
    )
    a_vals, b_vals = ab_dist[:, 0], ab_dist[:, 1]
    print(f"  a: median={np.median(a_vals):.4f} "
          f"std={a_vals.std():.4f} "
          f"[p5={np.percentile(a_vals, 5):.4f}, "
          f"p95={np.percentile(a_vals, 95):.4f}]")
    print(f"  b: median={np.median(b_vals):.4f} "
          f"std={b_vals.std():.4f} "
          f"[p5={np.percentile(b_vals, 5):.4f}, "
          f"p95={np.percentile(b_vals, 95):.4f}]")

    # OOF evaluation: apply median (a,b) to OOF blend
    a_med, b_med = np.median(a_vals), np.median(b_vals)
    p48_oof_ab = logit_ab(blend[48], a_med, b_med)
    oof_closed_loop(blend, p48_oof_ab, y_time, y_event,
                    "Exp19 OOF (proj)")

    # Also bagging on anchor's p48
    p48_ab_anchor, _ = bootstrap_ab_bagging(
        p48_elig, y48_elig, anchor_p48, n_boot=1000, lam=0.1,
    )

    # ==================================================================
    # Exp20: bin-PAVA bagging
    # ==================================================================
    print("\n" + "=" * 60)
    print("=== Exp20: bin-PAVA bagging (1000 bootstrap, 10 bins) ===")
    print("=" * 60)

    p48_pava_bag = bootstrap_pava_bagging(
        p48_elig, y48_elig, test_p48, n_boot=1000, n_bins=10,
    )

    # OOF: apply single PAVA to full OOF
    centers, cal_vals = fit_bin_pava(p48_elig, y48_elig, n_bins=10)
    print(f"  PAVA bins: {len(centers)} centers")
    for c, v in zip(centers, cal_vals):
        print(f"    p48={c:.4f} -> cal={v:.4f}")

    p48_oof_pava = apply_bin_pava(blend[48], centers, cal_vals)
    oof_closed_loop(blend, p48_oof_pava, y_time, y_event,
                    "Exp20 OOF (proj)")

    # Also bagging on anchor's p48
    p48_pava_anchor = bootstrap_pava_bagging(
        p48_elig, y48_elig, anchor_p48, n_boot=1000, n_bins=10,
    )

    # ==================================================================
    # Submissions: use our model's test p48 (calibrated)
    # ==================================================================
    print("\n" + "=" * 60)
    print("=== Generating submissions (our model p48, anchor p12/p24) ===")
    print("=" * 60)

    # Exp19 submission
    print("\n--- Exp19: (a,b) bagging ---")
    make_submission(anchor, p48_ab_bag, "submission_exp19_ab_bag.csv")

    # Exp20 submission
    print("\n--- Exp20: bin-PAVA bagging ---")
    make_submission(anchor, p48_pava_bag, "submission_exp20_pava_bag.csv")

    # 50/50 blend
    print("\n--- Exp19+20: 50/50 blend ---")
    p48_blend = 0.5 * p48_ab_bag + 0.5 * p48_pava_bag
    make_submission(anchor, p48_blend, "submission_exp19_20_blend.csv")

    # ==================================================================
    # Also try: calibrate anchor's p48 directly
    # ==================================================================
    print("\n" + "=" * 60)
    print("=== Alt: calibrate anchor p48 directly ===")
    print("=" * 60)

    print("\n--- Exp19-anchor: (a,b) on anchor p48 ---")
    make_submission(anchor, p48_ab_anchor,
                    "submission_exp19_anchor_ab.csv")

    print("\n--- Exp20-anchor: PAVA on anchor p48 ---")
    make_submission(anchor, p48_pava_anchor,
                    "submission_exp20_anchor_pava.csv")

    # Anchor p48 blend
    p48_anchor_blend = 0.5 * p48_ab_anchor + 0.5 * p48_pava_anchor
    print("\n--- Exp19+20-anchor: 50/50 blend ---")
    make_submission(anchor, p48_anchor_blend,
                    "submission_exp19_20_anchor_blend.csv")

    print("\n=== All submissions generated ===")


if __name__ == "__main__":
    main()
