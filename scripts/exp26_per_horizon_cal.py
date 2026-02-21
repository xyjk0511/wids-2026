"""Exp26: Per-horizon decoupled (A,B) calibration + gap-gated push.

Key change vs Exp22: fit separate (A24,B24) and (A48,B48) instead of shared (A,B).

Usage:
    python -m scripts.exp26_per_horizon_cal
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit as sp_logit
from scipy.stats import spearmanr

from src.config import HORIZONS, PROB_COLS, SAMPLE_SUB_PATH, TIME_COL, EVENT_COL
from src.labels import build_horizon_labels
from src.monotonic import submission_postprocess
from src.train import load_data, run_cv, _print_score

ANCHOR_PATH = "submissions/submission_0.96624.csv"
EPS = 1e-7
# Exp22 shared params for comparison
A_SHARED, B_SHARED = 1.0655, -0.0108


def safe_logit(p):
    return sp_logit(np.clip(np.asarray(p, float), EPS, 1.0 - EPS))


def logit_ab_transform(probs, a, b):
    p = np.clip(np.asarray(probs, float), EPS, 1.0 - EPS)
    return expit(a * sp_logit(p) + b)


def fit_logit_ab_brier(probs, labels, lam_a=0.1, lam_b=0.1):
    """Fit (a, b) minimizing regularized Brier: pull toward identity (a=1, b=0)."""
    p = np.asarray(probs, float)
    y = np.asarray(labels, float)
    if p.size == 0:
        return 1.0, 0.0

    def objective(params):
        a, b = params
        if a <= 0:
            return 1e6
        p2 = logit_ab_transform(p, a, b)
        return float(np.mean((p2 - y) ** 2)) + lam_a * (a - 1) ** 2 + lam_b * b ** 2

    res = minimize(objective, x0=[1.0, 0.0], method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-10})
    a, b = res.x
    return (float(a), float(b)) if a > 0 else (1.0, 0.0)


def bootstrap_calibration(probs, labels, n_boot=1000, lam_a=0.1, lam_b=0.1, seed=42):
    rng = np.random.default_rng(seed)
    n = len(probs)
    ab = np.empty((n_boot, 2))
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        ab[i] = fit_logit_ab_brier(probs[idx], labels[idx], lam_a, lam_b)
    return float(np.median(ab[:, 0])), float(np.median(ab[:, 1])), ab


def gap_gated_push_decoupled(p48_a, p24_a, a48, b48, a24, b24,
                              lam, r_hi=1.0, r_lo=0.7,
                              g_lo=0.005, g_hi=0.025):
    """Per-horizon (A,B) — p48 full cal, p24 gap-gated r coupling.

    Matches PB pipeline (exp22f): p48 ungated, p24 r from gap_gated_r.
    """
    gap = p48_a - p24_a
    gate = np.clip((gap - g_lo) / (g_hi - g_lo), 0.0, 1.0)

    # p48: FULL calibration, no gating (same as exp22f)
    lp48 = safe_logit(p48_a)
    lp48_cal = (1.0 + lam * (a48 - 1.0)) * lp48 + lam * b48
    p48_new = expit(lp48_cal)

    # p24: gap-gated r coupling (small gap → r_hi, large gap → r_lo)
    lp24 = safe_logit(p24_a)
    lp24_cal = (1.0 + lam * (a24 - 1.0)) * lp24 + lam * b24
    r_vec = r_hi - (r_hi - r_lo) * gate
    p24_new = expit((1.0 - r_vec) * lp24 + r_vec * lp24_cal)

    return p48_new, p24_new


def make_sub(anchor, p48_new, p24_new, path):
    p48 = np.clip(p48_new, 1e-6, 1.0)
    p24 = np.clip(p24_new, 1e-6, 1.0)
    viol = (p48 < p24 + 1e-7).sum()
    mask = p48 < p24 + 1e-7
    p24[mask] = p48[mask] - 1e-7
    p12 = np.minimum(anchor["prob_12h"].values, p24 - 1e-7)

    pp = submission_postprocess({12: p12, 24: p24, 48: p48, 72: np.ones(len(anchor))})
    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = pp[h]

    sr48, _ = spearmanr(sub["prob_48h"], anchor["prob_48h"])
    sr24, _ = spearmanr(sub["prob_24h"], anchor["prob_24h"])
    sub.to_csv(path, index=False)
    print(f"  {path}")
    print(f"    viol={viol}  rho48={sr48:.6f}  rho24={sr24:.6f}  "
          f"med48={np.median(sub['prob_48h']):.4f}  "
          f"med24={np.median(sub['prob_24h']):.4f}")
    return sub


def main():
    print("=" * 60)
    print("  Exp26v2: Per-Horizon A (OOF) + Negative B (corrected)")
    print("=" * 60)

    anchor = pd.read_csv(ANCHOR_PATH)
    p48_a = anchor["prob_48h"].values
    p24_a = anchor["prob_24h"].values

    # Per-horizon A from OOF bootstrap; B kept negative (PB-validated direction)
    # lam10: regularization=0.1, lam20: regularization=0.2
    configs = [
        # (label, A24, A48, B_shared)
        ("A10_Bpb",   1.0584, 1.0565, -0.0108),
        ("A10_Blo",   1.0584, 1.0565, -0.0130),
        ("A10_Bhi",   1.0584, 1.0565, -0.0085),
        ("A10_B145",  1.0584, 1.0565, -0.0145),
        ("A10_B160",  1.0584, 1.0565, -0.0160),
        ("A10_B175",  1.0584, 1.0565, -0.0175),
        ("A20_Bpb",   1.0328, 1.0318, -0.0108),
        ("A20_Blo",   1.0328, 1.0318, -0.0130),
        ("A20_Bhi",   1.0328, 1.0318, -0.0085),
    ]
    # Fixed gate params (PB config)
    lam, rh, rl, g_lo, g_hi = 6.0, 1.1, 0.7, 0.012, 0.018

    print(f"\n  Gate: lam={lam} rh={rh} rl={rl} gate=[{g_lo},{g_hi}]")
    print(f"  Shared baseline: A={A_SHARED:.4f} B={B_SHARED:.4f}\n")

    for label, a24, a48, b in configs:
        p48_new, p24_new = gap_gated_push_decoupled(
            p48_a, p24_a, a48, b, a24, b,
            lam=lam, r_hi=rh, r_lo=rl, g_lo=g_lo, g_hi=g_hi)
        make_sub(anchor, p48_new, p24_new,
                 f"submissions/submission_exp26v2_{label}.csv")

    # Shared-param baseline for comparison
    print("\n=== Shared-param baseline ===")
    p48_new, p24_new = gap_gated_push_decoupled(
        p48_a, p24_a, A_SHARED, B_SHARED, A_SHARED, B_SHARED,
        lam=lam, r_hi=rh, r_lo=rl, g_lo=g_lo, g_hi=g_hi)
    make_sub(anchor, p48_new, p24_new,
             "submissions/submission_exp26v2_shared_baseline.csv")

    # Regression check: p48 must not depend on gate params
    print("\n=== Regression: p48 independence from gate ===")
    a48_t, b_t = 1.0565, -0.0108
    p48_g1, _ = gap_gated_push_decoupled(
        p48_a, p24_a, a48_t, b_t, 1.0, 0.0,
        lam=lam, r_hi=rh, r_lo=rl, g_lo=g_lo, g_hi=g_hi)
    p48_g2, _ = gap_gated_push_decoupled(
        p48_a, p24_a, a48_t, b_t, 1.0, 0.0,
        lam=lam, r_hi=0.5, r_lo=0.3, g_lo=0.001, g_hi=0.050)
    diff = np.max(np.abs(p48_g1 - p48_g2))
    status = "PASS" if diff < 1e-12 else "FAIL"
    print(f"  max|p48_diff| = {diff:.2e}  [{status}]")
    assert diff < 1e-12, "p48 calibration must not depend on gap-gate params!"

    print("\n=== Exp26v2 complete ===")


if __name__ == "__main__":
    main()
