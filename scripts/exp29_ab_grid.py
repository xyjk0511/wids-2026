"""Exp29: Fine grid search over (A, B) at PB gate config.

Current PB: A=1.0655, B=-0.0108, lam=6.0, rh=1.1, rl=0.7, gate=[0.012,0.018]
Search: A in [1.04, 1.10], B in [-0.025, 0.005], generate candidates for LB testing.
"""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.special import expit, logit as sp_logit
from scipy.stats import spearmanr
from src.config import HORIZONS, PROB_COLS, SAMPLE_SUB_PATH
from src.monotonic import submission_postprocess

ANCHOR_PATH = "submissions/submission_0.96624.csv"
EPS = 1e-7

def safe_logit(p):
    return sp_logit(np.clip(np.asarray(p, float), EPS, 1.0 - EPS))

def calibrate(p48_a, p24_a, a, b, lam, rh, rl, g_lo, g_hi):
    gap = p48_a - p24_a
    gate = np.clip((gap - g_lo) / (g_hi - g_lo), 0.0, 1.0)
    lp48, lp24 = safe_logit(p48_a), safe_logit(p24_a)
    cal48, cal24 = a * lp48 + b, a * lp24 + b
    lam_eff = lam * gate
    p48_new = expit((1 - lam_eff) * lp48 + lam_eff * cal48)
    r = rh - (rh - rl) * gate
    lam24 = lam * r * gate
    p24_new = expit((1 - lam24) * lp24 + lam24 * cal24)
    return p48_new, p24_new

def make_sub(anchor, p48_new, p24_new, path=None):
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
    med48 = np.median(sub["prob_48h"])
    if path:
        sub.to_csv(path, index=False)
    return sub, viol, sr48, med48

def main():
    anchor = pd.read_csv(ANCHOR_PATH)
    p48_a = anchor["prob_48h"].values
    p24_a = anchor["prob_24h"].values

    # PB gate config
    lam, rh, rl, g_lo, g_hi = 6.0, 1.1, 0.7, 0.012, 0.018

    # Grid search
    A_vals = np.arange(1.04, 1.11, 0.005)
    B_vals = np.arange(-0.025, 0.006, 0.005)

    print(f"{'A':>7} {'B':>7} {'viol':>5} {'rho48':>8} {'med48':>7}")
    print("-" * 40)

    results = []
    for a in A_vals:
        for b in B_vals:
            p48_new, p24_new = calibrate(p48_a, p24_a, a, b, lam, rh, rl, g_lo, g_hi)
            _, viol, sr48, med48 = make_sub(anchor, p48_new, p24_new)
            print(f"{a:7.4f} {b:7.4f} {viol:5d} {sr48:8.6f} {med48:7.4f}")
            results.append((a, b, viol, sr48, med48))

    # Save top candidates (low violations, diverse med48)
    print("\n=== Saving top candidates ===")
    targets = [
        (1.0655, -0.0108, "pb_repro"),   # PB reproduction
        (1.07, -0.015, "a107_b015"),
        (1.08, -0.020, "a108_b020"),
        (1.06, -0.005, "a106_b005"),
        (1.05, -0.015, "a105_b015"),
    ]
    for a, b, tag in targets:
        p48_new, p24_new = calibrate(p48_a, p24_a, a, b, lam, rh, rl, g_lo, g_hi)
        sub, viol, sr48, med48 = make_sub(
            anchor, p48_new, p24_new,
            f"submissions/submission_exp29_{tag}.csv")
        print(f"  {tag}: A={a} B={b} viol={viol} rho48={sr48:.6f} med48={med48:.4f}")

    print("\n=== Exp29 complete ===")

if __name__ == "__main__":
    main()
