"""Exp22f round 2: Robustness check + small grid for gap-gated p24.

Phase 1: Bootstrap robustness (local, no submissions)
Phase 2: Small grid (8-12 submissions)

Usage:
    python -m scripts.exp22f_piecewise_gated_r2
"""

import numpy as np
import pandas as pd
from scipy.special import expit, logit as sp_logit
from scipy.stats import spearmanr

from src.config import HORIZONS, PROB_COLS, SAMPLE_SUB_PATH
from src.monotonic import submission_postprocess

ANCHOR_PATH = "submission_0.96624.csv"
EPS = 1e-7
A, B = 1.0655, -0.0108


def safe_logit(p):
    return sp_logit(np.clip(np.asarray(p, float), EPS, 1.0 - EPS))


def apply_coupled_push(p48, p24, lam, r_vec):
    """Apply logit-space push to p48 (full lam) and p24 (r_vec * lam)."""
    lp48 = safe_logit(p48)
    lp24 = safe_logit(p24)
    lp48_cal = (1.0 + lam * (A - 1.0)) * lp48 + lam * B
    lp24_cal = (1.0 + lam * (A - 1.0)) * lp24 + lam * B
    p48_new = expit(lp48_cal)
    p24_new = expit((1.0 - r_vec) * lp24 + r_vec * lp24_cal)
    return p48_new, p24_new


def gap_gated_r(gap, r_hi, r_lo, g_lo, g_hi):
    gate = np.clip((gap - g_lo) / (g_hi - g_lo), 0.0, 1.0)
    return r_hi - (r_hi - r_lo) * gate


def make_sub(anchor, p48_new, p24_new, path=None):
    p48 = np.clip(p48_new.copy(), 1e-6, 1.0)
    p24 = np.clip(p24_new.copy(), 1e-6, 1.0)
    viol = (p48 < p24 + 1e-7).sum()
    mask = p48 < p24 + 1e-7
    p24[mask] = p48[mask] - 1e-7
    p12 = np.minimum(anchor["prob_12h"].values.copy(), p24 - 1e-7)
    pp = submission_postprocess({12: p12, 24: p24, 48: p48, 72: np.ones(len(anchor))})
    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = pp[h]
    if path:
        sub.to_csv(path, index=False)
    sr48, _ = spearmanr(sub["prob_48h"], anchor["prob_48h"])
    mad48 = np.mean(np.abs(sub["prob_48h"] - anchor["prob_48h"]))
    return sub, {"viol": viol, "rho48": sr48, "MAD48": mad48}


def main():
    anchor = pd.read_csv(ANCHOR_PATH)
    p48_a = anchor["prob_48h"].values.copy()
    p24_a = anchor["prob_24h"].values.copy()
    gap = p48_a - p24_a
    n = len(p48_a)

    # ================================================================
    # Phase 1: Bootstrap robustness — compare uniform r=0.7 vs gated
    # ================================================================
    print("=" * 60)
    print("  Phase 1: Bootstrap Robustness (lam=4.0)")
    print("=" * 60)

    lam = 4.0
    n_boot = 500
    np.random.seed(42)

    deltas = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        p48_b, p24_b, gap_b = p48_a[idx], p24_a[idx], gap[idx]

        # Baseline: uniform r=0.7
        r_uni = np.full(n, 0.7)
        p48_u, p24_u = apply_coupled_push(p48_b, p24_b, lam, r_uni)

        # Gated: r_hi=0.9, r_lo=0.7
        r_g = gap_gated_r(gap_b, 0.9, 0.7, 0.005, 0.025)
        p48_g, p24_g = apply_coupled_push(p48_b, p24_b, lam, r_g)

        # Metric: mean absolute diff from anchor (lower = more push)
        mad_u = np.mean(np.abs(p48_u - p48_b))
        mad_g = np.mean(np.abs(p48_g - p48_b))

        # Violations
        v_u = (p48_u < p24_u + 1e-7).sum()
        v_g = (p48_g < p24_g + 1e-7).sum()

        deltas.append(v_u - v_g)  # positive = gated has fewer violations

    deltas = np.array(deltas)
    print(f"  Bootstrap violation reduction (uniform - gated):")
    print(f"    mean={deltas.mean():.2f}  median={np.median(deltas):.1f}  "
          f"std={deltas.std():.2f}")
    print(f"    P(gated fewer viol)={100*(deltas > 0).mean():.1f}%  "
          f"P(same)={100*(deltas == 0).mean():.1f}%")

    # Leave-one-out sensitivity
    print("\n  Leave-one-out: which samples drive the difference?")
    r_uni = np.full(n, 0.7)
    r_gated = gap_gated_r(gap, 0.9, 0.7, 0.005, 0.025)

    p48_u, p24_u = apply_coupled_push(p48_a, p24_a, lam, r_uni)
    p48_g, p24_g = apply_coupled_push(p48_a, p24_a, lam, r_gated)

    diff_full = p24_g - p24_u  # per-sample p24 difference
    affected = np.abs(diff_full) > 1e-6
    print(f"    Samples affected: {affected.sum()}/{n}")
    if affected.sum() > 0:
        print(f"    Affected gap range: [{gap[affected].min():.4f}, "
              f"{gap[affected].max():.4f}]")
        print(f"    Mean p24 shift (affected): {diff_full[affected].mean():.6f}")
        print(f"    r range (affected): [{r_gated[affected].min():.3f}, "
              f"{r_gated[affected].max():.3f}]")

    # ================================================================
    # Phase 2: Small grid — 12 configs
    # ================================================================
    print("\n" + "=" * 60)
    print("  Phase 2: Submission Grid")
    print("=" * 60)

    configs = []
    # r_hi x r_lo x gap_width x lambda
    for lam_val in [4.0, 4.5]:
        for r_hi, r_lo in [(1.0, 0.7), (1.1, 0.7), (1.2, 0.7),
                           (0.9, 0.65), (1.0, 0.65), (1.1, 0.65)]:
            for g_lo, g_hi, gtag in [(0.005, 0.025, "mid")]:
                configs.append((lam_val, r_hi, r_lo, g_lo, g_hi, gtag))

    print(f"\n{'label':>35} {'viol':>5} {'rho48':>10} {'MAD48':>8}")
    print("-" * 65)

    for lam_val, r_hi, r_lo, g_lo, g_hi, gtag in configs:
        r_vec = gap_gated_r(gap, r_hi, r_lo, g_lo, g_hi)
        p48_new, p24_new = apply_coupled_push(p48_a, p24_a, lam_val, r_vec)
        label = f"lam{lam_val:.0f}_rh{r_hi*10:.0f}_rl{r_lo*10:.0f}"
        _, metrics = make_sub(anchor, p48_new, p24_new)
        tag = " ***" if metrics["viol"] == 0 and metrics["rho48"] > 0.999 else ""
        print(f"{label:>35} {metrics['viol']:5d} {metrics['rho48']:10.6f} "
              f"{metrics['MAD48']:8.4f}{tag}")

    # Generate top candidates as CSV
    print("\n=== Generating submissions ===")
    top_configs = [
        (4.0, 1.0, 0.7,  "lam40_rh10_rl07"),
        (4.0, 1.1, 0.7,  "lam40_rh11_rl07"),
        (4.0, 1.2, 0.7,  "lam40_rh12_rl07"),
        (4.5, 1.0, 0.7,  "lam45_rh10_rl07"),
        (4.5, 1.1, 0.7,  "lam45_rh11_rl07"),
        (4.5, 1.2, 0.7,  "lam45_rh12_rl07"),
    ]
    for lam_val, r_hi, r_lo, label in top_configs:
        r_vec = gap_gated_r(gap, r_hi, r_lo, 0.005, 0.025)
        p48_new, p24_new = apply_coupled_push(p48_a, p24_a, lam_val, r_vec)
        path = f"submission_exp22f_{label}.csv"
        sub, m = make_sub(anchor, p48_new, p24_new, path)
        print(f"  {path}  viol={m['viol']}  rho48={m['rho48']:.6f}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
