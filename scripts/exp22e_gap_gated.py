"""Exp22e: Gap-gated coupled push.

Instead of uniform lambda, scale push by per-sample gap(p48-p24).
Large-gap samples get full push, tight-gap samples get reduced push.
This breaks the lam=4~5 plateau by pushing harder where safe.

Usage:
    python -m scripts.exp22e_gap_gated
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


def gap_gated_push(p48_a, p24_a, a, b, lam, r, gate_lo=0.005, gate_hi=0.020):
    """Per-sample lambda scaled by gap.

    gate(gap) = clip((gap - gate_lo) / (gate_hi - gate_lo), 0, 1)
    lam_eff[i] = lam * gate[i]
    """
    gap = p48_a - p24_a
    gate = np.clip((gap - gate_lo) / (gate_hi - gate_lo), 0.0, 1.0)

    lp48 = safe_logit(p48_a)
    lp24 = safe_logit(p24_a)
    cal48 = a * lp48 + b
    cal24 = a * lp24 + b

    lam_eff = lam * gate
    p48_new = expit((1.0 - lam_eff) * lp48 + lam_eff * cal48)

    lam24_eff = lam * r * gate
    p24_new = expit((1.0 - lam24_eff) * lp24 + lam24_eff * cal24)

    return p48_new, p24_new, gate


def make_sub(anchor, p48_new, p24_new, path):
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

    sr48, _ = spearmanr(sub["prob_48h"], anchor["prob_48h"])
    sr24, _ = spearmanr(sub["prob_24h"], anchor["prob_24h"])
    mad48 = np.mean(np.abs(sub["prob_48h"] - anchor["prob_48h"]))
    mad24 = np.mean(np.abs(sub["prob_24h"] - anchor["prob_24h"]))
    sub.to_csv(path, index=False)
    print(f"  {path}")
    print(f"    viol={viol}  rho48={sr48:.6f}  rho24={sr24:.6f}  "
          f"MAD48={mad48:.4f}  MAD24={mad24:.4f}  "
          f"med48={np.median(sub['prob_48h']):.4f}  "
          f"med24={np.median(sub['prob_24h']):.4f}")
    return sub


def main():
    print("=" * 60)
    print("  Exp22e: Gap-Gated Coupled Push")
    print("=" * 60)

    anchor = pd.read_csv(ANCHOR_PATH)
    p48_a = anchor["prob_48h"].values.copy()
    p24_a = anchor["prob_24h"].values.copy()
    gap = p48_a - p24_a

    print(f"\n  Gap: min={gap.min():.4f}  p10={np.percentile(gap,10):.4f}"
          f"  med={np.median(gap):.4f}  p90={np.percentile(gap,90):.4f}"
          f"  max={gap.max():.4f}")

    # ================================================================
    # Phase 1: Gate window search at lam=5.0, r=0.7
    # (lam=5 r=0.7 had 66 violations uniform; gate should fix this)
    # ================================================================
    print("\n=== Phase 1: Gate window search (lam=5.0, r=0.7) ===")
    for lo, hi in [(0.003, 0.015), (0.005, 0.020), (0.005, 0.030),
                   (0.010, 0.020), (0.010, 0.030)]:
        print(f"\n--- gate=[{lo},{hi}] ---")
        p48_new, p24_new, gate = gap_gated_push(
            p48_a, p24_a, A, B, lam=5.0, r=0.7, gate_lo=lo, gate_hi=hi)
        n_full = (gate > 0.99).sum()
        n_zero = (gate < 0.01).sum()
        n_partial = 95 - n_full - n_zero
        print(f"  gate: {n_zero} off, {n_partial} partial, {n_full} full")
        make_sub(anchor, p48_new, p24_new,
                 f"submission_exp22e_g{int(lo*1000):03d}_{int(hi*1000):03d}_lam500.csv")

    # ================================================================
    # Phase 2: Push harder with gating (lam=6,8,10)
    # ================================================================
    print("\n=== Phase 2: Large lambda with gating ===")
    # Use gate=[0.005, 0.020] as default (covers the 0.013 cluster)
    for lam in [6.0, 8.0, 10.0, 15.0]:
        for r in [0.7, 1.0]:
            print(f"\n--- lam={lam}, r={r}, gate=[0.005,0.020] ---")
            p48_new, p24_new, gate = gap_gated_push(
                p48_a, p24_a, A, B, lam=lam, r=r,
                gate_lo=0.005, gate_hi=0.020)
            tag = f"lam{int(lam*100):04d}_r{int(r*10):02d}"
            make_sub(anchor, p48_new, p24_new,
                     f"submission_exp22e_{tag}.csv")

    print("\n=== Exp22e complete ===")


if __name__ == "__main__":
    main()
