"""Exp22d: Break the cliff - p24 yields to p48.

Instead of clipping p48 UP to p24 (which eats our gains),
clip p24 DOWN to p48 on violation samples (lower weight: 0.21 vs 0.28).

This should let us push past lam=1.05 without the cliff.

Usage:
    python -m scripts.exp22d_p24_yields
"""

import numpy as np
import pandas as pd
from scipy.special import expit, logit as sp_logit
from scipy.stats import spearmanr

from src.config import HORIZONS, PROB_COLS, SAMPLE_SUB_PATH
from src.monotonic import submission_postprocess


ANCHOR_PATH = "submission_0.96624.csv"
EPS = 1e-7
A_FIXED = 1.0655
B_FIXED = -0.0108


def safe_logit(p):
    return sp_logit(np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS))


def apply_transform(p, a, b, lam):
    lp = safe_logit(p)
    lp_new = (1.0 - lam) * lp + lam * (a * lp + b)
    return expit(lp_new)


def make_sub_p24_yields(anchor, p48_new, path):
    """Monotonicity fix: p24 yields to p48 (not the other way around)."""
    p12 = anchor["prob_12h"].values.copy()
    p24 = anchor["prob_24h"].values.copy()
    p48 = np.clip(p48_new.copy(), 1e-6, 1.0)

    # Where p48 < p24: push p24 DOWN instead of pushing p48 UP
    viol_mask = p48 < p24 + 1e-7
    n_viol = viol_mask.sum()
    p24_fixed = p24.copy()
    p24_fixed[viol_mask] = p48[viol_mask] - 1e-7

    # Also ensure p12 <= p24_fixed
    p12_fixed = np.minimum(p12, p24_fixed - 1e-7)

    prob_dict = {
        12: p12_fixed,
        24: p24_fixed,
        48: p48,
        72: np.ones(len(anchor)),
    }
    pp = submission_postprocess(prob_dict)

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = pp[h]

    # Diagnostics
    d12 = np.max(np.abs(sub["prob_12h"] - anchor["prob_12h"]))
    d24 = np.max(np.abs(sub["prob_24h"] - anchor["prob_24h"]))
    sr48, _ = spearmanr(sub["prob_48h"], anchor["prob_48h"])
    sr24, _ = spearmanr(sub["prob_24h"], anchor["prob_24h"])
    sr12, _ = spearmanr(sub["prob_12h"], anchor["prob_12h"])
    mad48 = np.mean(np.abs(sub["prob_48h"] - anchor["prob_48h"]))
    mad24 = np.mean(np.abs(sub["prob_24h"] - anchor["prob_24h"]))

    sub.to_csv(path, index=False)
    print(f"  Saved: {path}")
    print(f"    p48_viol={n_viol}  rho12={sr12:.6f}  rho24={sr24:.6f}  "
          f"rho48={sr48:.6f}")
    print(f"    MAD24={mad24:.6f}  MAD48={mad48:.6f}  "
          f"|d12|={d12:.8f}  |d24|={d24:.6f}")
    print(f"    med24: {np.median(anchor['prob_24h']):.4f}"
          f"->{np.median(sub['prob_24h']):.4f}  "
          f"med48: {np.median(anchor['prob_48h']):.4f}"
          f"->{np.median(sub['prob_48h']):.4f}")
    return sub


def make_sub_old_way(anchor, p48_new, path):
    """Old way: p48 clipped UP to p24 (for comparison)."""
    p24 = anchor["prob_24h"].values.copy()
    p48 = np.clip(p48_new.copy(), 1e-6, 1.0)
    viol = (p48 < p24 + 1e-7).sum()
    p48 = np.maximum(p48, p24 + 1e-7)

    prob_dict = {
        12: anchor["prob_12h"].values.copy(),
        24: p24,
        48: p48,
        72: np.ones(len(anchor)),
    }
    pp = submission_postprocess(prob_dict)

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = pp[h]

    sr48, _ = spearmanr(sub["prob_48h"], anchor["prob_48h"])
    mad48 = np.mean(np.abs(sub["prob_48h"] - anchor["prob_48h"]))
    sub.to_csv(path, index=False)
    print(f"  [old] {path}  viol={viol}  rho48={sr48:.6f}  MAD48={mad48:.6f}")
    return sub


def main():
    print("=" * 60)
    print("  Exp22d: Break the Cliff (p24 yields to p48)")
    print("=" * 60)

    anchor = pd.read_csv(ANCHOR_PATH)
    p48_anchor = anchor["prob_48h"].values.copy()
    p24_anchor = anchor["prob_24h"].values.copy()

    gap = p48_anchor - p24_anchor
    print(f"\n  p48-p24 gap: min={gap.min():.4f}  p5={np.percentile(gap,5):.4f}"
          f"  p10={np.percentile(gap,10):.4f}  med={np.median(gap):.4f}")

    # ================================================================
    # Compare old vs new at the cliff point (lam=1.05)
    # ================================================================
    print("\n=== Comparison at lam=1.05 (current best, LB=0.96670) ===")
    p48_new = apply_transform(p48_anchor, A_FIXED, B_FIXED, 1.05)
    print("\n  Old way (p48 clips up):")
    make_sub_old_way(anchor, p48_new, "submission_exp22d_lam105_old.csv")
    print("\n  New way (p24 yields):")
    make_sub_p24_yields(anchor, p48_new, "submission_exp22d_lam105_new.csv")

    # ================================================================
    # Push past the cliff with new strategy
    # ================================================================
    print("\n=== Push past cliff with p24-yields strategy ===")

    for lam in [1.10, 1.15, 1.20, 1.30, 1.50, 2.00]:
        print(f"\n--- lam={lam:.2f} ---")
        p48_new = apply_transform(p48_anchor, A_FIXED, B_FIXED, lam)

        # How many violations?
        viol = (p48_new < p24_anchor + 1e-7).sum()
        med = np.median(p48_new)
        print(f"  raw: viol={viol}  med48={med:.4f}")

        make_sub_p24_yields(anchor, p48_new,
                            f"submission_exp22d_lam{int(lam*100):03d}.csv")

    # ================================================================
    # Also: search a at lam=1.20 (was cliff before, now accessible)
    # ================================================================
    print("\n=== a-search at lam=1.20 (previously cliff) ===")
    for a in [1.04, 1.05, 1.06, 1.0655, 1.07, 1.08]:
        p48_new = apply_transform(p48_anchor, a, B_FIXED, 1.20)
        viol = (p48_new < p24_anchor + 1e-7).sum()
        print(f"\n  a={a:.4f}, lam=1.20, viol={viol}")
        make_sub_p24_yields(anchor, p48_new,
                            f"submission_exp22d_a{int(a*1000):04d}_lam120.csv")

    print("\n=== Exp22d complete ===")


if __name__ == "__main__":
    main()
