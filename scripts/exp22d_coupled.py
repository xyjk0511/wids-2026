"""Exp22d round2: 24+48 coupled down-push.

Instead of passively yielding p24 when violations occur,
proactively push p24 down proportionally: push24 = r * push48.
This prevents violations from happening, allowing smoother p48 push.

LB evidence so far:
  lam=1.05 old: 0.96670
  lam=1.05 new(p24-yields): 0.96671
  lam=1.10 new: 0.96674
  lam=1.20 new: 0.96679  <-- current PB

Usage:
    python -m scripts.exp22d_coupled
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


def coupled_push(p48_anchor, p24_anchor, a, b, lam, r):
    """Push p48 and p24 in coupled fashion.

    p48: logit(p48_new) = (1-lam)*logit(p48) + lam*(a*logit(p48)+b)
    p24: logit(p24_new) = (1-lam*r)*logit(p24) + lam*r*(a*logit(p24)+b)

    r=0: only push p48 (old behavior)
    r=0.3: p24 gets 30% of p48's push
    r=0.5: p24 gets 50% of p48's push
    """
    lp48 = safe_logit(p48_anchor)
    lp48_new = (1.0 - lam) * lp48 + lam * (a * lp48 + b)
    p48_new = expit(lp48_new)

    lp24 = safe_logit(p24_anchor)
    lam24 = lam * r
    lp24_new = (1.0 - lam24) * lp24 + lam24 * (a * lp24 + b)
    p24_new = expit(lp24_new)

    return p48_new, p24_new


def make_submission(anchor, p48_new, p24_new, path):
    """Build submission with coupled p24+p48, hard monotonicity."""
    p12 = anchor["prob_12h"].values.copy()
    p48 = np.clip(p48_new.copy(), 1e-6, 1.0)
    p24 = np.clip(p24_new.copy(), 1e-6, 1.0)

    # Diagnostics before any fix
    viol_48_24 = (p48 < p24 + 1e-7).sum()
    gap_min = np.min(p48 - p24)

    # Hard monotonicity: if still any violations, p24 yields
    mask = p48 < p24 + 1e-7
    p24[mask] = p48[mask] - 1e-7

    # Ensure p12 <= p24
    p12 = np.minimum(p12, p24 - 1e-7)

    prob_dict = {12: p12, 24: p24, 48: p48, 72: np.ones(len(anchor))}
    pp = submission_postprocess(prob_dict)

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = pp[h]

    sr48, _ = spearmanr(sub["prob_48h"], anchor["prob_48h"])
    sr24, _ = spearmanr(sub["prob_24h"], anchor["prob_24h"])
    sr12, _ = spearmanr(sub["prob_12h"], anchor["prob_12h"])
    mad48 = np.mean(np.abs(sub["prob_48h"] - anchor["prob_48h"]))
    mad24 = np.mean(np.abs(sub["prob_24h"] - anchor["prob_24h"]))

    sub.to_csv(path, index=False)
    print(f"  {path}")
    print(f"    viol={viol_48_24}  gap_min={gap_min:.6f}")
    print(f"    rho12={sr12:.6f}  rho24={sr24:.6f}  rho48={sr48:.6f}")
    print(f"    MAD24={mad24:.6f}  MAD48={mad48:.6f}")
    print(f"    med24: {np.median(anchor['prob_24h']):.4f}"
          f"->{np.median(sub['prob_24h']):.4f}  "
          f"med48: {np.median(anchor['prob_48h']):.4f}"
          f"->{np.median(sub['prob_48h']):.4f}")
    return sub


def main():
    print("=" * 60)
    print("  Exp22d Round 2: Coupled 24+48 Down-Push")
    print("=" * 60)

    anchor = pd.read_csv(ANCHOR_PATH)
    p48_a = anchor["prob_48h"].values.copy()
    p24_a = anchor["prob_24h"].values.copy()

    print(f"\n  Anchor: med24={np.median(p24_a):.4f}  med48={np.median(p48_a):.4f}")
    print(f"  Gap: min={np.min(p48_a-p24_a):.4f}  med={np.median(p48_a-p24_a):.4f}")

    # ================================================================
    # Main grid: r x lam
    # ================================================================
    ratios = [0.0, 0.3, 0.5, 0.7, 1.0]
    lambdas = [1.05, 1.20, 1.50, 2.00]

    print(f"\n{'r':>4} {'lam':>5} {'viol':>5} {'rho48':>10} {'rho24':>10} "
          f"{'MAD48':>8} {'MAD24':>8} {'med48':>7} {'med24':>7}")
    print("-" * 75)

    for r in ratios:
        for lam in lambdas:
            p48_new, p24_new = coupled_push(p48_a, p24_a, A_FIXED, B_FIXED, lam, r)

            viol = (p48_new < p24_new + 1e-7).sum()
            gap = np.min(p48_new - p24_new)

            # Quick safety check without full submission
            sr48, _ = spearmanr(p48_new, p48_a)
            sr24, _ = spearmanr(p24_new, p24_a)
            mad48 = np.mean(np.abs(p48_new - p48_a))
            mad24 = np.mean(np.abs(p24_new - p24_a))

            print(f"{r:4.1f} {lam:5.2f} {viol:5d} {sr48:10.6f} {sr24:10.6f} "
                  f"{mad48:8.4f} {mad24:8.4f} {np.median(p48_new):7.4f} "
                  f"{np.median(p24_new):7.4f}")

    # ================================================================
    # Generate submissions for best candidates
    # ================================================================
    print("\n=== Generating Submissions ===")

    configs = [
        # (r, lam, label)
        (0.3, 1.05, "r03_lam105"),
        (0.3, 1.20, "r03_lam120"),
        (0.3, 1.50, "r03_lam150"),
        (0.3, 2.00, "r03_lam200"),
        (0.5, 1.05, "r05_lam105"),
        (0.5, 1.20, "r05_lam120"),
        (0.5, 1.50, "r05_lam150"),
        (0.5, 2.00, "r05_lam200"),
        (0.7, 1.20, "r07_lam120"),
        (0.7, 1.50, "r07_lam150"),
        (1.0, 1.20, "r10_lam120"),
        (1.0, 1.50, "r10_lam150"),
    ]

    for r, lam, label in configs:
        print(f"\n--- r={r}, lam={lam} ---")
        p48_new, p24_new = coupled_push(p48_a, p24_a, A_FIXED, B_FIXED, lam, r)
        make_submission(anchor, p48_new, p24_new,
                        f"submission_exp22d_{label}.csv")

    print("\n=== Exp22d Round 2 complete ===")


if __name__ == "__main__":
    main()
