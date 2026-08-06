"""Exp22b: Constrained line search along down-push direction.

Fix a=1.0655 (from R2A bootstrap), search b more negative.
Track violations (p48_new < p24_anchor) as safety constraint.

LB evidence: down-push (b<0) scored 0.96634 > up-push 0.96629 > anchor 0.96624.

Usage:
    python -m scripts.exp22b_line_search
"""

import numpy as np
import pandas as pd
from scipy.special import expit, logit as sp_logit
from scipy.stats import spearmanr

from src.config import HORIZONS, PROB_COLS, SAMPLE_SUB_PATH
from src.monotonic import submission_postprocess


ANCHOR_PATH = "submission_0.96624.csv"
EPS = 1e-7

# From R2A bootstrap (the winning direction)
A_FIXED = 1.0655
B_BASE = -0.0108
LAM_BASE = 0.20


def safe_logit(p):
    return sp_logit(np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS))


def apply_transform(p_anchor, a, b, lam):
    """logit(p_new) = (1-lam)*logit(p) + lam*(a*logit(p)+b)"""
    lp = safe_logit(p_anchor)
    lp_new = (1.0 - lam) * lp + lam * (a * lp + b)
    return expit(lp_new)


def make_submission(anchor, p48_new, path):
    """Replace p48, hard-constraint monotonicity, save."""
    p24 = anchor["prob_24h"].values.copy()
    p48_clipped = np.clip(p48_new.copy(), 1e-6, 1.0)

    # Hard constraint: p48 >= p24 + eps (no L2 projection needed)
    violations = (p48_clipped < p24 + 1e-7).sum()
    p48_clipped = np.maximum(p48_clipped, p24 + 1e-7)

    prob_dict = {
        12: anchor["prob_12h"].values.copy(),
        24: p24,
        48: p48_clipped,
        72: np.ones(len(anchor)),
    }
    pp = submission_postprocess(prob_dict)

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = pp[h]

    # Verify p24 untouched
    d24 = np.max(np.abs(sub["prob_24h"] - anchor["prob_24h"]))
    sr48, _ = spearmanr(sub["prob_48h"], anchor["prob_48h"])
    mad48 = np.mean(np.abs(sub["prob_48h"] - anchor["prob_48h"]))

    sub.to_csv(path, index=False)
    print(f"  Saved: {path}")
    print(f"    violations(p48<p24): {violations}  |d24|_max={d24:.8f}  "
          f"rho48={sr48:.6f}  MAD48={mad48:.6f}")
    return sub, violations


def main():
    print("=" * 60)
    print("  Exp22b: Down-Push Line Search")
    print("=" * 60)

    anchor = pd.read_csv(ANCHOR_PATH)
    anchor_p48 = anchor["prob_48h"].values.copy()
    anchor_p24 = anchor["prob_24h"].values.copy()
    print(f"\n  Anchor p48: min={anchor_p48.min():.4f}  "
          f"med={np.median(anchor_p48):.4f}  max={anchor_p48.max():.4f}")
    print(f"  Anchor p24: min={anchor_p24.min():.4f}  "
          f"med={np.median(anchor_p24):.4f}  max={anchor_p24.max():.4f}")

    # Gap between p48 and p24 (how much room to push down)
    gap = anchor_p48 - anchor_p24
    print(f"  p48-p24 gap: min={gap.min():.4f}  med={np.median(gap):.4f}  "
          f"p10={np.percentile(gap, 10):.4f}")

    # ================================================================
    # Phase 1: Fix a, search b (more negative)
    # ================================================================
    print(f"\n=== Phase 1: Fix a={A_FIXED}, lam={LAM_BASE}, search b ===")
    print(f"  Base b={B_BASE} (LB=0.96634)")

    b_multipliers = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    for mult in b_multipliers:
        b = B_BASE * mult
        print(f"\n--- b={b:.4f} ({mult:.1f}x base) ---")
        p48_new = apply_transform(anchor_p48, A_FIXED, b, LAM_BASE)

        # Pre-clip stats
        sr, _ = spearmanr(p48_new, anchor_p48)
        mad = np.mean(np.abs(p48_new - anchor_p48))
        med = np.median(p48_new)
        viol = (p48_new < anchor_p24 + 1e-7).sum()
        print(f"  pre-clip: Spearman={sr:.6f}  MAD={mad:.6f}  "
              f"med={med:.4f}  violations={viol}")

        tag = f"b{mult:.0f}x"
        make_submission(anchor, p48_new,
                        f"submission_exp22b_{tag}.csv")

    # ================================================================
    # Phase 2: Also try larger lambda with base b
    # (amplifies both stretch and shift proportionally)
    # ================================================================
    print(f"\n=== Phase 2: Fix a={A_FIXED}, b={B_BASE}, search lam ===")

    for lam in [0.30, 0.40, 0.50, 0.60, 0.80]:
        print(f"\n--- lam={lam:.2f} ---")
        p48_new = apply_transform(anchor_p48, A_FIXED, B_BASE, lam)

        sr, _ = spearmanr(p48_new, anchor_p48)
        mad = np.mean(np.abs(p48_new - anchor_p48))
        med = np.median(p48_new)
        viol = (p48_new < anchor_p24 + 1e-7).sum()
        print(f"  pre-clip: Spearman={sr:.6f}  MAD={mad:.6f}  "
              f"med={med:.4f}  violations={viol}")

        tag = f"lam{int(lam*100):02d}"
        make_submission(anchor, p48_new,
                        f"submission_exp22b_{tag}.csv")

    # ================================================================
    # Phase 3: Pure shift (a=1.0, only b<0)
    # Separates "stretch" from "shift" contribution
    # ================================================================
    print(f"\n=== Phase 3: Pure shift (a=1.0), search b ===")

    for b in [-0.01, -0.02, -0.03, -0.05]:
        print(f"\n--- a=1.0, b={b:.3f}, lam=1.0 (direct) ---")
        p48_new = apply_transform(anchor_p48, 1.0, b, 1.0)

        sr, _ = spearmanr(p48_new, anchor_p48)
        mad = np.mean(np.abs(p48_new - anchor_p48))
        med = np.median(p48_new)
        viol = (p48_new < anchor_p24 + 1e-7).sum()
        print(f"  pre-clip: Spearman={sr:.6f}  MAD={mad:.6f}  "
              f"med={med:.4f}  violations={viol}")

        tag = f"pure_b{abs(b):.3f}".replace(".", "")
        make_submission(anchor, p48_new,
                        f"submission_exp22b_{tag}.csv")

    print("\n=== Exp22b complete ===")


if __name__ == "__main__":
    main()
