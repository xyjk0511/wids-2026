"""Exp22c: Continue line search - larger lambda.

LB evidence: lam=0.20→0.96634, 0.40→0.96643, 0.60→0.96652 (monotonic, no peak yet).
Pure shift useless (0.96628). Stretch (a>1) is the main driver.

Usage:
    python -m scripts.exp22c_push_further
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


def make_submission(anchor, p48_new, path):
    p24 = anchor["prob_24h"].values.copy()
    p48_c = np.clip(p48_new.copy(), 1e-6, 1.0)
    viol = (p48_c < p24 + 1e-7).sum()
    p48_c = np.maximum(p48_c, p24 + 1e-7)

    prob_dict = {
        12: anchor["prob_12h"].values.copy(),
        24: p24,
        48: p48_c,
        72: np.ones(len(anchor)),
    }
    pp = submission_postprocess(prob_dict)

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = pp[h]

    d24 = np.max(np.abs(sub["prob_24h"] - anchor["prob_24h"]))
    sr48, _ = spearmanr(sub["prob_48h"], anchor["prob_48h"])
    mad48 = np.mean(np.abs(sub["prob_48h"] - anchor["prob_48h"]))
    med48 = np.median(sub["prob_48h"])

    sub.to_csv(path, index=False)
    print(f"  Saved: {path}")
    print(f"    viol={viol}  |d24|={d24:.8f}  rho48={sr48:.6f}  "
          f"MAD48={mad48:.6f}  med48={med48:.4f}")
    return sub


def main():
    print("=" * 60)
    print("  Exp22c: Push Further (lam > 0.60)")
    print("=" * 60)

    anchor = pd.read_csv(ANCHOR_PATH)
    anchor_p48 = anchor["prob_48h"].values.copy()
    anchor_p24 = anchor["prob_24h"].values.copy()

    print(f"\n  Known LB trend: lam 0.20->0.96634  0.40->0.96643  0.60->0.96652")
    print(f"  Anchor p48 med={np.median(anchor_p48):.4f}  "
          f"p48-p24 gap min={np.min(anchor_p48 - anchor_p24):.4f}")

    lambdas = [0.80, 1.00, 1.20, 1.50, 2.00]

    for lam in lambdas:
        print(f"\n--- lam={lam:.2f} (a={A_FIXED}, b={B_FIXED}) ---")
        p48_new = apply_transform(anchor_p48, A_FIXED, B_FIXED, lam)

        sr, _ = spearmanr(p48_new, anchor_p48)
        mad = np.mean(np.abs(p48_new - anchor_p48))
        med = np.median(p48_new)
        viol = (p48_new < anchor_p24 + 1e-7).sum()
        print(f"  pre-clip: rho={sr:.6f}  MAD={mad:.6f}  "
              f"med={med:.4f}  viol={viol}")

        tag = f"lam{int(lam*100):03d}"
        make_submission(anchor, p48_new,
                        f"submission_exp22c_{tag}.csv")

    # Also: try a slightly larger 'a' with moderate lambda
    print(f"\n=== Bonus: search a with lam=0.60 (current best) ===")
    for a in [1.04, 1.08, 1.10, 1.12]:
        print(f"\n--- a={a}, b={B_FIXED}, lam=0.60 ---")
        p48_new = apply_transform(anchor_p48, a, B_FIXED, 0.60)

        sr, _ = spearmanr(p48_new, anchor_p48)
        mad = np.mean(np.abs(p48_new - anchor_p48))
        med = np.median(p48_new)
        print(f"  rho={sr:.6f}  MAD={mad:.6f}  med={med:.4f}")

        tag = f"a{int(a*100):03d}_lam060"
        make_submission(anchor, p48_new,
                        f"submission_exp22c_{tag}.csv")

    print("\n=== Exp22c complete ===")


if __name__ == "__main__":
    main()
