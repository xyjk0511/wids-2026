"""Exp28: Blend exp27 model predictions into 0.96624 anchor, then apply PB calibration."""
import sys; sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from scipy.special import expit, logit as sp_logit
from scipy.stats import spearmanr

from src.config import HORIZONS, PROB_COLS, SAMPLE_SUB_PATH
from src.monotonic import submission_postprocess

ANCHOR_PATH = "submissions/submission_0.96624.csv"
BLEND_PATH = "submissions/submission_exp27_best_blend.csv"
EPS = 1e-7
A, B = 1.0655, -0.0108


def safe_logit(p):
    return sp_logit(np.clip(np.asarray(p, float), EPS, 1.0 - EPS))


def gap_gated_push(p48, p24, a, b, lam, r_hi=1.0, r_lo=0.7, g_lo=0.005, g_hi=0.025):
    """PB calibration: gap-gated logit push on p48+p24."""
    gap = p48 - p24
    gate = np.clip((gap - g_lo) / (g_hi - g_lo), 0.0, 1.0)
    lp48, lp24 = safe_logit(p48), safe_logit(p24)
    cal48, cal24 = a * lp48 + b, a * lp24 + b

    lam_eff = lam * gate
    p48_new = expit((1.0 - lam_eff) * lp48 + lam_eff * cal48)

    r = r_hi - (r_hi - r_lo) * gate
    lam24 = lam * r * gate
    p24_new = expit((1.0 - lam24) * lp24 + lam24 * cal24)
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
    mad48 = np.mean(np.abs(sub["prob_48h"] - anchor["prob_48h"]))
    sub.to_csv(path, index=False)
    print(f"  {path}")
    print(f"    viol={viol}  rho48={sr48:.6f}  MAD48={mad48:.4f}  "
          f"med48={np.median(sub['prob_48h']):.4f}")
    return sub


def main():
    print("=== Exp28: Anchor + Model Blend + PB Calibration ===")
    anchor = pd.read_csv(ANCHOR_PATH)
    blend = pd.read_csv(BLEND_PATH)

    # Align by event_id
    blend = blend.set_index("event_id").loc[anchor["event_id"]].reset_index()

    print(f"\n--- Baseline: PB calibration on pure anchor (lam=6.0) ---")
    p48_cal, p24_cal = gap_gated_push(
        anchor["prob_48h"].values, anchor["prob_24h"].values,
        A, B, lam=6.0)
    make_sub(anchor, p48_cal, p24_cal,
             "submissions/submission_exp28_baseline.csv")

    # Blend anchor + exp27 at various alpha, then calibrate
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        print(f"\n--- alpha={alpha:.2f} (anchor={1-alpha:.2f}, exp27={alpha:.2f}) ---")
        # Blend in probability space
        p48_mix = (1 - alpha) * anchor["prob_48h"].values + alpha * blend["prob_48h"].values
        p24_mix = (1 - alpha) * anchor["prob_24h"].values + alpha * blend["prob_24h"].values

        # Apply PB calibration
        p48_cal, p24_cal = gap_gated_push(p48_mix, p24_mix, A, B, lam=6.0)
        tag = f"a{int(alpha*100):02d}"
        make_sub(anchor, p48_cal, p24_cal,
                 f"submissions/submission_exp28_{tag}.csv")

    # Also try logit-space blending
    print("\n--- Logit-space blending ---")
    for alpha in [0.05, 0.10]:
        lp48_a = safe_logit(anchor["prob_48h"].values)
        lp48_b = safe_logit(blend["prob_48h"].values)
        lp24_a = safe_logit(anchor["prob_24h"].values)
        lp24_b = safe_logit(blend["prob_24h"].values)

        p48_mix = expit((1 - alpha) * lp48_a + alpha * lp48_b)
        p24_mix = expit((1 - alpha) * lp24_a + alpha * lp24_b)

        p48_cal, p24_cal = gap_gated_push(p48_mix, p24_mix, A, B, lam=6.0)
        tag = f"logit_a{int(alpha*100):02d}"
        make_sub(anchor, p48_cal, p24_cal,
                 f"submissions/submission_exp28_{tag}.csv")

    print("\n=== Exp28 complete ===")


if __name__ == "__main__":
    main()
