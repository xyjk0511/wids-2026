"""Exp22f: Piecewise-logit p48 + gap-gated p24.

p48: 3-segment monotonic piecewise-logit (continuous, all slopes > 0).
p24: gap-gated coupling — small gap gets higher r (more aggressive yield).

Usage:
    python -m scripts.exp22f_piecewise_gated
"""

import numpy as np
import pandas as pd
from scipy.special import expit, logit as sp_logit
from scipy.stats import spearmanr

from src.config import HORIZONS, PROB_COLS, SAMPLE_SUB_PATH
from src.monotonic import submission_postprocess

ANCHOR_PATH = "submissions/submission_0.96624.csv"
EPS = 1e-7
# Current best linear params (logit-space effective)
A, B = 1.0655, -0.0108
LAM_BASE = 4.0
R_BASE = 0.7


def safe_logit(p):
    return sp_logit(np.clip(np.asarray(p, float), EPS, 1.0 - EPS))


def piecewise_logit_transform(lp, slopes, intercepts, breaks):
    """3-segment piecewise-linear in logit space.

    slopes: [s1, s2, s3]  (all > 0)
    intercepts: [d1]  (d2, d3 derived from continuity)
    breaks: [c1, c2]
    """
    s1, s2, s3 = slopes
    d1 = intercepts[0]
    c1, c2 = breaks
    d2 = d1 + (s1 - s2) * c1
    d3 = d2 + (s2 - s3) * c2

    out = np.empty_like(lp)
    m1 = lp < c1
    m3 = lp >= c2
    m2 = ~m1 & ~m3
    out[m1] = s1 * lp[m1] + d1
    out[m2] = s2 * lp[m2] + d2
    out[m3] = s3 * lp[m3] + d3
    return out


def gap_gated_r(gap, r_hi=0.9, r_lo=0.5, g_lo=0.005, g_hi=0.025):
    """Small gap -> r_hi (aggressive p24 push), large gap -> r_lo."""
    gate = np.clip((gap - g_lo) / (g_hi - g_lo), 0.0, 1.0)
    return r_hi - (r_hi - r_lo) * gate


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
    print("  Exp22f: Piecewise-Logit p48 + Gap-Gated p24")
    print("=" * 60)

    anchor = pd.read_csv(ANCHOR_PATH)
    p48_a = anchor["prob_48h"].values.copy()
    p24_a = anchor["prob_24h"].values.copy()
    lp48 = safe_logit(p48_a)
    gap = p48_a - p24_a

    # Breakpoints at 33rd/67th percentile of logit
    c1 = np.percentile(lp48, 33)
    c2 = np.percentile(lp48, 67)
    print(f"\n  Breakpoints: c1={c1:.3f} (p33)  c2={c2:.3f} (p67)")
    print(f"  Segments: {(lp48 < c1).sum()} | "
          f"{((lp48 >= c1) & (lp48 < c2)).sum()} | {(lp48 >= c2).sum()}")

    # Current best linear: effective slope = 1 + LAM*(A-1), intercept = LAM*B
    s_base = 1.0 + LAM_BASE * (A - 1.0)
    d_base = LAM_BASE * B
    print(f"  Linear baseline: slope={s_base:.4f}  intercept={d_base:.4f}")

    # ================================================================
    # Phase 1: Piecewise-logit p48 (uniform r for p24)
    # Vary outer slopes relative to base, keep middle = base
    # ================================================================
    print("\n=== Phase 1: Piecewise-logit p48 search ===")
    print(f"{'s1':>6} {'s2':>6} {'s3':>6} {'rho48':>10} {'viol':>5} "
          f"{'MAD48':>8} {'med48':>7}")
    print("-" * 55)

    best_rho = 0
    best_cfg = None
    # s2 = base slope; vary s1 (low-p segment) and s3 (high-p segment)
    for s1_mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
        for s3_mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
            s1 = s_base * s1_mult
            s3 = s_base * s3_mult
            lp_new = piecewise_logit_transform(
                lp48, [s1, s_base, s3], [d_base], [c1, c2])
            p48_new = expit(lp_new)

            # Coupled p24 with uniform r
            lp24 = safe_logit(p24_a)
            lp24_new = piecewise_logit_transform(
                lp24, [s1, s_base, s3], [d_base], [c1, c2])
            # Blend: same as coupled but using piecewise
            r = R_BASE
            p24_blend = expit((1 - r) * lp24 + r * lp24_new)
            # Actually for p24 we should use the same coupling logic
            # p24_new = expit((1-lam*r)*lp24 + lam*r*(a*lp24+b))
            # But with piecewise, the "calibrated" version IS lp_new
            # So: p24_coupled = expit((1-r)*lp24 + r*lp24_new)
            # Wait, that's not right either. Let me think...
            # The piecewise transform IS the full transform (not just the cal part)
            # For p48: lp48_out = piecewise(lp48) directly
            # For p24: lp24_out = (1-r)*lp24 + r*piecewise(lp24)
            p24_new = expit((1.0 - r) * lp24 + r * lp24_new)

            sr48, _ = spearmanr(p48_new, p48_a)
            viol = (p48_new < p24_new + 1e-7).sum()
            mad48 = np.mean(np.abs(p48_new - p48_a))
            med48 = np.median(p48_new)

            tag = f"{s1:.3f} {s_base:.3f} {s3:.3f}"
            print(f"{s1:6.3f} {s_base:6.3f} {s3:6.3f} {sr48:10.6f} "
                  f"{viol:5d} {mad48:8.4f} {med48:7.4f}")

            if sr48 > best_rho:
                best_rho = sr48
                best_cfg = (s1, s_base, s3)

    print(f"\n  Best piecewise: s=({best_cfg[0]:.3f}, {best_cfg[1]:.3f}, "
          f"{best_cfg[2]:.3f})  rho48={best_rho:.6f}")

    # ================================================================
    # Phase 2: Gap-gated p24 (with linear p48 baseline)
    # p48 uses current best linear; p24 r varies by gap
    # ================================================================
    print("\n=== Phase 2: Gap-gated p24 coupling ===")

    lp48_cal = (1.0 + LAM_BASE * (A - 1.0)) * lp48 + LAM_BASE * B
    p48_lin = expit(lp48_cal)
    lp24 = safe_logit(p24_a)
    lp24_cal = (1.0 + LAM_BASE * (A - 1.0)) * lp24 + LAM_BASE * B

    for r_hi, r_lo in [(0.9, 0.5), (0.9, 0.7), (0.8, 0.5),
                        (0.8, 0.6), (1.0, 0.5), (1.0, 0.7)]:
        r_vec = gap_gated_r(gap, r_hi=r_hi, r_lo=r_lo)
        lp24_new = (1.0 - r_vec) * lp24 + r_vec * lp24_cal
        p24_new = expit(lp24_new)

        sr24, _ = spearmanr(p24_new, p24_a)
        viol = (p48_lin < p24_new + 1e-7).sum()
        mad24 = np.mean(np.abs(p24_new - p24_a))
        print(f"  r_hi={r_hi:.1f} r_lo={r_lo:.1f}  "
              f"rho24={sr24:.6f}  viol={viol}  MAD24={mad24:.4f}  "
              f"n_full={int((r_vec > r_hi - 0.01).sum())}  "
              f"n_base={int((r_vec < r_lo + 0.01).sum())}")

    # ================================================================
    # Phase 3: Generate submissions — best piecewise + gated combos
    # ================================================================
    print("\n=== Phase 3: Submissions ===")

    configs = [
        # (label, s1_mult, s3_mult, r_hi, r_lo, use_gated_r)
        ("pw_s09_s11_r07",  0.9, 1.1, None, None, False),
        ("pw_s11_s09_r07",  1.1, 0.9, None, None, False),
        ("pw_s08_s12_r07",  0.8, 1.2, None, None, False),
        ("pw_s12_s08_r07",  1.2, 0.8, None, None, False),
        ("lin_gr_90_50",    1.0, 1.0, 0.9, 0.5, True),
        ("lin_gr_90_70",    1.0, 1.0, 0.9, 0.7, True),
        ("pw_s09_s11_gr90_50", 0.9, 1.1, 0.9, 0.5, True),
        ("pw_s11_s09_gr90_70", 1.1, 0.9, 0.9, 0.7, True),
    ]

    for label, s1m, s3m, r_hi, r_lo, use_gated in configs:
        s1, s3 = s_base * s1m, s_base * s3m
        lp48_new = piecewise_logit_transform(
            lp48, [s1, s_base, s3], [d_base], [c1, c2])
        p48_new = expit(lp48_new)

        lp24_cal_pw = piecewise_logit_transform(
            lp24, [s1, s_base, s3], [d_base], [c1, c2])

        if use_gated:
            r_vec = gap_gated_r(gap, r_hi=r_hi, r_lo=r_lo)
            p24_new = expit((1.0 - r_vec) * lp24 + r_vec * lp24_cal_pw)
        else:
            p24_new = expit((1.0 - R_BASE) * lp24 + R_BASE * lp24_cal_pw)

        print(f"\n--- {label} ---")
        make_sub(anchor, p48_new, p24_new,
                 f"submission_exp22f_{label}.csv")

    print("\n=== Exp22f complete ===")


if __name__ == "__main__":
    main()
