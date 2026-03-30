"""Exp19c: Rank-binned monotone calibration for p48.

Core idea: PAVA in p-space degrades to step function (bimodal OOF, empty
middle). Fix: bin by rank percentile (every bin has samples), learn monotone
hit-rate mapping g(u), shrink toward identity.

p48_new = (1-lam)*p48_anchor + lam*g(u_test)
where u_test = rank(p48_anchor)/(n+1)

Usage:
    python -m scripts.exp19c_rank_cal
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import (
    HORIZONS, PROB_COLS, TIME_COL, EVENT_COL, SAMPLE_SUB_PATH,
)
from src.labels import build_horizon_labels
from src.evaluation import hybrid_score
from src.monotonic import submission_postprocess
from src.train import load_data, run_cv, _print_score


ANCHOR_PATH = "submission_0.96624.csv"


# ---------------------------------------------------------------------------
# Rank-binned monotone calibration
# ---------------------------------------------------------------------------
def rank_percentile(p):
    """Convert probabilities to rank percentiles in (0, 1)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    # Average rank for ties, then scale to (0,1)
    from scipy.stats import rankdata
    ranks = rankdata(p, method="average")
    return ranks / (n + 1)


def pava_increasing(values):
    """Pool Adjacent Violators for non-decreasing sequence."""
    v = list(np.asarray(values, dtype=float))
    w = [1] * len(v)
    i = 0
    while i < len(v) - 1:
        if v[i] > v[i + 1]:
            wt = w[i] + w[i + 1]
            v[i] = (w[i] * v[i] + w[i + 1] * v[i + 1]) / wt
            w[i] = wt
            v.pop(i + 1)
            w.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1
    return v, w


def fit_rank_bins(p_elig, y_elig, K=6):
    """Fit K equal-frequency rank bins on eligible samples.

    Returns (bin_centers_u, cal_values) for piecewise linear interpolation.
    bin_centers_u are in rank-percentile space [0, 1].
    """
    u = rank_percentile(p_elig)
    n = len(u)

    # Equal-frequency bin edges in u-space
    edges = np.linspace(0, 1, K + 1)
    bin_idx = np.clip(np.digitize(u, edges[1:-1]), 0, K - 1)

    centers_u = []
    centers_p = []
    hit_rates = []
    counts = []

    for b in range(K):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        centers_u.append(float(u[mask].mean()))
        centers_p.append(float(p_elig[mask].mean()))
        hit_rates.append(float(y_elig[mask].mean()))
        counts.append(int(mask.sum()))

    # PAVA on hit rates to enforce monotonicity
    pava_vals, _ = pava_increasing(hit_rates)

    # Expand PAVA output back to bin count
    cal_vals = []
    j = 0
    for i in range(len(centers_u)):
        if j < len(pava_vals):
            cal_vals.append(pava_vals[j])
        if i < len(centers_u) - 1 and j < len(pava_vals) - 1:
            j += 1
    while len(cal_vals) < len(centers_u):
        cal_vals.append(cal_vals[-1])

    return (np.array(centers_u), np.array(centers_p),
            np.array(cal_vals), np.array(counts))


def apply_rank_cal(p_target, centers_u, cal_vals):
    """Apply rank-binned calibration: map rank percentile -> calibrated prob."""
    u = rank_percentile(p_target)
    return np.interp(u, centers_u, cal_vals,
                     left=cal_vals[0], right=cal_vals[-1])


def bootstrap_rank_cal(p_elig, y_elig, p_target, K=6, n_boot=1000, seed=42):
    """Bootstrap rank-binned calibration, return bagged g(u) on target."""
    rng = np.random.default_rng(seed)
    n = len(p_elig)
    accum = np.zeros(len(p_target), dtype=float)

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        centers_u, _, cal_vals, _ = fit_rank_bins(
            p_elig[idx], y_elig[idx], K=K,
        )
        accum += apply_rank_cal(p_target, centers_u, cal_vals)

    return accum / n_boot


# ---------------------------------------------------------------------------
# Safety & submission helpers
# ---------------------------------------------------------------------------
def safety_check(p_final, p_anchor, label):
    sr, _ = spearmanr(p_final, p_anchor)
    mad = np.mean(np.abs(p_final - p_anchor))
    print(f"  [{label}] Spearman={sr:.6f}  MAD={mad:.6f}  "
          f"med: {np.median(p_anchor):.4f}->{np.median(p_final):.4f}")
    if sr < 0.995:
        print(f"    WARNING: Spearman < 0.995")
    return sr


def make_submission(anchor, p48_new, path):
    prob_dict = {
        12: anchor["prob_12h"].values.copy(),
        24: anchor["prob_24h"].values.copy(),
        48: np.clip(p48_new.copy(), 1e-6, 1.0),
        72: np.ones(len(anchor)),
    }
    # Hard constraint: p48 >= p24 (avoid projection pulling 24h)
    p24 = prob_dict[24]
    prob_dict[48] = np.maximum(prob_dict[48], p24 + 1e-7)

    pp = submission_postprocess(prob_dict)
    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = pp[h]

    mono_viol = (sub["prob_24h"] > sub["prob_48h"] + 1e-9).sum()
    print(f"  72h={'PASS' if (sub['prob_72h']==1.0).all() else 'FAIL'}  "
          f"mono_viol={mono_viol}  Shape: {sub.shape}")
    sub.to_csv(path, index=False)
    print(f"  Saved: {path}")

    print(f"  Spearman vs anchor:")
    for col in PROB_COLS[:-1]:
        sr, _ = spearmanr(sub[col], anchor[col])
        print(f"    {col}: rho={sr:.6f}")
    return sub


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== Exp19c: Rank-Binned Monotone Calibration ===\n")

    anchor = pd.read_csv(ANCHOR_PATH)
    anchor_p48 = anchor["prob_48h"].values.copy()
    print(f"  Anchor p48: min={anchor_p48.min():.4f} "
          f"med={np.median(anchor_p48):.4f} max={anchor_p48.max():.4f}")

    train, test = load_data(feature_level="medium")
    from src.features import get_feature_set
    feature_cols = get_feature_set(train, level="medium")
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    print("\n=== Running CV for OOF ===")
    oof = run_cv(train, feature_cols)
    blend = {h: 0.5 * oof["RSF"][h] + 0.5 * oof["EST"][h] for h in HORIZONS}
    pp_base = submission_postprocess(blend)
    _print_score("Baseline (proj)", pp_base, y_time, y_event)

    # 48h eligible
    labels48, elig48 = build_horizon_labels(y_time, y_event, 48)
    p48_elig = blend[48][elig48]
    y48_elig = labels48[elig48]
    n_elig = elig48.sum()
    print(f"\n  48h eligible: n={n_elig}, pos_rate={y48_elig.mean():.3f}")

    # ==================================================================
    # Diagnostic: show rank-bin structure
    # ==================================================================
    print("\n=== Rank-bin diagnostic (K=6) ===")
    centers_u, centers_p, cal_vals, counts = fit_rank_bins(
        p48_elig, y48_elig, K=6,
    )
    print(f"  {'Bin':>4} {'u_center':>10} {'p_center':>10} "
          f"{'hit_rate':>10} {'n':>5}")
    for i in range(len(centers_u)):
        print(f"  {i:4d} {centers_u[i]:10.4f} {centers_p[i]:10.4f} "
              f"{cal_vals[i]:10.4f} {counts[i]:5d}")

    # OOF evaluation with single fit
    g_oof = apply_rank_cal(blend[48], centers_u, cal_vals)
    for lam in [0.20, 0.30, 0.40, 0.50, 1.00]:
        p48_cal = (1 - lam) * blend[48] + lam * g_oof
        cal_dict = {h: blend[h].copy() for h in HORIZONS}
        cal_dict[48] = p48_cal
        pp = submission_postprocess(cal_dict)
        _print_score(f"lam={lam:.2f} (proj)", pp, y_time, y_event)

    # Also K=5
    print("\n=== Rank-bin diagnostic (K=5) ===")
    c5_u, c5_p, c5_v, c5_n = fit_rank_bins(p48_elig, y48_elig, K=5)
    print(f"  {'Bin':>4} {'u_center':>10} {'p_center':>10} "
          f"{'hit_rate':>10} {'n':>5}")
    for i in range(len(c5_u)):
        print(f"  {i:4d} {c5_u[i]:10.4f} {c5_p[i]:10.4f} "
              f"{c5_v[i]:10.4f} {c5_n[i]:5d}")

    # ==================================================================
    # Bootstrap bagging + shrink on anchor p48
    # ==================================================================
    print("\n=== Bootstrap rank-cal on anchor p48 (1000x) ===")

    configs = [
        (6, 0.25, "K6_lam025"),
        (6, 0.40, "K6_lam040"),
        (5, 0.30, "K5_lam030"),
    ]

    for K, lam, tag in configs:
        print(f"\n--- {tag} (K={K}, lam={lam}) ---")
        g_anchor = bootstrap_rank_cal(
            p48_elig, y48_elig, anchor_p48, K=K, n_boot=1000,
        )
        p48_new = (1 - lam) * anchor_p48 + lam * g_anchor
        p48_new = np.clip(p48_new, 1e-6, 1.0)

        safety_check(p48_new, anchor_p48, tag)

        # Five-number summary
        print(f"  p48 summary: min={p48_new.min():.4f} "
              f"p25={np.percentile(p48_new, 25):.4f} "
              f"med={np.median(p48_new):.4f} "
              f"p75={np.percentile(p48_new, 75):.4f} "
              f"max={p48_new.max():.4f}")

        path = f"submission_exp19c_{tag}.csv"
        make_submission(anchor, p48_new, path)

    print("\n=== Exp19c complete ===")


if __name__ == "__main__":
    main()
