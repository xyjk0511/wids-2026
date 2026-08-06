"""Exp19b: Quantile-Transport + PAVA increment calibration.

Core idea: PAVA learned on OOF can't directly transfer to anchor (different
distribution). Solution: quantile-transport anchor p48 into OOF domain,
apply PAVA there, then use only the *increment* blended back at safe alpha.

p_final = p_anchor + alpha * (g(p_tilde) - p_tilde)
where p_tilde = F_oof_inv(F_anchor(p_anchor))  [quantile transport]

Usage:
    python -m scripts.exp19b_quantile_pava
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import (
    HORIZONS, PROB_COLS, TIME_COL, EVENT_COL, SAMPLE_SUB_PATH,
)
from src.labels import build_horizon_labels
from src.evaluation import hybrid_score, horizon_brier_score, weighted_brier_score
from src.monotonic import submission_postprocess
from src.train import load_data, run_cv, _print_score


ANCHOR_PATH = "submission_0.96624.csv"


# ---------------------------------------------------------------------------
# Quantile transport
# ---------------------------------------------------------------------------
def quantile_transport(p_source, p_ref_from, p_ref_to):
    """Map p_source from ref_from distribution to ref_to distribution.

    For each value in p_source, find its quantile in ref_from,
    then map to the same quantile in ref_to.
    """
    p_source = np.asarray(p_source, dtype=float)
    # Compute empirical CDF of ref_from
    sorted_from = np.sort(p_ref_from)
    sorted_to = np.sort(p_ref_to)
    n_from = len(sorted_from)
    n_to = len(sorted_to)

    # For each source value, find its quantile in ref_from
    # then map to same quantile in ref_to
    ranks = np.searchsorted(sorted_from, p_source, side="right")
    quantiles = ranks / n_from  # [0, 1]
    quantiles = np.clip(quantiles, 0.5 / n_to, 1.0 - 0.5 / n_to)

    # Map quantile to ref_to value
    indices = (quantiles * n_to).astype(int)
    indices = np.clip(indices, 0, n_to - 1)
    return sorted_to[indices]


# ---------------------------------------------------------------------------
# Smoothed bin-PAVA (more bins + linear interp for smoother mapping)
# ---------------------------------------------------------------------------
def pava_increasing(values):
    """Pool Adjacent Violators for non-decreasing sequence."""
    n = len(values)
    result = np.array(values, dtype=float)
    block_start = np.arange(n)
    block_end = np.arange(n)

    i = 0
    while i < n - 1:
        if result[i] > result[i + 1]:
            # Merge blocks
            s = block_start[i]
            e = block_end[i + 1]
            total = sum(result[j] for j in range(s, e + 1))
            avg = total / (e - s + 1)
            for j in range(s, e + 1):
                result[j] = avg
                block_start[j] = s
                block_end[j] = e
            # Step back
            i = max(0, s - 1) if s > 0 else 0
        else:
            i += 1
    return result


def fit_smooth_pava(probs, labels, n_bins=20):
    """Fit smoothed bin-PAVA: more bins + PAVA + linear interpolation."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(p, quantiles))
    if len(edges) < 3:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    bin_idx = np.digitize(p, edges[1:-1])
    centers, means = [], []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        centers.append(float(p[mask].mean()))
        means.append(float(y[mask].mean()))

    if len(centers) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    cal_vals = pava_increasing(means)
    return np.array(centers), np.array(cal_vals)


def apply_pava(probs, centers, cal_vals):
    """Apply PAVA mapping via linear interpolation."""
    return np.interp(probs, centers, cal_vals,
                     left=cal_vals[0], right=cal_vals[-1])


# ---------------------------------------------------------------------------
# Bootstrap bagging for smooth PAVA
# ---------------------------------------------------------------------------
def bootstrap_pava_bag(p_train, y_train, p_apply, n_boot=1000,
                       n_bins=20, seed=42):
    """Bootstrap smooth-PAVA, apply each to p_apply, return average."""
    rng = np.random.default_rng(seed)
    n = len(p_train)
    accum = np.zeros_like(p_apply, dtype=float)

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        centers, cal_vals = fit_smooth_pava(p_train[idx], y_train[idx],
                                           n_bins=n_bins)
        accum += apply_pava(p_apply, centers, cal_vals)

    return accum / n_boot


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------
def safety_check(p_final, p_anchor, label):
    """Print safety metrics for a calibrated submission."""
    sr, _ = spearmanr(p_final, p_anchor)
    mad = np.mean(np.abs(p_final - p_anchor))
    print(f"  [{label}] Spearman vs anchor: {sr:.6f}  "
          f"MAD: {mad:.6f}  "
          f"median: {np.median(p_anchor):.4f}->{np.median(p_final):.4f}")
    if sr < 0.995:
        print(f"    WARNING: Spearman {sr:.4f} < 0.995 threshold")
    return sr, mad


def make_submission(anchor, p48_new, path):
    """Replace p48 in anchor, postprocess, save."""
    prob_dict = {
        12: anchor["prob_12h"].values.copy(),
        24: anchor["prob_24h"].values.copy(),
        48: np.clip(p48_new.copy(), 1e-6, 1.0),
        72: np.ones(len(anchor)),
    }
    pp = submission_postprocess(prob_dict)

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = pp[h]

    h72_ok = (sub["prob_72h"] == 1.0).all()
    mono_violations = ((sub["prob_24h"] > sub["prob_48h"] + 1e-9).sum()
                       + (sub["prob_48h"] > sub["prob_72h"] + 1e-9).sum())
    print(f"  72h={('PASS' if h72_ok else 'FAIL')}  "
          f"mono_violations={mono_violations}  Shape: {sub.shape}")

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
    print("=== Exp19b: Quantile-Transport + PAVA Increment ===\n")

    anchor = pd.read_csv(ANCHOR_PATH)
    anchor_p48 = anchor["prob_48h"].values.copy()
    print(f"  Anchor: {ANCHOR_PATH} ({len(anchor)} rows)")
    print(f"  Anchor p48: min={anchor_p48.min():.4f} "
          f"med={np.median(anchor_p48):.4f} max={anchor_p48.max():.4f}")

    # Load data + OOF
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
    p48_oof = blend[48]
    p48_elig = p48_oof[elig48]
    y48_elig = labels48[elig48]
    print(f"\n  48h eligible: n={elig48.sum()}, "
          f"pos_rate={y48_elig.mean():.3f}")
    print(f"  OOF p48: min={p48_oof.min():.4f} "
          f"med={np.median(p48_oof):.4f} max={p48_oof.max():.4f}")

    # ==================================================================
    # Step 1: Quantile transport anchor -> OOF domain
    # ==================================================================
    print("\n=== Step 1: Quantile Transport ===")
    p_tilde = quantile_transport(anchor_p48, anchor_p48, p48_oof)
    print(f"  Transported p_tilde: min={p_tilde.min():.4f} "
          f"med={np.median(p_tilde):.4f} max={p_tilde.max():.4f}")
    sr_transport, _ = spearmanr(p_tilde, anchor_p48)
    print(f"  Spearman(p_tilde, anchor): {sr_transport:.6f} "
          f"(should be ~1.0, rank-preserving)")

    # ==================================================================
    # Step 2: Learn PAVA on OOF eligible (smoothed, 20 bins)
    # ==================================================================
    print("\n=== Step 2: Smooth PAVA (20 bins) on OOF eligible ===")
    centers, cal_vals = fit_smooth_pava(p48_elig, y48_elig, n_bins=20)
    print(f"  PAVA mapping ({len(centers)} points):")
    for c, v in zip(centers, cal_vals):
        print(f"    p={c:.4f} -> cal={v:.4f}")

    # Apply to full OOF for diagnostic
    p48_oof_cal = apply_pava(p48_oof, centers, cal_vals)
    cal_dict = {h: blend[h].copy() for h in HORIZONS}
    cal_dict[48] = p48_oof_cal
    pp_cal = submission_postprocess(cal_dict)
    _print_score("PAVA OOF (proj)", pp_cal, y_time, y_event)

    # ==================================================================
    # Step 3: Bootstrap PAVA bagging on transported anchor p48
    # ==================================================================
    print("\n=== Step 3: Bootstrap PAVA bagging (1000x, 20 bins) ===")
    g_tilde = bootstrap_pava_bag(
        p48_elig, y48_elig, p_tilde, n_boot=1000, n_bins=20,
    )
    increment = g_tilde - p_tilde
    print(f"  Increment stats: min={increment.min():.4f} "
          f"med={np.median(increment):.4f} max={increment.max():.4f}")
    print(f"  mean(|increment|)={np.mean(np.abs(increment)):.4f}")

    # ==================================================================
    # Step 4: Alpha blending + submissions
    # ==================================================================
    print("\n=== Step 4: Alpha blending ===")
    alphas = [0.25, 0.40]

    for alpha in alphas:
        print(f"\n--- alpha={alpha:.2f} ---")
        p48_final = anchor_p48 + alpha * increment
        p48_final = np.clip(p48_final, 1e-6, 1.0)

        safety_check(p48_final, anchor_p48, f"a={alpha}")

        path = f"submission_exp19b_a{alpha:.0e}.csv".replace(
            "+", "").replace("e-0", "e-").replace("e0", "e")
        # Cleaner naming
        tag = str(alpha).replace(".", "")
        path = f"submission_exp19b_a{tag}.csv"
        make_submission(anchor, p48_final, path)

    # ==================================================================
    # Also: direct PAVA on anchor (no increment, for comparison)
    # ==================================================================
    print("\n=== Alt: Direct bagged PAVA on anchor p48 (no transport) ===")
    g_direct = bootstrap_pava_bag(
        p48_elig, y48_elig, anchor_p48, n_boot=1000, n_bins=20,
    )
    safety_check(g_direct, anchor_p48, "direct PAVA")
    make_submission(anchor, g_direct, "submission_exp20_anchor_pava.csv")

    print("\n=== Exp19b complete ===")


if __name__ == "__main__":
    main()
