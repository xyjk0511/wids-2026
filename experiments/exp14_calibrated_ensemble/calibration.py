"""Three calibration strategies for RSF+EST ensemble predictions (v2).

Strategy A: weibull_blend - Per-horizon weight search (RSF_EST vs Weibull)
Strategy B: adaptive_floor - Percentile replacement + KM power calibration
Strategy C: reference_quantile_map - Reference distribution quantile mapping
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from lifelines import KaplanMeierFitter

from src.config import HORIZONS
from src.evaluation import hybrid_score


def _km_targets(y_time, y_event):
    """Compute KM event probability P(T <= h) for each horizon."""
    kmf = KaplanMeierFitter()
    kmf.fit(y_time, y_event)
    return {h: float(1.0 - kmf.predict(h)) for h in HORIZONS}


# ---------------------------------------------------------------------------
# Strategy A: weibull_blend
# ---------------------------------------------------------------------------

def weibull_blend(blend_probs, weibull_probs, y_time, y_event):
    """Per-horizon weight search: final_h = w*blend + (1-w)*weibull.

    Sequential search 12h -> 24h -> 48h, each step fixes previous optima.
    Evaluates full hybrid_score at each candidate.

    Returns (calibrated_dict, info_dict with weights).
    """
    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event, dtype=int)

    weights = {}
    current = {h: blend_probs[h].copy() for h in HORIZONS}

    # Search order: 12h first (protects CI), then 24h, 48h (improve Brier)
    search_horizons = [12, 24, 48]
    candidates = np.arange(0.0, 1.05, 0.05)

    for h in search_horizons:
        best_w, best_score = 1.0, -np.inf

        for w in candidates:
            trial = {k: v.copy() for k, v in current.items()}
            trial[h] = w * blend_probs[h] + (1.0 - w) * weibull_probs[h]
            score, _ = hybrid_score(y_time, y_event, trial)
            if score > best_score:
                best_score = score
                best_w = w

        weights[h] = float(best_w)
        current[h] = best_w * blend_probs[h] + (1.0 - best_w) * weibull_probs[h]
        print(f"    h={h}: w={best_w:.2f} -> hybrid={best_score:.4f} "
              f"mean={current[h].mean():.4f}")

    # 72h stays as-is
    weights[72] = 1.0
    result = current
    print(f"    Weights: {weights}")
    return result, {"weights": weights}


def apply_weibull_blend(blend, weibull, info):
    """Apply stored weights to test predictions."""
    weights = info["weights"]
    result = {}
    for h in HORIZONS:
        w = weights.get(h, 1.0)
        result[h] = w * blend[h] + (1.0 - w) * weibull[h]
        print(f"    h={h}: w={w:.2f} mean={result[h].mean():.4f}")
    return result


# ---------------------------------------------------------------------------
# Strategy B: adaptive_floor
# ---------------------------------------------------------------------------

def adaptive_floor(blend_probs, weibull_probs, y_time, y_event):
    """Percentile-based replacement + KM power calibration.

    For each horizon:
    1. Search percentile p -> threshold = percentile(blend, p)
    2. Below threshold: replace with Weibull predictions
    3. KM power calibration: mean(merged^alpha) = KM(h)
    4. Pick p that maximizes hybrid_score

    Returns (calibrated_dict, info_dict with per-horizon params).
    """
    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event, dtype=int)
    km = _km_targets(y_time, y_event)

    percentile_candidates = [30, 40, 50, 55, 60, 65, 70]
    params = {}
    current = {h: blend_probs[h].copy() for h in HORIZONS}

    search_horizons = [12, 24, 48]

    for h in search_horizons:
        best_p, best_score, best_alpha = 0, -np.inf, 1.0
        best_calibrated = blend_probs[h].copy()

        for p in percentile_candidates:
            merged = blend_probs[h].copy()
            threshold = np.percentile(merged, p)
            low_mask = merged <= threshold
            merged[low_mask] = weibull_probs[h][low_mask]

            # KM power calibration
            target = km[h]
            alpha = _solve_power_alpha(merged, target)
            calibrated = np.where(merged > 0, merged ** alpha, 0.0)
            calibrated = np.clip(calibrated, 0.0, 1.0)

            trial = {k: v.copy() for k, v in current.items()}
            trial[h] = calibrated
            score, _ = hybrid_score(y_time, y_event, trial)

            if score > best_score:
                best_score = score
                best_p = p
                best_alpha = alpha
                best_calibrated = calibrated

        params[h] = {"percentile": best_p, "alpha": best_alpha}
        current[h] = best_calibrated
        n_replaced = int((blend_probs[h] <= np.percentile(blend_probs[h], best_p)).sum())
        print(f"    h={h}: p={best_p}% alpha={best_alpha:.4f} "
              f"replaced={n_replaced}/{len(blend_probs[h])} "
              f"hybrid={best_score:.4f} mean={best_calibrated.mean():.4f} (KM={km[h]:.4f})")

    # 72h untouched
    params[72] = {"percentile": 0, "alpha": 1.0}
    return current, {"params": params}


def _solve_power_alpha(p, target):
    """Solve mean(p^alpha) = target via Brent's method."""
    p = np.asarray(p, dtype=float)
    pos = p > 0

    if pos.sum() == 0:
        return 1.0

    def obj(alpha):
        cal = np.zeros_like(p)
        cal[pos] = p[pos] ** alpha
        return cal.mean() - target

    try:
        return brentq(obj, 0.05, 20.0)
    except ValueError:
        return 1.0


def apply_adaptive_floor(blend, weibull, info):
    """Apply stored (percentile, alpha) params to test predictions."""
    params = info["params"]
    result = {}
    for h in HORIZONS:
        hp = params.get(h, {"percentile": 0, "alpha": 1.0})
        p_pct, alpha = hp["percentile"], hp["alpha"]

        if p_pct == 0 and alpha == 1.0:
            result[h] = blend[h].copy()
            continue

        merged = blend[h].copy()
        threshold = np.percentile(merged, p_pct)
        low_mask = merged <= threshold
        merged[low_mask] = weibull[h][low_mask]

        calibrated = np.where(merged > 0, merged ** alpha, 0.0)
        result[h] = np.clip(calibrated, 0.0, 1.0)
        print(f"    h={h}: p={p_pct}% alpha={alpha:.4f} "
              f"replaced={low_mask.sum()}/{len(blend[h])} "
              f"mean={result[h].mean():.4f}")
    return result


# ---------------------------------------------------------------------------
# Strategy C: reference_quantile_map
# ---------------------------------------------------------------------------

_REF_PATH = "submission 0.96624.csv"


def reference_quantile_map(blend_probs, weibull_probs, y_time, y_event):
    """Map OOF predictions to reference submission's distribution via quantile mapping.

    1. Load reference submission
    2. For each horizon: rank blend -> uniform quantile -> interp to reference
    3. OOF uses np.interp with sorted reference values

    Returns (calibrated_dict, info_dict with sorted reference arrays).
    """
    import pandas as pd

    try:
        ref = pd.read_csv(_REF_PATH)
    except FileNotFoundError:
        print(f"    [SKIP] '{_REF_PATH}' not found")
        return {h: blend_probs[h].copy() for h in HORIZONS}, {}

    prob_cols = [f"prob_{h}h" for h in HORIZONS]
    ref_sorted = {}
    result = {}

    for h, col in zip(HORIZONS, prob_cols):
        if h == 72:
            result[h] = blend_probs[h].copy()
            ref_sorted[h] = np.ones(1)
            continue

        ref_vals = np.sort(ref[col].values)
        ref_sorted[h] = ref_vals

        p = blend_probs[h]
        n = len(p)

        # Rank -> uniform quantile in (0, 1)
        ranks = p.argsort().argsort()
        u = (ranks + 0.5) / n

        # Interpolate into reference distribution
        n_ref = len(ref_vals)
        ref_quantiles = (np.arange(n_ref) + 0.5) / n_ref
        calibrated = np.interp(u, ref_quantiles, ref_vals)

        result[h] = np.clip(calibrated, 0.0, 1.0)
        print(f"    h={h}: mean={result[h].mean():.4f} "
              f"median={np.median(result[h]):.4f} "
              f"ref_mean={ref_vals.mean():.4f}")

    return result, {"ref_sorted": ref_sorted}


def apply_reference_map(blend, weibull, info):
    """Apply reference quantile mapping to test predictions.

    Test set (n=95) matches reference (n=95), so direct rank mapping.
    """
    ref_sorted = info.get("ref_sorted", {})
    if not ref_sorted:
        print("    [SKIP] No reference data stored")
        return {h: blend[h].copy() for h in HORIZONS}

    result = {}
    for h in HORIZONS:
        if h == 72 or h not in ref_sorted:
            result[h] = blend[h].copy()
            continue

        ref_vals = ref_sorted[h]
        p = blend[h]
        n = len(p)

        ranks = p.argsort().argsort()
        u = (ranks + 0.5) / n

        n_ref = len(ref_vals)
        ref_quantiles = (np.arange(n_ref) + 0.5) / n_ref
        calibrated = np.interp(u, ref_quantiles, ref_vals)

        result[h] = np.clip(calibrated, 0.0, 1.0)
        print(f"    h={h}: mean={result[h].mean():.4f} "
              f"median={np.median(result[h]):.4f}")

    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES = {
    "weibull_blend": weibull_blend,
    "adaptive_floor": adaptive_floor,
    "reference_quantile_map": reference_quantile_map,
}

APPLY_FNS = {
    "weibull_blend": apply_weibull_blend,
    "adaptive_floor": apply_adaptive_floor,
    "reference_quantile_map": apply_reference_map,
}


# ---------------------------------------------------------------------------
# v3: Piecewise power calibration
# ---------------------------------------------------------------------------

MILD_PARAMS = {12: (0.0385, 0.15), 24: (0.0723, 0.20), 48: (0.0853, 0.25)}


def piecewise_power(p, tau, alpha):
    """g(p) = tau*(p/tau)^alpha if p<tau, else p."""
    result = np.asarray(p, dtype=float).copy()
    mask = result < tau
    if mask.any():
        result[mask] = tau * (result[mask] / tau) ** alpha
    return result


def compute_match_ref_params(prob_dict, ref_path="submission 0.96624.csv"):
    """Dynamically compute (tau, alpha) per horizon from reference submission.

    tau = ref p25, alpha = ln(ref_q10/tau) / ln(cur_q10/tau).
    Uses q10 instead of min/p1 for stability.
    """
    try:
        ref = pd.read_csv(ref_path)
    except FileNotFoundError:
        print(f"    [WARN] '{ref_path}' not found, falling back to MILD_PARAMS")
        return dict(MILD_PARAMS)

    params = {}
    for h in [12, 24, 48]:
        col = f"prob_{h}h"
        ref_vals = ref[col].values
        cur_vals = prob_dict[h]
        tau = float(np.percentile(ref_vals, 25))
        cur_q10 = float(np.percentile(cur_vals, 10))
        ref_q10 = float(np.percentile(ref_vals, 10))
        if cur_q10 > 0 and cur_q10 < tau and ref_q10 < tau:
            alpha = np.log(ref_q10 / tau) / np.log(cur_q10 / tau)
            alpha = float(np.clip(alpha, 0.05, 0.60))
        else:
            alpha = 1.0
        params[h] = (tau, alpha)
        print(f"    h={h}: tau={tau:.4f} alpha={alpha:.4f} "
              f"(cur_q10={cur_q10:.4f} -> ~{piecewise_power(np.array([cur_q10]), tau, alpha)[0]:.4f})")
    return params


def apply_piecewise_power(prob_dict, params):
    """Apply piecewise power to 12/24/48h; 72h untouched."""
    result = {}
    for h in HORIZONS:
        if h in params:
            tau, alpha = params[h]
            result[h] = piecewise_power(prob_dict[h], tau, alpha)
        else:
            result[h] = prob_dict[h].copy()
    return result
