"""Monotonicity enforcement and calibration for horizon probabilities."""

import numpy as np
from lifelines import KaplanMeierFitter
from scipy.optimize import brentq

from src.config import HORIZONS


def enforce_monotonicity(
    prob_dict: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Ensure prob_12h <= prob_24h <= prob_48h <= prob_72h per sample.

    Uses max-forward: each horizon takes max(self, previous horizon).
    """
    result = {}
    prev = None
    for h in HORIZONS:
        p = np.asarray(prob_dict[h], dtype=float).copy()
        if prev is not None:
            p = np.maximum(p, prev)
        p = np.clip(p, 0.0, 1.0)
        result[h] = p
        prev = p
    return result


def _pava_1d(values, weights):
    """Weighted L2 isotonic regression (non-decreasing) via PAVA."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.ndim != 1:
        raise ValueError("PAVA expects 1D values.")
    if w.shape != v.shape:
        raise ValueError("weights shape must match values shape.")

    # Use tiny positive lower bound to avoid zero-weight blocks.
    w = np.clip(w, 1e-12, None)

    means = []
    masses = []
    starts = []
    ends = []

    for i, (vi, wi) in enumerate(zip(v, w)):
        means.append(float(vi))
        masses.append(float(wi))
        starts.append(i)
        ends.append(i)

        while len(means) >= 2 and means[-2] > means[-1]:
            m0, m1 = means[-2], means[-1]
            w0, w1 = masses[-2], masses[-1]
            new_w = w0 + w1
            new_m = (w0 * m0 + w1 * m1) / new_w

            means[-2] = new_m
            masses[-2] = new_w
            ends[-2] = ends[-1]

            means.pop()
            masses.pop()
            starts.pop()
            ends.pop()

    out = np.empty_like(v)
    for m, s, e in zip(means, starts, ends):
        out[s : e + 1] = m
    return out


def project_monotone_l2(
    prob_dict: dict[int, np.ndarray],
    horizons: list[int] | None = None,
    floor: float = 0.0,
    ceiling: float = 1.0,
    weights: list[float] | None = None,
    fix_72_to_one: bool = False,
) -> dict[int, np.ndarray]:
    """Project probabilities to monotone sequence with minimum L2 change."""
    horizons = horizons or HORIZONS
    n = len(next(iter(prob_dict.values())))
    w = np.asarray(weights if weights is not None else [1.0] * len(horizons), dtype=float)
    if w.shape[0] != len(horizons):
        raise ValueError("weights length must match horizons length.")

    mat = np.column_stack([np.asarray(prob_dict[h], dtype=float) for h in horizons])
    mat = np.clip(mat, floor, ceiling)

    if fix_72_to_one and 72 in horizons:
        h72_idx = horizons.index(72)
    else:
        h72_idx = None

    proj = np.zeros_like(mat, dtype=float)
    for i in range(n):
        row = mat[i].copy()
        if h72_idx is not None:
            row[h72_idx] = 1.0
        row_proj = _pava_1d(row, w)
        if h72_idx is not None:
            row_proj[h72_idx] = 1.0
        proj[i] = np.clip(row_proj, floor, ceiling)

    return {h: proj[:, j] for j, h in enumerate(horizons)}


def submission_postprocess(
    prob_dict,
    floor=1e-6,
    floor_12=None,
    floor_24_48=None,
    use_projection=True,
    cap_12_by_24_eps=None,
):
    """Competition postprocessing (split chain):
    - 12h: clip independently (only for C-index ranking).
    - 24h/48h/72h: separate monotonicity chain starting from raw_24h.
    - 72h: hardcoded 1.0
    - 24h/48h: clip with dedicated floor
    """
    floor_12 = float(floor if floor_12 is None else floor_12)
    floor_24_48 = float(floor if floor_24_48 is None else floor_24_48)

    result = {}
    p12 = np.clip(prob_dict[12].copy(), floor_12, 0.99)
    result[12] = p12

    p24_48 = {
        24: np.clip(prob_dict[24].copy(), floor_24_48, 0.99),
        48: np.clip(prob_dict[48].copy(), floor_24_48, 0.99),
        72: np.ones(len(prob_dict[24])),
    }

    if use_projection:
        proj = project_monotone_l2(
            p24_48,
            horizons=[24, 48, 72],
            floor=floor_24_48,
            ceiling=1.0,
            weights=[1.0, 1.0, 10.0],
            fix_72_to_one=True,
        )
        result.update(proj)
    else:
        prev = p24_48[24]
        result[24] = prev
        for t in [48, 72]:
            current = np.maximum(p24_48[t], prev)
            result[t] = current
            prev = current
        result[72] = np.ones(len(result[72]))

    # cap p12 by projected p24 (after projection to avoid residual violations)
    if cap_12_by_24_eps is not None:
        eps = float(cap_12_by_24_eps)
        result[12] = np.clip(np.minimum(result[12], result[24] - eps), 0.0, 0.99)

    return result


def submission_postprocess_full_mono(
    prob_dict,
    floor=1e-6,
    use_projection=True,
    cap_12_by_24_eps=None,
):
    """Full-chain monotonic postprocessing (all 4 horizons linked):
    - clip each horizon to [floor, 0.99]
    - 72h: hardcoded 1.0
    - enforce 12h <= 24h <= 48h <= 72h
    """
    result = {}
    for h in [12, 24, 48]:
        result[h] = np.clip(prob_dict[h].copy(), floor, 0.99)
    if cap_12_by_24_eps is not None:
        eps = float(cap_12_by_24_eps)
        result[12] = np.minimum(result[12], result[24] - eps)
        result[12] = np.clip(result[12], 0.0, 0.99)
    result[72] = np.ones(len(result[12]))
    if use_projection:
        return project_monotone_l2(
            result,
            horizons=[12, 24, 48, 72],
            floor=floor,
            ceiling=1.0,
            weights=[1.0, 1.0, 1.0, 10.0],
            fix_72_to_one=True,
        )
    return enforce_monotonicity(result)


def km_power_calibrate(prob_dict, y_time, y_event):
    """Power-law calibration: find alpha_h so mean(pred^alpha) = KM(h).

    Preserves ranking (CI unchanged) while calibrating marginal probabilities
    to match Kaplan-Meier estimates (improves Brier score).
    """
    kmf = KaplanMeierFitter()
    kmf.fit(y_time, y_event)

    result = {}
    alphas = {}
    for h in HORIZONS:
        preds = np.asarray(prob_dict[h], dtype=float)
        if h == 72:
            result[h] = preds.copy()
            alphas[h] = 1.0
            continue

        km_target = 1.0 - kmf.predict(h)

        def obj(alpha):
            return np.where(preds > 0, preds ** alpha, 0.0).mean() - km_target

        alpha = brentq(obj, 0.05, 20.0)
        result[h] = np.where(preds > 0, preds ** alpha, 0.0)
        alphas[h] = alpha
        print(f"    h={h}: alpha={alpha:.4f}, mean={result[h].mean():.4f} (KM={km_target:.4f})")

    return result, alphas
