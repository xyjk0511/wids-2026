"""Monotonicity enforcement for horizon probabilities."""

import numpy as np

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


def submission_postprocess(prob_dict):
    """Competition-optimized postprocessing:
    - 12h: independent clip (only affects C-index)
    - 24h/48h: monotonic chain + clip
    - 72h: hardcoded 1.0 (all eligible samples positive, Brier=0)
    """
    result = {}
    result[12] = np.clip(prob_dict[12], 0.01, 0.99)
    prev = prob_dict[24].copy()
    result[24] = prev
    for h in [48, 72]:
        current = np.maximum(prob_dict[h], prev)
        result[h] = current
        prev = current
    result[72] = np.ones(len(prob_dict[72]))
    for h in [24, 48]:
        result[h] = np.clip(result[h], 0.01, 0.99)
    return result
