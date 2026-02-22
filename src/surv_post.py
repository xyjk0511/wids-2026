"""Deterministic survival-function to probability conversion helpers."""

from __future__ import annotations

import numpy as np


def _get_domain(fn) -> tuple[float, float]:
    """Return valid evaluation interval for a survival step function."""
    if hasattr(fn, "domain") and fn.domain is not None:
        lo, hi = fn.domain
        return float(lo), float(hi)

    if hasattr(fn, "x"):
        x = np.asarray(fn.x, dtype=float)
        if x.size == 0:
            raise ValueError("Step function has empty x grid.")
        return float(x[0]), float(x[-1])

    raise ValueError("Unsupported survival function type: missing domain/x.")


def _eval_survival(fn, t_eval: float) -> float:
    """Evaluate a survival step function at one time point."""
    value = fn(float(t_eval))
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Step function returned empty value.")
    return float(arr[0])


def sf_to_cdf(fn, horizon: float, policy: str = "clip") -> float:
    """Convert one survival function to cumulative event probability.

    Returns p(T <= horizon) = 1 - S(horizon), with explicit boundary policy.

    Supported policies:
    - clip: clip horizon to [lo, hi].
    - strict: require horizon in [lo, hi], otherwise raise.
    - left_survival_one: if horizon < lo, return p=0 directly.
    """
    lo, hi = _get_domain(fn)
    h = float(horizon)

    if policy == "clip":
        t_eval = float(np.clip(h, lo, hi))
        try:
            s_val = _eval_survival(fn, t_eval)
        except ValueError:
            # Some versions can still throw on exact boundary values.
            eps = np.finfo(float).eps
            t_safe = float(np.clip(t_eval, lo + eps, hi - eps))
            s_val = _eval_survival(fn, t_safe)
    elif policy == "strict":
        if h < lo or h > hi:
            raise ValueError(f"horizon {h} outside [{lo}, {hi}]")
        s_val = _eval_survival(fn, h)
    elif policy == "left_survival_one":
        if h < lo:
            s_val = 1.0
        else:
            t_eval = min(h, hi)
            s_val = _eval_survival(fn, t_eval)
    else:
        raise ValueError(f"Unknown StepFunction policy: {policy}")

    return float(np.clip(1.0 - s_val, 0.0, 1.0))


def surv_fns_to_probs(
    surv_fns,
    horizons: list[int] | tuple[int, ...],
    policy: str = "clip",
) -> dict[int, np.ndarray]:
    """Vectorize sf_to_cdf across samples and horizons."""
    surv_fns = list(surv_fns)
    result = {}
    for h in horizons:
        probs = np.empty(len(surv_fns), dtype=float)
        for i, fn in enumerate(surv_fns):
            probs[i] = sf_to_cdf(fn, h, policy=policy)
        result[int(h)] = probs
    return result
