"""Probability calibration utilities."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar


def platt_scaling(
    probs: np.ndarray,
    labels: np.ndarray,
) -> LogisticRegression:
    """Fit a Platt scaling calibrator (logistic regression on raw probs).

    Args:
        probs: Raw predicted probabilities, shape (n,).
        labels: Binary ground truth, shape (n,).

    Returns:
        Fitted LogisticRegression model.
    """
    probs = np.asarray(probs, dtype=float).reshape(-1, 1)
    labels = np.asarray(labels, dtype=float)
    lr = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000)
    lr.fit(probs, labels)
    return lr


def calibrate(
    calibrator: LogisticRegression,
    probs: np.ndarray,
) -> np.ndarray:
    """Apply fitted calibrator to raw probabilities."""
    probs = np.asarray(probs, dtype=float).reshape(-1, 1)
    return calibrator.predict_proba(probs)[:, 1]


def odds_scale(
    probs: np.ndarray,
    scale: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Monotone 1-parameter mapping on odds.

    p' = (scale * p/(1-p)) / (1 + scale * p/(1-p))
    """
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    odds = p / (1.0 - p)
    odds_scaled = float(scale) * odds
    return odds_scaled / (1.0 + odds_scaled)


def fit_odds_scale_brier(
    probs: np.ndarray,
    labels: np.ndarray,
    bounds: tuple[float, float] = (1e-3, 1e3),
) -> float:
    """Fit a single odds scale that minimizes Brier score."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)

    if p.size == 0:
        return 1.0

    def objective(log_scale: float) -> float:
        scale = float(np.exp(log_scale))
        p2 = odds_scale(p, scale=scale)
        return float(np.mean((p2 - y) ** 2))

    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    res = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
    if not res.success:
        return 1.0
    return float(np.exp(res.x))
