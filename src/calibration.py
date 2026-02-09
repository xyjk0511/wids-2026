"""Probability calibration via Platt Scaling."""

import numpy as np
from sklearn.linear_model import LogisticRegression


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
