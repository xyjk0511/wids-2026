"""Weighted ensemble with weight optimization and stacking meta-learner."""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from src.config import HORIZONS
from src.evaluation import (
    combined_score, mean_brier_score, c_index, hybrid_score,
    horizon_brier_score,
)


def ensemble_predict(
    model_preds: list[dict[int, np.ndarray]],
    weights: np.ndarray,
) -> dict[int, np.ndarray]:
    """Weighted average of model predictions.

    Args:
        model_preds: List of {horizon: prob_array} from each model.
        weights: Array of weights summing to 1.

    Returns:
        {horizon: weighted_avg_prob_array}
    """
    weights = np.asarray(weights, dtype=float)
    result = {}
    for h in HORIZONS:
        stacked = np.column_stack([mp[h] for mp in model_preds])
        result[h] = stacked @ weights
    return result


def optimize_weights(
    model_preds: list[dict[int, np.ndarray]],
    y_time: np.ndarray,
    y_event: np.ndarray,
    n_models: int | None = None,
) -> np.ndarray:
    """Find optimal ensemble weights using competition hybrid score."""
    n = n_models or len(model_preds)

    def objective(w):
        w = w / w.sum()
        ens = ensemble_predict(model_preds, w)
        score, _ = hybrid_score(y_time, y_event, ens)
        uniform = np.ones(len(w)) / len(w)
        penalty = 0.1 * np.sum((w - uniform) ** 2)
        return -score + penalty

    # constraints: weights sum to 1
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    # bounds: each weight in [0, 1]
    bounds = [(0.0, 1.0)] * n
    # start with equal weights
    w0 = np.ones(n) / n

    res = minimize(
        objective, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-8},
    )
    weights = res.x / res.x.sum()
    return weights


def robust_optimize_weights(model_preds, y_time, y_event, n_bootstrap=30):
    """Bootstrap-robust ensemble weight optimization.

    Runs optimize_weights on multiple bootstrap samples and returns
    the median weights, reducing sensitivity to specific samples.
    """
    all_weights = []
    n = len(y_time)
    for seed in range(n_bootstrap):
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, n, replace=True)
        sampled = [{h: mp[h][idx] for h in HORIZONS} for mp in model_preds]
        w = optimize_weights(sampled, y_time[idx], y_event[idx])
        all_weights.append(w)
    median_w = np.median(all_weights, axis=0)
    median_w = median_w / median_w.sum()
    return median_w


def _optimize_brier_weights(model_preds, y_time, y_event, horizon):
    """Optimize weights for a single horizon minimizing Brier + L2."""
    n = len(model_preds)

    def objective(w):
        w = w / w.sum()
        stacked = np.column_stack([mp[horizon] for mp in model_preds])
        ens_h = stacked @ w
        brier = horizon_brier_score(y_time, y_event, ens_h, horizon)
        uniform = np.ones(n) / n
        penalty = 0.1 * np.sum((w - uniform) ** 2)
        return brier + penalty

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n
    res = minimize(
        objective, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-8},
    )
    return res.x / res.x.sum()


def optimize_weights_per_horizon(model_preds, y_time, y_event):
    """Optimize ensemble weights independently for each horizon.

    Returns:
        dict {horizon: weights_array}
        12h uses global weights (stacking handles it separately).
        24h/48h each minimize Brier@h + L2 regularization.
        72h uses uniform weights (hardcoded to 1.0 in postprocess).
    """
    n = len(model_preds)
    result = {}
    # 12h: use global hybrid-optimized weights (stacking replaces later)
    result[12] = optimize_weights(model_preds, y_time, y_event)
    # 24h, 48h: per-horizon Brier optimization
    for h in [24, 48]:
        w = _optimize_brier_weights(model_preds, y_time, y_event, h)
        result[h] = w
    # 72h: uniform (postprocess sets to 1.0 anyway)
    result[72] = np.ones(n) / n
    return result


def ensemble_predict_per_horizon(model_preds, weights_dict):
    """Weighted average using per-horizon weights.

    Args:
        model_preds: List of {horizon: prob_array} from each model.
        weights_dict: {horizon: weights_array} from optimize_weights_per_horizon.

    Returns:
        {horizon: weighted_avg_prob_array}
    """
    result = {}
    for h in HORIZONS:
        w = np.asarray(weights_dict[h], dtype=float)
        stacked = np.column_stack([mp[h] for mp in model_preds])
        result[h] = stacked @ w
    return result


def stacking_meta_learner(oof_preds_list, y_time, y_event, model_names):
    """Train a LogisticRegression meta-learner on OOF 12h predictions.

    Args:
        oof_preds_list: List of {horizon: array} for each model (OOF).
        y_time, y_event: Ground truth arrays.
        model_names: List of model name strings.

    Returns:
        Fitted LogisticRegression model, eligible mask used for training.
    """
    from src.labels import build_horizon_labels

    labels, eligible = build_horizon_labels(y_time, y_event, 12)

    # Stack 12h OOF predictions from all models as features
    X_meta = np.column_stack([
        oof_preds[12][eligible] for oof_preds in oof_preds_list
    ])
    y_meta = labels[eligible]

    meta = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    meta.fit(X_meta, y_meta)

    # Report meta-learner coefficients
    print("  Stacking meta-learner coefficients:")
    for name, coef in zip(model_names, meta.coef_[0]):
        print(f"    {name:10s}: {coef:.4f}")

    return meta


def stacking_predict_12h(meta, model_preds_list):
    """Apply trained meta-learner to produce stacked 12h predictions.

    Args:
        meta: Fitted LogisticRegression from stacking_meta_learner().
        model_preds_list: List of {horizon: array} for each model (test set).

    Returns:
        1D array of stacked 12h probabilities.
    """
    X_meta = np.column_stack([mp[12] for mp in model_preds_list])
    return meta.predict_proba(X_meta)[:, 1]
