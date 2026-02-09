import numpy as np
from sksurv.metrics import concordance_index_censored

from src.config import HORIZONS


def horizon_brier_score(
    y_time: np.ndarray,
    y_event: np.ndarray,
    probs: np.ndarray,
    horizon: float,
) -> float:
    """Compute Brier Score for a single horizon.

    Eligible samples:
      - event=1 AND time <= horizon  -> label=1
      - event=1 AND time > horizon   -> label=0
      - event=0 AND time >= horizon  -> label=0
      - event=0 AND time < horizon   -> excluded (censored before horizon)
    """
    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event, dtype=int)
    probs = np.asarray(probs, dtype=float)

    labels = np.full(len(y_time), np.nan)

    # event=1, hit before or at horizon -> label 1
    mask_hit = (y_event == 1) & (y_time <= horizon)
    labels[mask_hit] = 1.0

    # event=1, hit after horizon -> label 0 (survived this horizon)
    mask_survived_event = (y_event == 1) & (y_time > horizon)
    labels[mask_survived_event] = 0.0

    # event=0, observed at least until horizon -> label 0
    mask_censored_ok = (y_event == 0) & (y_time >= horizon)
    labels[mask_censored_ok] = 0.0

    # event=0, censored before horizon -> excluded
    eligible = ~np.isnan(labels)
    if eligible.sum() == 0:
        return 0.0

    return float(np.mean((probs[eligible] - labels[eligible]) ** 2))


def mean_brier_score(
    y_time: np.ndarray,
    y_event: np.ndarray,
    prob_dict: dict[int, np.ndarray],
) -> float:
    """Average Brier Score across all horizons."""
    scores = []
    for h in HORIZONS:
        scores.append(horizon_brier_score(y_time, y_event, prob_dict[h], h))
    return float(np.mean(scores))


def c_index(
    y_time: np.ndarray,
    y_event: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    """Concordance index using prob_12h as risk score.

    Higher prob_12h = higher risk = shorter survival.
    """
    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event, dtype=bool)
    risk_scores = np.asarray(risk_scores, dtype=float)

    ci, _, _, _, _ = concordance_index_censored(y_event, y_time, risk_scores)
    return float(ci)


def combined_score(
    y_time: np.ndarray,
    y_event: np.ndarray,
    prob_dict: dict[int, np.ndarray],
) -> float:
    """Combined metric: mean Brier Score + (1 - C-index).

    Lower is better. Used for ensemble weight optimization.
    """
    brier = mean_brier_score(y_time, y_event, prob_dict)
    ci = c_index(y_time, y_event, prob_dict[12])
    return brier + (1.0 - ci)


BRIER_WEIGHTS = {24: 0.3, 48: 0.4, 72: 0.3}


def weighted_brier_score(y_time, y_event, prob_dict):
    """Competition weighted Brier: 0.3*B@24h + 0.4*B@48h + 0.3*B@72h (no 12h)."""
    score = 0.0
    for h, w in BRIER_WEIGHTS.items():
        score += w * horizon_brier_score(y_time, y_event, prob_dict[h], h)
    return score


def hybrid_score(y_time, y_event, prob_dict):
    """Competition hybrid score: 0.3*CI + 0.7*(1-WBrier), higher is better."""
    wb = weighted_brier_score(y_time, y_event, prob_dict)
    ci = c_index(y_time, y_event, prob_dict[12])
    score = 0.3 * ci + 0.7 * (1.0 - wb)
    details = {"hybrid": score, "c_index": ci, "weighted_brier": wb}
    for h, w in BRIER_WEIGHTS.items():
        details[f"brier_{h}h"] = horizon_brier_score(y_time, y_event, prob_dict[h], h)
    return score, details
