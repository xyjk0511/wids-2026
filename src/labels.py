"""Public horizon label builder for binary classification."""

import numpy as np


def build_horizon_labels(y_time, y_event, horizon):
    """Build binary labels for a single horizon.

    Positive: event=1 and time <= horizon
    Negative: event=1 and time > horizon; event=0 and time >= horizon
    Excluded: event=0 and time < horizon (censored before horizon)

    Returns: (labels, eligible_mask)
    """
    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event, dtype=int)
    n = len(y_time)
    labels = np.full(n, np.nan)

    labels[(y_event == 1) & (y_time <= horizon)] = 1.0
    labels[(y_event == 1) & (y_time > horizon)] = 0.0
    labels[(y_event == 0) & (y_time >= horizon)] = 0.0

    eligible = ~np.isnan(labels)
    return labels, eligible
