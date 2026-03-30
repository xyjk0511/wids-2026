"""Reusable CV protocol utilities for stability-focused survival experiments."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from typing import Callable
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    EVENT_COL,
    HORIZONS,
    N_REPEATS,
    N_SPLITS,
    RANDOM_STATE,
    TIME_COL,
    TRAIN_PATH,
)
from src.evaluation import hybrid_score  # noqa: E402
from src.features import add_engineered, get_feature_set, remove_redundant  # noqa: E402
from src.monotonic import enforce_monotonicity, submission_postprocess  # noqa: E402

# dist_zone_score uses exp(); large magnitudes may emit benign overflow warnings.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")


ProbDict = dict[int, np.ndarray]
ModelBuilder = Callable[[int], object]


def load_prepared_train(feature_level: str = "medium") -> tuple[pd.DataFrame, list[str]]:
    """Load train.csv and apply the same feature preprocessing as production code."""
    train = pd.read_csv(TRAIN_PATH)
    if feature_level in ("v96624", "v96624_plus"):
        train = add_engineered(train)
    else:
        train = add_engineered(remove_redundant(train))
    feature_cols = get_feature_set(train, level=feature_level)
    return train, feature_cols


def _merge_rare_labels(labels: np.ndarray, min_count: int) -> np.ndarray:
    """Merge rare composite classes so RepeatedStratifiedKFold stays valid.

    Label encoding:
      - censored classes: 0..3 (time bins)
      - event classes:    10..13 (time bins)
    """
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        counts = Counter(labels.tolist())
        for lab, cnt in sorted(counts.items(), key=lambda x: x[1]):
            if cnt == 0 or cnt >= min_count:
                continue

            is_event = lab >= 10
            base = 10 if is_event else 0
            b = lab - base

            candidates = []
            for bb in range(4):
                cand = base + bb
                c_cnt = counts.get(cand, 0)
                if cand == lab or c_cnt == 0:
                    continue
                candidates.append((cand, abs(bb - b), -c_cnt))
            if not candidates:
                continue

            # nearest same-type bin first, then the larger class.
            target = sorted(candidates, key=lambda x: (x[1], x[2]))[0][0]
            labels[labels == lab] = target
            changed = True
    return labels


def build_strat_labels(
    y_time: np.ndarray,
    y_event: np.ndarray,
    mode: str = "event_time",
    n_splits: int = N_SPLITS,
) -> np.ndarray:
    """Build stratification labels for CV.

    Modes:
      - event: original 2-class split (event vs censored)
      - event_time: event/censor + horizon-aware time bins
    """
    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event, dtype=int)

    if mode == "event":
        return y_event.astype(int)

    if mode != "event_time":
        raise ValueError(f"Unknown strat mode: {mode}")

    # bins: <=12, (12,24], (24,48], >48
    tbin = np.digitize(y_time, bins=[12.0, 24.0, 48.0], right=True)
    labels = np.where(y_event == 1, 10 + tbin, tbin).astype(int)
    labels = _merge_rare_labels(labels, min_count=n_splits)
    return labels


def describe_strat_labels(labels: np.ndarray) -> str:
    """Compact label-count summary string for logs."""
    c = Counter(labels.tolist())
    parts = [f"{k}:{v}" for k, v in sorted(c.items(), key=lambda x: x[0])]
    return " ".join(parts)


def run_oof_cv(
    train: pd.DataFrame,
    feature_cols: list[str],
    model_builders: dict[str, ModelBuilder],
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    random_state: int = RANDOM_STATE,
    strat_mode: str = "event_time",
    verbose: bool = True,
) -> tuple[dict[str, ProbDict], np.ndarray, np.ndarray]:
    """Run repeated stratified CV and return averaged OOF predictions per model."""
    X = train[feature_cols]
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values
    n = len(train)

    oof = {name: {h: np.zeros(n) for h in HORIZONS} for name in model_builders}
    oof_counts = np.zeros(n, dtype=float)

    strat_labels = build_strat_labels(y_time, y_event, mode=strat_mode, n_splits=n_splits)
    if verbose:
        print(f"  Strat mode={strat_mode} labels: {describe_strat_labels(strat_labels)}")

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    fold_idx = 0
    for tr_idx, va_idx in rskf.split(X, strat_labels):
        fold_idx += 1
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        yt_tr = y_time[tr_idx]
        ye_tr = y_event[tr_idx]

        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(
            scaler.fit_transform(X_tr),
            columns=feature_cols,
            index=X_tr.index,
        )
        X_va_s = pd.DataFrame(
            scaler.transform(X_va),
            columns=feature_cols,
            index=X_va.index,
        )

        fold_seed = random_state + fold_idx
        for name, builder in model_builders.items():
            model = builder(fold_seed)
            model.fit(X_tr_s, yt_tr, ye_tr)
            preds = model.predict_proba(X_va_s)
            for h in HORIZONS:
                oof[name][h][va_idx] += preds[h]

        oof_counts[va_idx] += 1.0

        if verbose and fold_idx % n_splits == 0:
            rep = fold_idx // n_splits
            print(f"    Repeat {rep}/{n_repeats} done")

    mask = oof_counts > 0
    for name in model_builders:
        for h in HORIZONS:
            oof[name][h][mask] /= oof_counts[mask]

    return oof, y_time, y_event


def blend_two_models(prob_a: ProbDict, prob_b: ProbDict, w_a: float) -> ProbDict:
    """Weighted blend of two model probability dictionaries."""
    return {h: w_a * prob_a[h] + (1.0 - w_a) * prob_b[h] for h in HORIZONS}


def search_global_weight(
    prob_a: ProbDict,
    prob_b: ProbDict,
    y_time: np.ndarray,
    y_event: np.ndarray,
    low: float = 0.50,
    high: float = 1.00,
    step: float = 0.05,
) -> tuple[float, float]:
    """Search best global blend weight by raw Hybrid score."""
    best_w, best_score = low, -np.inf
    n_steps = int(round((high - low) / step))
    for i in range(n_steps + 1):
        w = low + i * step
        blended = blend_two_models(prob_a, prob_b, w)
        score, _ = hybrid_score(y_time, y_event, blended)
        if score > best_score:
            best_w, best_score = w, score
    return best_w, best_score


def score_probs(
    prob_dict: ProbDict,
    y_time: np.ndarray,
    y_event: np.ndarray,
    apply_postprocess: bool = False,
) -> tuple[float, dict]:
    """Score predictions with optional submission-style postprocessing."""
    scored = prob_dict
    if apply_postprocess:
        scored = submission_postprocess(scored)
        scored = enforce_monotonicity(scored)
    return hybrid_score(y_time, y_event, scored)
