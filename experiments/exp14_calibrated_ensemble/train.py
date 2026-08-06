"""Exp14: Calibration-first ensemble pipeline.

Separates ranking (RSF+EST) from calibration (Weibull floor + KM power),
addressing the ~60% zero-probability problem in full retrain.

Usage: python -m experiments.exp14_calibrated_ensemble.train
"""

import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import (
    TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH, SUBMISSION_PATH,
    ID_COL, TIME_COL, EVENT_COL, HORIZONS, PROB_COLS,
    N_SPLITS, N_REPEATS, RANDOM_STATE,
)
from src.features import remove_redundant, add_engineered, get_feature_set
from src.evaluation import hybrid_score
from src.models import RSF, EST, WeibullAFT
from src.monotonic import submission_postprocess
from .calibration import apply_piecewise_power, MILD_PARAMS, compute_match_ref_params

warnings.filterwarnings("ignore")

FEATURE_LEVEL = "medium"
RETRAIN_SEEDS = [42, 123, 456, 789, 2026]


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train = add_engineered(remove_redundant(train))
    test = add_engineered(remove_redundant(test))
    return train, test


def _strat_labels(y_time, y_event, n_splits=N_SPLITS):
    """Event-time composite stratification (proven +0.0012)."""
    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event, dtype=int)
    tbin = np.digitize(y_time, bins=[12.0, 24.0, 48.0], right=True)
    labels = np.where(y_event == 1, 10 + tbin, tbin).astype(int)
    # Merge rare labels
    changed = True
    while changed:
        changed = False
        counts = Counter(labels.tolist())
        for lab, cnt in sorted(counts.items(), key=lambda x: x[1]):
            if cnt == 0 or cnt >= n_splits:
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
            target = sorted(candidates, key=lambda x: (x[1], x[2]))[0][0]
            labels[labels == lab] = target
            changed = True
    return labels


def run_oof_cv(train, feature_cols, test=None):
    """5-fold x 10-repeat CV collecting RSF, EST, and WeibullAFT OOF predictions.

    If test is provided, also collects averaged test predictions across all folds.
    Returns (oof, test_preds) where test_preds is None if test was not given.
    """
    X = train[feature_cols]
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values
    n = len(train)

    model_names = ["RSF", "EST", "Weibull"]
    oof = {name: {h: np.zeros(n) for h in HORIZONS} for name in model_names}
    oof_counts = np.zeros(n)

    # Test prediction accumulators
    has_test = test is not None
    if has_test:
        X_test = test[feature_cols]
        test_acc = {name: {h: np.zeros(len(test)) for h in HORIZONS}
                    for name in model_names}
        n_folds_total = 0

    strat = _strat_labels(y_time, y_event)
    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE,
    )

    fold_idx = 0
    for tr_idx, va_idx in rskf.split(X, strat):
        fold_idx += 1
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        yt_tr, ye_tr = y_time[tr_idx], y_event[tr_idx]

        # StandardScaler for RSF/EST
        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(
            scaler.fit_transform(X_tr), columns=feature_cols, index=X_tr.index,
        )
        X_va_s = pd.DataFrame(
            scaler.transform(X_va), columns=feature_cols, index=X_va.index,
        )

        # RSF
        rsf = RSF(min_samples_leaf=5)
        rsf.fit(X_tr_s, yt_tr, ye_tr)
        rsf_p = rsf.predict_proba(X_va_s)

        # EST
        est = EST(min_samples_leaf=5)
        est.fit(X_tr_s, yt_tr, ye_tr)
        est_p = est.predict_proba(X_va_s)

        # WeibullAFT (uses its own internal scaling, pass unscaled data)
        waft = WeibullAFT()
        waft.fit(train.iloc[tr_idx], yt_tr, ye_tr)
        waft_p = waft.predict_proba(train.iloc[va_idx])

        for h in HORIZONS:
            oof["RSF"][h][va_idx] += rsf_p[h]
            oof["EST"][h][va_idx] += est_p[h]
            oof["Weibull"][h][va_idx] += waft_p[h]
        oof_counts[va_idx] += 1

        # Collect test predictions from each fold
        if has_test:
            X_test_s = pd.DataFrame(
                scaler.transform(X_test), columns=feature_cols, index=X_test.index,
            )
            rsf_test = rsf.predict_proba(X_test_s)
            est_test = est.predict_proba(X_test_s)
            waft_test = waft.predict_proba(test)
            for h in HORIZONS:
                test_acc["RSF"][h] += rsf_test[h]
                test_acc["EST"][h] += est_test[h]
                test_acc["Weibull"][h] += waft_test[h]
            n_folds_total += 1

        if fold_idx % N_SPLITS == 0:
            rep = fold_idx // N_SPLITS
            print(f"  Repeat {rep}/{N_REPEATS} done")

    # Average over repeats
    mask = oof_counts > 0
    for name in model_names:
        for h in HORIZONS:
            oof[name][h][mask] /= oof_counts[mask]

    # Average test predictions over all folds
    if has_test:
        for name in model_names:
            for h in HORIZONS:
                test_acc[name][h] /= n_folds_total

    return oof, (test_acc if has_test else None)



def _print_dist(prefix, prob_dict):
    """Print probability distribution summary per horizon."""
    for h in HORIZONS:
        p = prob_dict[h]
        print(f"{prefix}h={h}: min={p.min():.4f} p25={np.percentile(p,25):.4f} "
              f"median={np.median(p):.4f} p75={np.percentile(p,75):.4f} "
              f"max={p.max():.4f} mean={p.mean():.4f}")


def generate_submission(test_preds, path=None):
    """Postprocess, validate, save, and print diagnostics."""
    sub_path = path or SUBMISSION_PATH
    test_preds = submission_postprocess(test_preds)

    # R1: clamp 12h down to 24h (never let aggressive 12h pollute 24h Brier)
    test_preds[12] = np.minimum(test_preds[12], test_preds[24])
    # Forward monotonicity 24->48->72 only
    test_preds[48] = np.maximum(test_preds[48], test_preds[24])
    test_preds[72] = np.ones(len(test_preds[72]))

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = test_preds[h]

    # Validation
    h72_ok = (sub[PROB_COLS[HORIZONS.index(72)]] == 1.0).all()
    mono_ok = all(
        all(sub[PROB_COLS[j]].iloc[i] >= sub[PROB_COLS[j - 1]].iloc[i] - 1e-9
            for j in range(1, len(PROB_COLS)))
        for i in range(len(sub))
    )
    print(f"  72h all-ones: {'PASS' if h72_ok else 'FAIL'}  "
          f"Monotonicity: {'PASS' if mono_ok else 'FAIL'}  Shape: {sub.shape}")

    # Distribution
    print("\n=== Probability Distribution ===")
    targets = {12: (0.036, 0.15, 0.99), 24: (0.01, 0.71, 0.99),
               48: (0.01, 0.83, 0.99)}
    for h, col in zip(HORIZONS, PROB_COLS):
        vals = sub[col]
        line = f"  {col}: min={vals.min():.4f} median={vals.median():.4f} max={vals.max():.4f}"
        if h in targets:
            t = targets[h]
            line += f"  (target: min~{t[0]} med~{t[1]} max~{t[2]})"
        print(line)

    sub.to_csv(sub_path, index=False)
    print(f"\n  Saved to {sub_path}")

    # Spearman vs 0.96624 reference
    ref_path = "submission 0.96624.csv"
    try:
        ref = pd.read_csv(ref_path)
        from scipy.stats import spearmanr
        print("\n=== Spearman vs 0.96624 ===")
        for col in PROB_COLS[:-1]:
            sr, _ = spearmanr(sub[col], ref[col])
            print(f"  {col}: rho={sr:.4f}")
    except FileNotFoundError:
        print(f"\n  [WARN] '{ref_path}' not found, skipping Spearman")

    return sub


def main():
    print("=" * 60)
    print("Exp14 v3: CV Ensemble + Piecewise Power Calibration")
    print("=" * 60)

    # --- Phase 0: Load data ---
    print("\n=== Loading data ===")
    train, test = load_data()
    feature_cols = get_feature_set(train, level=FEATURE_LEVEL)
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    # --- Phase 1: OOF CV + test prediction collection ---
    print("\n=== Phase 1: OOF CV (RSF + EST + WeibullAFT) + test preds ===")
    oof, cv_test = run_oof_cv(train, feature_cols, test=test)

    # Print per-model OOF scores
    print("\n=== OOF Scores (per-model) ===")
    for name in oof:
        score, det = hybrid_score(y_time, y_event, oof[name])
        print(f"  {name:10s}  Hybrid={score:.4f}  CI={det['c_index']:.4f}  "
              f"WBrier={det['weighted_brier']:.4f}")

    # --- Phase 2: OOF baseline evaluation ---
    blend_oof = {h: 0.5 * oof["RSF"][h] + 0.5 * oof["EST"][h] for h in HORIZONS}
    score, det = hybrid_score(y_time, y_event, blend_oof)
    print(f"\n  OOF Hybrid={score:.4f}  CI={det['c_index']:.4f}  "
          f"WBrier={det['weighted_brier']:.4f}")

    # --- Phase 3: CV ensemble test predictions ---
    print("\n=== Phase 3: CV Ensemble Test Predictions ===")
    blend_test = {h: 0.5 * cv_test["RSF"][h] + 0.5 * cv_test["EST"][h]
                  for h in HORIZONS}
    _print_dist("  CV ensemble test ", blend_test)

    # --- Phase 4: Generate submission variants ---
    # B: Pure CV ensemble (main submission)
    print("\n=== Variant B: Pure CV Ensemble ===")
    generate_submission(blend_test, path=SUBMISSION_PATH)

    # C-mild: CV + mild piecewise power
    print("\n=== Variant C-mild: CV + Mild Piecewise Power ===")
    mild = apply_piecewise_power(blend_test, MILD_PARAMS)
    _print_dist("  C-mild ", mild)
    generate_submission(mild, path="submission_cv_mild.csv")

    # C-match: CV + reference-anchored piecewise power
    print("\n=== Variant C-match: CV + Reference-Anchored Power ===")
    match_params = compute_match_ref_params(blend_test)
    matched = apply_piecewise_power(blend_test, match_params)
    _print_dist("  C-match ", matched)
    generate_submission(matched, path="submission_cv_match.csv")

    print("\n" + "=" * 60)
    print("Exp14 v3 complete. 3 submissions generated:")
    print(f"  B:       {SUBMISSION_PATH}")
    print(f"  C-mild:  submission_cv_mild.csv")
    print(f"  C-match: submission_cv_match.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
