"""Main training script: CV + ensemble + calibration + monotonicity + submission."""

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import (
    TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH, SUBMISSION_PATH,
    ID_COL, TIME_COL, EVENT_COL, HORIZONS, PROB_COLS,
    N_SPLITS, N_REPEATS, RANDOM_STATE,
    FEATURES_MEDIUM, FEATURES_FULL,
)
from src.features import remove_redundant, add_engineered, get_feature_set
from src.evaluation import (
    horizon_brier_score, mean_brier_score, c_index, combined_score,
    hybrid_score, weighted_brier_score,
)
from src.models import (
    CoxPH, RSF, GBSA, MultiHorizonXGB, RankXGB,
    WeibullAFT, LogNormalAFT, _build_horizon_labels,
)
from src.calibration import platt_scaling, calibrate
from src.ensemble import (
    ensemble_predict, optimize_weights,
    robust_optimize_weights,
    optimize_weights_per_horizon, ensemble_predict_per_horizon,
    stacking_meta_learner, stacking_predict_12h,
)
from src.monotonic import enforce_monotonicity, submission_postprocess

warnings.filterwarnings("ignore")

MODEL_NAMES = ["coxph", "rsf", "gbsa", "xgb", "rankxgb", "weibull", "lognormal"]


def load_data():
    """Load and prepare train/test data."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train = add_engineered(remove_redundant(train))
    test = add_engineered(remove_redundant(test))

    return train, test


def make_models(feature_cols):
    """Create fresh model instances."""
    return {
        "coxph": CoxPH(penalizer=0.1, features=FEATURES_MEDIUM),
        "rsf": RSF(),
        "gbsa": GBSA(),
        "xgb": MultiHorizonXGB(random_state=RANDOM_STATE),
        "rankxgb": RankXGB(random_state=RANDOM_STATE),
        "weibull": WeibullAFT(penalizer=0.05, features=FEATURES_MEDIUM),
        "lognormal": LogNormalAFT(penalizer=0.05, features=FEATURES_MEDIUM),
    }


def _strat_labels(y_time, y_event):
    """Build 4-class stratification labels for balanced CV folds.

    Classes: 0=hit<=12h, 1=hit 12-24h, 2=hit>24h, 3=censored.
    """
    labels = np.full(len(y_time), 3, dtype=int)
    hit = y_event == 1
    labels[hit & (y_time <= 12)] = 0
    labels[hit & (y_time > 12) & (y_time <= 24)] = 1
    labels[hit & (y_time > 24)] = 2
    return labels


def run_cv(train, feature_cols):
    """Run repeated stratified K-fold CV, collect OOF predictions."""
    X = train[feature_cols]
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    n = len(train)
    # OOF predictions: {model_name: {horizon: array}}
    oof_preds = {
        name: {h: np.zeros(n) for h in HORIZONS}
        for name in MODEL_NAMES
    }
    oof_counts = np.zeros(n)

    strat = _strat_labels(y_time, y_event)
    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    fold_idx = 0
    for tr_idx, va_idx in rskf.split(X, strat):
        fold_idx += 1
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        yt_tr, yt_va = y_time[tr_idx], y_time[va_idx]
        ye_tr, ye_va = y_event[tr_idx], y_event[va_idx]

        models = make_models(feature_cols)
        for name, model in models.items():
            try:
                model.fit(X_tr, yt_tr, ye_tr)
                preds = model.predict_proba(X_va)
                for h in HORIZONS:
                    oof_preds[name][h][va_idx] += preds[h]
            except Exception as e:
                print(f"  [WARN] {name} fold {fold_idx} failed: {e}")
                for h in HORIZONS:
                    oof_preds[name][h][va_idx] += 0.5

        oof_counts[va_idx] += 1

        if fold_idx % N_SPLITS == 0:
            rep = fold_idx // N_SPLITS
            print(f"  Repeat {rep}/{N_REPEATS} done")

    # Average OOF predictions across repeats
    for name in MODEL_NAMES:
        for h in HORIZONS:
            mask = oof_counts > 0
            oof_preds[name][h][mask] /= oof_counts[mask]

    return oof_preds


def print_oof_scores(oof_preds, y_time, y_event):
    """Print per-model OOF scores using competition hybrid metric."""
    print("\n=== OOF Scores (per model) ===")
    for name in MODEL_NAMES:
        score, details = hybrid_score(y_time, y_event, oof_preds[name])
        print(f"  {name:10s}  Hybrid={score:.4f}  CI={details['c_index']:.4f}  WBrier={details['weighted_brier']:.4f}")


def main():
    print("=== Loading data ===")
    train, test = load_data()
    feature_cols = get_feature_set(train, level="medium")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    # --- Step 1: CV ---
    print("\n=== Running CV ===")
    oof_preds = run_cv(train, feature_cols)
    print_oof_scores(oof_preds, y_time, y_event)

    # --- Step 2: Bootstrap-robust ensemble weights + per-horizon ---
    print("\n=== Optimizing ensemble weights (bootstrap-robust) ===")
    oof_list = [oof_preds[name] for name in MODEL_NAMES]
    weights = robust_optimize_weights(oof_list, y_time, y_event, n_bootstrap=30)
    print("  Global weights (bootstrap median):")
    for name, w in zip(MODEL_NAMES, weights):
        print(f"    {name:10s}: {w:.4f}")

    print("\n=== Per-horizon weight optimization ===")
    weights_dict = optimize_weights_per_horizon(oof_list, y_time, y_event)
    for h in HORIZONS:
        w_str = "  ".join(f"{name}={w:.3f}" for name, w in zip(MODEL_NAMES, weights_dict[h]))
        print(f"  {h}h: {w_str}")

    ens_oof = ensemble_predict_per_horizon(oof_list, weights_dict)
    ens_oof = enforce_monotonicity(ens_oof)
    oof_score, oof_details = hybrid_score(y_time, y_event, ens_oof)
    print(f"\n  Ensemble OOF (per-horizon, raw):")
    print(f"    Hybrid={oof_score:.4f}  CI={oof_details['c_index']:.4f}  WBrier={oof_details['weighted_brier']:.4f}")

    # Evaluate with competition postprocessing
    ens_oof_pp = submission_postprocess(ens_oof)
    pp_score, pp_details = hybrid_score(y_time, y_event, ens_oof_pp)
    print(f"  Ensemble OOF (postprocessed):")
    print(f"    Hybrid={pp_score:.4f}  CI={pp_details['c_index']:.4f}  WBrier={pp_details['weighted_brier']:.4f}")
    for h in [24, 48, 72]:
        print(f"    Brier@{h}h={pp_details[f'brier_{h}h']:.4f}")

    # --- Step 2b: Train stacking meta-learner on OOF 12h ---
    print("\n=== Training stacking meta-learner ===")
    meta = stacking_meta_learner(oof_list, y_time, y_event, MODEL_NAMES)

    # Evaluate stacking on OOF (eligible samples only)
    from src.models import _build_horizon_labels as _bhl
    _, eligible = _bhl(y_time, y_event, 12)
    X_meta_oof = np.column_stack([oof_preds[name][12][eligible] for name in MODEL_NAMES])
    stacked_12h_oof = np.zeros(len(y_time))
    stacked_12h_oof[eligible] = meta.predict_proba(X_meta_oof)[:, 1]
    stacked_12h_oof[~eligible] = ens_oof[12][~eligible]
    ci_stacked = c_index(y_time, y_event, stacked_12h_oof)
    print(f"  Stacking OOF C-index(12h)={ci_stacked:.4f}  (vs weighted avg {oof_details['c_index']:.4f})")

    # --- Step 2c: Platt Scaling diagnostic (disabled -- worsens Brier) ---
    print("\n=== Platt Scaling diagnostic (not applied) ===")
    for h in [24, 48]:
        labels_h, elig_h = _bhl(y_time, y_event, h)
        cal = platt_scaling(ens_oof[h][elig_h], labels_h[elig_h])
        cal_probs = calibrate(cal, ens_oof[h][elig_h])
        brier_before = horizon_brier_score(y_time, y_event, ens_oof[h], h)
        brier_after = horizon_brier_score(
            y_time[elig_h], y_event[elig_h], cal_probs, h,
        )
        print(f"  {h}h: Brier before={brier_before:.4f}  after={brier_after:.4f}  (skipped)")

    # --- Step 3: Retrain on full data ---
    print("\n=== Retraining on full data ===")
    X_train = train[feature_cols]
    X_test = test[feature_cols]
    final_models = make_models(feature_cols)

    test_preds_list = []
    for name in MODEL_NAMES:
        model = final_models[name]
        print(f"  Training {name}...")
        try:
            model.fit(X_train, y_time, y_event)
            preds = model.predict_proba(X_test)
            test_preds_list.append(preds)
            print(f"    {name} done")
        except Exception as e:
            print(f"    [WARN] {name} failed: {e}")
            fallback = {h: np.full(len(test), 0.5) for h in HORIZONS}
            test_preds_list.append(fallback)

    # --- Step 4: Ensemble + stacking + postprocess ---
    print("\n=== Generating submission ===")
    test_ens = ensemble_predict_per_horizon(test_preds_list, weights_dict)

    # Replace 12h with stacking prediction
    stacked_12h = stacking_predict_12h(meta, test_preds_list)
    print(f"  Stacking 12h range: [{stacked_12h.min():.4f}, {stacked_12h.max():.4f}]")
    test_ens[12] = stacked_12h

    test_ens = submission_postprocess(test_ens)

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = test_ens[h]

    # --- Validation checks ---
    # 72h all-ones check
    h72_col = PROB_COLS[HORIZONS.index(72)]
    h72_ok = (sub[h72_col] == 1.0).all()

    # Clip range check for 12h/24h/48h
    clip_cols = [PROB_COLS[HORIZONS.index(h)] for h in [12, 24, 48]]
    clip_ok = all(
        (sub[c] >= 0.01 - 1e-9).all() and (sub[c] <= 0.99 + 1e-9).all()
        for c in clip_cols
    )

    # Monotonicity check (24h <= 48h <= 72h)
    mono_ok = True
    for i in range(len(sub)):
        vals = [sub[col].iloc[i] for col in PROB_COLS]
        for j in range(1, len(vals)):
            if vals[j] < vals[j - 1] - 1e-9:
                mono_ok = False
                break

    print(f"  72h all-ones check: {'PASS' if h72_ok else 'FAIL'}")
    print(f"  Clip [0.01,0.99]:   {'PASS' if clip_ok else 'FAIL'}")
    print(f"  Monotonicity check: {'PASS' if mono_ok else 'FAIL'}")
    print(f"  Shape: {sub.shape}")

    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n  Submission saved to {SUBMISSION_PATH}")
    print(f"  Preview:\n{sub.head()}")


if __name__ == "__main__":
    main()
