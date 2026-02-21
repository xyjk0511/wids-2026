"""Exp24: Extended base (CoxPH+WeibullAFT) + RankXGB 12h feature + eps sweep + logit blend.

Usage:
    .venv_sksurv22/Scripts/python.exe -m scripts.exp24_extended_base
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    TRAIN_PATH, TEST_PATH, FEATURES_MEDIUM, TIME_COL, EVENT_COL,
    HORIZONS, SUBMISSION_PATH,
)
from src.features import add_engineered
from src.evaluation import hybrid_score, c_index, horizon_brier_score
from src.monotonic import submission_postprocess
from src.stacking import (
    EXTENDED_BASE_NAMES, _train_predict_base, train_horizon_heads,
    predict_horizon_heads, HEAD_HORIZONS,
)


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train = add_engineered(train)
    test = add_engineered(test)
    feature_cols = [c for c in FEATURES_MEDIUM if c in train.columns]
    return train, test, feature_cols


def run_cv_extended(train, feature_cols, y_time, y_event):
    """Run CV with extended base models."""
    print("=== Phase 1: CV with extended base (RSF+EST+XGBCox+CoxPH+WeibullAFT) ===")
    head_oof, heads = train_horizon_heads(
        X_features=train[feature_cols],
        base_oof=None,  # will be generated internally
        y_time=y_time,
        y_event=y_event,
        n_splits=5,
        n_repeats=3,
        n_inner_splits=3,
        random_state=1042,
        head_model="xgb",
        base_feature_mode="raw",
        use_orig_features=True,
        calibration_mode="auto",
        base_names=EXTENDED_BASE_NAMES,
    )
    return head_oof, heads


def generate_test_preds(train, test, feature_cols, y_time, y_event, heads):
    """Full retrain with extended base and predict test."""
    SEEDS = [42, 123, 456, 789, 2026]
    X_train = train[feature_cols]
    X_test = test[feature_cols]

    print(f"\n=== Full retrain ({len(SEEDS)} seeds, extended base) ===")
    base_test_all = []
    for seed in SEEDS:
        bp = _train_predict_base(X_train, y_time, y_event, X_test,
                                 seed=seed, base_names=EXTENDED_BASE_NAMES)
        base_test_all.append(bp)
        print(f"  Seed {seed} done")

    base_names = list(base_test_all[0].keys())
    base_test_preds = {}
    for name in base_names:
        base_test_preds[name] = {
            h: np.mean([bp[name][h] for bp in base_test_all], axis=0)
            for h in HEAD_HORIZONS
        }

    test_preds = predict_horizon_heads(heads, X_test, base_test_preds)
    test_preds[72] = np.ones(len(test))
    return test_preds


def eps_sweep(test_preds, tag="exp24"):
    """Sweep p12<=p24 eps values and save submissions."""
    eps_values = [None, 1e-7, 5e-7]
    for eps in eps_values:
        pp = submission_postprocess(test_preds, cap_12_by_24_eps=eps)
        eps_str = "none" if eps is None else f"{eps:.0e}"
        fname = SUBMISSION_PATH / f"submission_{tag}_eps{eps_str}.csv"
        sub = pd.read_csv(TEST_PATH)[["id"]]
        for h in HORIZONS:
            col = f"prob_{h}h"
            sub[col] = pp[h] if h != 72 else np.ones(len(sub))
        sub.to_csv(fname, index=False)
        print(f"  eps={eps_str}: saved {fname.name}")


def logit_blend(test_preds, anchor_path, tag="exp24"):
    """Low-DOF logit blend: alpha * new + (1-alpha) * anchor."""
    anchor = pd.read_csv(anchor_path)
    alphas = [0.1, 0.2, 0.3, 0.5]

    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    for alpha in alphas:
        sub = pd.read_csv(TEST_PATH)[["id"]]
        for h in [12, 24, 48]:
            col = f"prob_{h}h"
            p_new = np.clip(test_preds[h], 1e-6, 1 - 1e-6)
            p_anc = anchor[col].values
            blended = _sigmoid(alpha * _logit(p_new) + (1 - alpha) * _logit(p_anc))
            sub[col] = blended
        sub["prob_72h"] = 1.0
        fname = SUBMISSION_PATH / f"submission_{tag}_blend_a{alpha:.1f}.csv"
        sub.to_csv(fname, index=False)
        print(f"  alpha={alpha:.1f}: saved {fname.name}")


def main():
    train, test, feature_cols = load_data()
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    # Phase 1: CV with extended base
    head_oof, heads = run_cv_extended(train, feature_cols, y_time, y_event)

    # Evaluate OOF
    prob_dict = {h: head_oof[h] for h in [12, 24, 48]}
    prob_dict[72] = np.ones(len(train))
    score, details = hybrid_score(y_time, y_event, prob_dict)
    print(f"\n  Extended base OOF Hybrid: {score:.4f}")
    print(f"    CI={details['ci']:.4f} WBrier={details['wbrier']:.4f}")

    # Phase 2: Generate test predictions
    test_preds = generate_test_preds(train, test, feature_cols, y_time, y_event, heads)

    # Phase 3: eps sweep
    print("\n=== Eps sweep ===")
    eps_sweep(test_preds)

    # Phase 4: logit blend with anchor
    anchor_path = SUBMISSION_PATH / "submission_0.96624.csv"
    if anchor_path.exists():
        print("\n=== Logit blend with anchor 0.96624 ===")
        logit_blend(test_preds, anchor_path)
    else:
        print(f"\n  Anchor not found: {anchor_path}")

    # Also save raw submission
    pp = submission_postprocess(test_preds)
    sub = pd.read_csv(TEST_PATH)[["id"]]
    for h in HORIZONS:
        sub[f"prob_{h}h"] = pp[h] if h != 72 else np.ones(len(sub))
    raw_path = SUBMISSION_PATH / "submission_exp24_raw.csv"
    sub.to_csv(raw_path, index=False)
    print(f"\n  Raw submission: {raw_path.name}")


if __name__ == "__main__":
    main()
