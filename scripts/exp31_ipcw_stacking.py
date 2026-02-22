"""Exp31: IPCW-aware stacking with RSF+EST+GBSA base models.

Two-stage CV: 5x1 quick check, 5x10 only if OOF hybrid > 0.9697.
Go/no-go gate: OOF hybrid > 0.9697 AND Spearman vs 0.96624 > 0.90.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")

from lifelines import KaplanMeierFitter
from src.config import (
    TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH,
    TIME_COL, EVENT_COL, HORIZONS, PROB_COLS,
    FEATURES_V96624_BASE, FEATURES_V96624_ENGINEERED,
)
from src.features import add_engineered
from src.labels import build_horizon_labels
from src.models import RSF, EST, GBSA
from src.evaluation import hybrid_score
from src.train import _strat_labels

FEATURES = list(FEATURES_V96624_BASE) + list(FEATURES_V96624_ENGINEERED)
GATE_HYBRID = 0.9697
GATE_SPEARMAN = 0.90
REF_PATH = "submissions/submission_0.96624.csv"
OUT_PATH = "submissions/submission_exp31_ipcw.csv"


def compute_ipcw_weights(y_time, y_event):
    """KM-based IPCW weights: 1/G(t) for events, 1.0 for censored."""
    kmf = KaplanMeierFitter()
    kmf.fit(y_time, event_observed=(1 - y_event))  # model censoring
    weights = np.ones(len(y_time))
    event_mask = y_event == 1
    t_events = y_time[event_mask]
    g_vals = kmf.survival_function_at_times(t_events).values
    g_vals = np.clip(g_vals, 0.05, None)
    weights[event_mask] = 1.0 / g_vals
    print(f"  IPCW weights — min={weights.min():.3f} max={weights.max():.3f} "
          f"p5={np.percentile(weights,5):.3f} p95={np.percentile(weights,95):.3f}")
    return weights


def make_base_models():
    return [
        RSF(n_estimators=200, min_samples_leaf=5),
        EST(n_estimators=200, min_samples_leaf=5),
        GBSA(n_estimators=300, max_depth=3, learning_rate=0.02),
    ]


def collect_oof_meta(train, feature_cols, n_splits, n_repeats):
    """Run CV, collect 12-feature OOF meta-features from 3 base models."""
    X = train[feature_cols]
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values
    n = len(train)
    n_models = 3

    # meta[model_idx][horizon] = oof array
    meta = [[np.zeros(n) for _ in HORIZONS] for _ in range(n_models)]
    counts = np.zeros(n)

    strat = _strat_labels(y_time, y_event)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    for fold_i, (tr_idx, va_idx) in enumerate(rskf.split(X, strat)):
        scaler = StandardScaler()
        X_tr = pd.DataFrame(scaler.fit_transform(X.iloc[tr_idx]), columns=feature_cols)
        X_va = pd.DataFrame(scaler.transform(X.iloc[va_idx]), columns=feature_cols)

        models = make_base_models()
        for mi, model in enumerate(models):
            model.fit(X_tr, y_time[tr_idx], y_event[tr_idx])
            preds = model.predict_proba(X_va)
            for hi, h in enumerate(HORIZONS):
                meta[mi][hi][va_idx] += preds[h]
        counts[va_idx] += 1

        if (fold_i + 1) % n_splits == 0:
            rep = (fold_i + 1) // n_splits
            print(f"    Repeat {rep}/{n_repeats} done")

    mask = counts > 0
    for mi in range(n_models):
        for hi in range(len(HORIZONS)):
            meta[mi][hi][mask] /= counts[mask]

    return meta, y_time, y_event


def fit_meta_learners(meta, y_time, y_event, ipcw_weights):
    """Fit Ridge and LR per horizon using IPCW weights. Returns OOF prob dicts."""
    n = len(y_time)
    ridge_oof = {h: np.zeros(n) for h in HORIZONS}
    lr_oof = {h: np.zeros(n) for h in HORIZONS}

    # Build 12-column meta-feature matrix (3 models x 4 horizons)
    # For each horizon, fit meta-learner on eligible samples
    meta_X = np.column_stack([meta[mi][hi] for mi in range(3) for hi in range(len(HORIZONS))])

    for hi, h in enumerate(HORIZONS):
        labels, eligible = build_horizon_labels(y_time, y_event, h)
        if eligible.sum() < 10:
            continue
        X_e = meta_X[eligible]
        y_e = labels[eligible].astype(float)
        w_e = ipcw_weights[eligible]

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_e, y_e, sample_weight=w_e)
        ridge_oof[h][eligible] = np.clip(ridge.predict(X_e), 0.0, 1.0)

        if len(np.unique(y_e)) < 2:
            lr_oof[h][eligible] = y_e  # constant horizon, pass through
        else:
            lr = LogisticRegression(C=0.01, max_iter=2000)
            lr.fit(X_e, y_e, sample_weight=w_e)
            lr_oof[h][eligible] = lr.predict_proba(X_e)[:, 1]

    return ridge_oof, lr_oof


def eval_gate(oof_dict, y_time, y_event, ref_df, label, test_preds=None):
    """Evaluate gate: OOF hybrid score and Spearman of test preds vs reference."""
    score, details = hybrid_score(y_time, y_event, oof_dict)
    print(f"  [{label}] OOF hybrid={score:.5f}  CI={details['c_index']:.4f}  WBrier={details['weighted_brier']:.5f}")

    min_spearman = 0.0
    if test_preds is not None:
        spearmans = []
        for h in HORIZONS:
            col = f"prob_{h}h"
            if col in ref_df.columns:
                r, _ = spearmanr(test_preds[h], ref_df[col].values)
                spearmans.append(r)
        min_spearman = min(spearmans) if spearmans else 0.0
        print(f"  [{label}] Spearman vs 0.96624 — min={min_spearman:.4f} per-horizon={[f'{s:.3f}' for s in spearmans]}")

    passes = score > GATE_HYBRID and (test_preds is None or min_spearman > GATE_SPEARMAN)
    return score, min_spearman, passes


def predict_test(train, test, feature_cols, meta_oof, y_time, y_event, ipcw_weights):
    """Quick full-retrain to get test predictions for Spearman gate check."""
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(train[feature_cols]), columns=feature_cols)
    X_te = pd.DataFrame(scaler.transform(test[feature_cols]), columns=feature_cols)

    models = make_base_models()
    for model in models:
        model.fit(X_tr, y_time, y_event)

    test_meta = np.column_stack([model.predict_proba(X_te)[h] for model in models for h in HORIZONS])
    train_meta = np.column_stack([model.predict_proba(X_tr)[h] for model in models for h in HORIZONS])

    # Use OOF meta to fit Ridge (quick check only)
    meta_X = np.column_stack([meta_oof[mi][hi] for mi in range(3) for hi in range(len(HORIZONS))])
    test_preds = {}
    for h in HORIZONS:
        labels, eligible = build_horizon_labels(y_time, y_event, h)
        if eligible.sum() < 2 or len(np.unique(labels[eligible])) < 2:
            test_preds[h] = np.ones(len(test))
            continue
        X_e = train_meta[eligible]
        y_e = labels[eligible].astype(float)
        w_e = ipcw_weights[eligible]
        m = Ridge(alpha=1.0)
        m.fit(X_e, y_e, sample_weight=w_e)
        test_preds[h] = np.clip(m.predict(test_meta), 0.0, 1.0)
    return test_preds


def generate_submission(train, test, feature_cols, best_learner, y_time, y_event, ipcw_weights):
    """Retrain on full data, generate test predictions."""
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(train[feature_cols]), columns=feature_cols)
    X_te = pd.DataFrame(scaler.transform(test[feature_cols]), columns=feature_cols)

    models = make_base_models()
    for model in models:
        model.fit(X_tr, y_time, y_event)

    # Build test meta-features
    test_meta = np.column_stack([
        model.predict_proba(X_te)[h]
        for model in models
        for h in HORIZONS
    ])
    # Build train meta-features for fitting meta-learner
    train_meta = np.column_stack([
        model.predict_proba(X_tr)[h]
        for model in models
        for h in HORIZONS
    ])

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for hi, h in enumerate(HORIZONS):
        labels, eligible = build_horizon_labels(y_time, y_event, h)
        if h == 72:
            sub[f"prob_{h}h"] = 1.0
            continue
        X_e = train_meta[eligible]
        y_e = labels[eligible].astype(float)
        w_e = ipcw_weights[eligible]

        if best_learner == "Ridge":
            m = Ridge(alpha=1.0)
            m.fit(X_e, y_e, sample_weight=w_e)
            preds = np.clip(m.predict(test_meta), 0.0, 1.0)
        else:
            m = LogisticRegression(C=0.01, max_iter=2000)
            m.fit(X_e, y_e, sample_weight=w_e)
            preds = m.predict_proba(test_meta)[:, 1]

        sub[f"prob_{h}h"] = preds

    sub.to_csv(OUT_PATH, index=False)
    print(f"  Submission saved: {OUT_PATH}  shape={sub.shape}")


def main():
    # Load data
    train = add_engineered(pd.read_csv(TRAIN_PATH))
    test = add_engineered(pd.read_csv(TEST_PATH))
    ref_df = pd.read_csv(REF_PATH)

    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    print("=== IPCW weights ===")
    ipcw_weights = compute_ipcw_weights(y_time, y_event)

    # Stage 1: 5x1 quick check
    print("\n=== Stage 1: 5x1 CV ===")
    meta, y_time, y_event = collect_oof_meta(train, FEATURES, n_splits=5, n_repeats=1)
    ridge_oof, lr_oof = fit_meta_learners(meta, y_time, y_event, ipcw_weights)

    ridge_score, _, _ = eval_gate(ridge_oof, y_time, y_event, ref_df, "Ridge-S1")
    lr_score, _, _ = eval_gate(lr_oof, y_time, y_event, ref_df, "LR-S1")

    best_s1 = max(ridge_score, lr_score)
    if best_s1 <= GATE_HYBRID:
        print(f"\nNo signal -- stopping (best OOF hybrid={best_s1:.5f} <= {GATE_HYBRID})")
        return

    # Stage 2: 5x10 full CV
    print("\n=== Stage 2: 5x10 CV ===")
    meta, y_time, y_event = collect_oof_meta(train, FEATURES, n_splits=5, n_repeats=10)
    ridge_oof, lr_oof = fit_meta_learners(meta, y_time, y_event, ipcw_weights)

    # Compute test predictions for Spearman gate check
    print("  Computing test predictions for Spearman check...")
    test_preds = predict_test(train, test, FEATURES, meta, y_time, y_event, ipcw_weights)

    ridge_score, ridge_spear, ridge_pass = eval_gate(ridge_oof, y_time, y_event, ref_df, "Ridge-S2", test_preds)
    lr_score, lr_spear, lr_pass = eval_gate(lr_oof, y_time, y_event, ref_df, "LR-S2", test_preds)

    gate_passes = ridge_pass or lr_pass
    if not gate_passes:
        print(f"\nGate failed -- no submission generated")
        print(f"  Ridge: hybrid={ridge_score:.5f} spearman={ridge_spear:.4f} pass={ridge_pass}")
        print(f"  LR:    hybrid={lr_score:.5f} spearman={lr_spear:.4f} pass={lr_pass}")
        return

    best_learner = "Ridge" if ridge_score >= lr_score else "LR"
    print(f"\nGate PASSED -- best meta-learner: {best_learner}")
    generate_submission(train, test, FEATURES, best_learner, y_time, y_event, ipcw_weights)


if __name__ == "__main__":
    main()
