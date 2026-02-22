"""Exp31: IPCW-aware stacking with RSF+EST+GBSA base models.

Two-stage CV: 5x1 quick check, 5x10 only if OOF hybrid > 0.9697.
Go/no-go gate: OOF hybrid > 0.9697 AND Spearman vs 0.96624 > 0.90.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
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


def fit_censoring_km(y_time, y_event):
    """Fit KM on censoring distribution. Returns fitted KMF."""
    kmf = KaplanMeierFitter()
    kmf.fit(y_time, event_observed=(1 - y_event))
    return kmf


def compute_ipcw_weights_horizon(kmf, y_time, y_event, horizon, eligible):
    """Horizon-aware IPCW: event→1/G(t), survived→1/G(h), censored<h excluded by eligible."""
    t_e = y_time[eligible]
    ev_e = y_event[eligible]
    n = eligible.sum()
    weights = np.ones(n)
    event_mask = ev_e == 1
    if event_mask.any():
        g_t = kmf.survival_function_at_times(t_e[event_mask]).values
        weights[event_mask] = 1.0 / np.clip(g_t, 0.05, None)
    surv_mask = ~event_mask
    if surv_mask.any():
        g_h = max(kmf.survival_function_at_times([horizon]).values[0], 0.05)
        weights[surv_mask] = 1.0 / g_h
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


def fit_meta_learners(meta, y_time, y_event):
    """Cross-fitted Ridge and LR per horizon with fold-local KM and IPCW."""
    n = len(y_time)
    ridge_oof = {h: np.full(n, np.nan) for h in HORIZONS}
    lr_oof = {h: np.full(n, np.nan) for h in HORIZONS}

    meta_X = np.column_stack([meta[mi][hi] for mi in range(3) for hi in range(len(HORIZONS))])

    for hi, h in enumerate(HORIZONS):
        labels, eligible = build_horizon_labels(y_time, y_event, h)
        if eligible.sum() < 10:
            continue
        X_e = meta_X[eligible]
        y_e = labels[eligible].astype(float)
        elig_idx = np.where(eligible)[0]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, va in skf.split(X_e, y_e.astype(int)):
            # Refit KM on inner-train fold only
            kmf_fold = fit_censoring_km(y_time[elig_idx[tr]], y_event[elig_idx[tr]])
            elig_tr = np.zeros(n, dtype=bool); elig_tr[elig_idx[tr]] = True
            w_tr = compute_ipcw_weights_horizon(kmf_fold, y_time, y_event, h, elig_tr)

            ridge = Ridge(alpha=1.0)
            ridge.fit(X_e[tr], y_e[tr], sample_weight=w_tr)
            ridge_oof[h][elig_idx[va]] = np.clip(ridge.predict(X_e[va]), 0.0, 1.0)

            if len(np.unique(y_e[tr])) < 2:
                lr_oof[h][elig_idx[va]] = y_e[va]
            else:
                lr = LogisticRegression(C=0.01, max_iter=2000)
                lr.fit(X_e[tr], y_e[tr], sample_weight=w_tr)
                lr_oof[h][elig_idx[va]] = lr.predict_proba(X_e[va])[:, 1]

        # Fill non-eligible with eligible mean (neutral for CI)
        elig_mean_r = np.nanmean(ridge_oof[h][eligible])
        elig_mean_l = np.nanmean(lr_oof[h][eligible])
        ridge_oof[h][~eligible] = elig_mean_r
        lr_oof[h][~eligible] = elig_mean_l
        print(f"    h={h}: n_elig={eligible.sum()} non_elig={n-eligible.sum()} fill={elig_mean_r:.4f}")

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


def predict_test(train, test, feature_cols, meta_oof, y_time, y_event):
    """Full-retrain test predictions for both Ridge and LR."""
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(train[feature_cols]), columns=feature_cols)
    X_te = pd.DataFrame(scaler.transform(test[feature_cols]), columns=feature_cols)

    models = make_base_models()
    for model in models:
        model.fit(X_tr, y_time, y_event)

    test_meta = np.column_stack([model.predict_proba(X_te)[h] for model in models for h in HORIZONS])
    train_meta = np.column_stack([model.predict_proba(X_tr)[h] for model in models for h in HORIZONS])

    kmf_full = fit_censoring_km(y_time, y_event)
    ridge_preds, lr_preds = {}, {}
    for h in HORIZONS:
        labels, eligible = build_horizon_labels(y_time, y_event, h)
        if eligible.sum() < 2 or len(np.unique(labels[eligible])) < 2:
            ridge_preds[h] = np.ones(len(test))
            lr_preds[h] = np.ones(len(test))
            continue
        X_e = train_meta[eligible]
        y_e = labels[eligible].astype(float)
        w_e = compute_ipcw_weights_horizon(kmf_full, y_time, y_event, h, eligible)

        m_r = Ridge(alpha=1.0)
        m_r.fit(X_e, y_e, sample_weight=w_e)
        ridge_preds[h] = np.clip(m_r.predict(test_meta), 0.0, 1.0)

        m_l = LogisticRegression(C=0.01, max_iter=2000)
        m_l.fit(X_e, y_e, sample_weight=w_e)
        lr_preds[h] = m_l.predict_proba(test_meta)[:, 1]

    return ridge_preds, lr_preds


def generate_submission(train, test, feature_cols, best_learner, y_time, y_event):
    """Retrain on full data, generate test predictions."""
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(train[feature_cols]), columns=feature_cols)
    X_te = pd.DataFrame(scaler.transform(test[feature_cols]), columns=feature_cols)

    models = make_base_models()
    for model in models:
        model.fit(X_tr, y_time, y_event)

    test_meta = np.column_stack([model.predict_proba(X_te)[h] for model in models for h in HORIZONS])
    train_meta = np.column_stack([model.predict_proba(X_tr)[h] for model in models for h in HORIZONS])

    kmf_full = fit_censoring_km(y_time, y_event)
    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h in HORIZONS:
        labels, eligible = build_horizon_labels(y_time, y_event, h)
        if h == 72:
            sub[f"prob_{h}h"] = 1.0
            continue
        X_e = train_meta[eligible]
        y_e = labels[eligible].astype(float)
        w_e = compute_ipcw_weights_horizon(kmf_full, y_time, y_event, h, eligible)

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

    # Stage 1: 5x1 quick check
    print("\n=== Stage 1: 5x1 CV ===")
    meta, y_time, y_event = collect_oof_meta(train, FEATURES, n_splits=5, n_repeats=1)
    ridge_oof, lr_oof = fit_meta_learners(meta, y_time, y_event)

    ridge_score, _, _ = eval_gate(ridge_oof, y_time, y_event, ref_df, "Ridge-S1")
    lr_score, _, _ = eval_gate(lr_oof, y_time, y_event, ref_df, "LR-S1")

    best_s1 = max(ridge_score, lr_score)
    if best_s1 <= GATE_HYBRID:
        print(f"\nNo signal -- stopping (best OOF hybrid={best_s1:.5f} <= {GATE_HYBRID})")
        return

    # Stage 2: 5x10 full CV
    print("\n=== Stage 2: 5x10 CV ===")
    meta, y_time, y_event = collect_oof_meta(train, FEATURES, n_splits=5, n_repeats=10)
    ridge_oof, lr_oof = fit_meta_learners(meta, y_time, y_event)

    # Compute test predictions for Spearman gate check (separate for Ridge/LR)
    print("  Computing test predictions for Spearman check...")
    ridge_test, lr_test = predict_test(train, test, FEATURES, meta, y_time, y_event)

    ridge_score, ridge_spear, ridge_pass = eval_gate(ridge_oof, y_time, y_event, ref_df, "Ridge-S2", ridge_test)
    lr_score, lr_spear, lr_pass = eval_gate(lr_oof, y_time, y_event, ref_df, "LR-S2", lr_test)

    gate_passes = ridge_pass or lr_pass
    if not gate_passes:
        print(f"\nGate failed -- no submission generated")
        print(f"  Ridge: hybrid={ridge_score:.5f} spearman={ridge_spear:.4f} pass={ridge_pass}")
        print(f"  LR:    hybrid={lr_score:.5f} spearman={lr_spear:.4f} pass={lr_pass}")
        return

    best_learner = "Ridge" if ridge_score >= lr_score else "LR"
    print(f"\nGate PASSED -- best meta-learner: {best_learner}")
    generate_submission(train, test, FEATURES, best_learner, y_time, y_event)


if __name__ == "__main__":
    main()
