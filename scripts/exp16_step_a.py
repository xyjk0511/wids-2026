"""Exp16 Step A: Risk-score ranking + quantile alignment for 12h only.

Isolated experiment: lock 24/48/72h to 0.96624 reference values,
replace only 12h with risk-score-based quantile-aligned probabilities.

Usage:
    python -m scripts.exp16_step_a
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import (
    TRAIN_PATH, TEST_PATH, HORIZONS,
    N_SPLITS, N_REPEATS, RANDOM_STATE,
    TIME_COL, EVENT_COL,
)
from src.features import remove_redundant, add_engineered, get_feature_set
from src.models import RSF, EST
from src.train import _strat_labels

FEATURE_LEVEL = "medium"
REF_PATH = "submission_0.96624.csv"
OUT_PATH = "sub_96624_p12risk_quantile.csv"


def main():
    # ---- Load data ----
    print("=== Loading data ===")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train = add_engineered(remove_redundant(train))
    test = add_engineered(remove_redundant(test))
    feature_cols = get_feature_set(train, level=FEATURE_LEVEL)
    print(f"  Features ({len(feature_cols)}): {feature_cols[:5]}...")

    X_train = train[feature_cols]
    X_test = test[feature_cols]
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values
    n_train = len(train)
    n_test = len(test)

    # ---- CV loop: collect risk scores ----
    print(f"\n=== CV Risk Collection (5x10 = 50 folds) ===")
    strat = _strat_labels(y_time, y_event, mode="event_time", n_splits=N_SPLITS)
    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE,
    )

    rsf_risk_oof = np.zeros(n_train)
    est_risk_oof = np.zeros(n_train)
    oof_counts = np.zeros(n_train)

    rsf_risk_test_acc = np.zeros(n_test)
    est_risk_test_acc = np.zeros(n_test)
    n_folds = 0

    for fold_idx, (tr_idx, va_idx) in enumerate(rskf.split(X_train, strat), 1):
        X_tr = X_train.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]

        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(
            scaler.fit_transform(X_tr), columns=feature_cols, index=X_tr.index,
        )
        X_va_s = pd.DataFrame(
            scaler.transform(X_va), columns=feature_cols, index=X_va.index,
        )
        X_te_s = pd.DataFrame(
            scaler.transform(X_test), columns=feature_cols, index=X_test.index,
        )

        rsf = RSF(min_samples_leaf=5)
        est = EST(min_samples_leaf=5)
        rsf.fit(X_tr_s, y_time[tr_idx], y_event[tr_idx])
        est.fit(X_tr_s, y_time[tr_idx], y_event[tr_idx])

        rsf_risk_oof[va_idx] += rsf.predict_risk(X_va_s)
        est_risk_oof[va_idx] += est.predict_risk(X_va_s)
        oof_counts[va_idx] += 1

        rsf_risk_test_acc += rsf.predict_risk(X_te_s)
        est_risk_test_acc += est.predict_risk(X_te_s)
        n_folds += 1

        if fold_idx % N_SPLITS == 0:
            rep = fold_idx // N_SPLITS
            print(f"  Repeat {rep}/{N_REPEATS} done")

    # Average
    mask = oof_counts > 0
    rsf_risk_oof[mask] /= oof_counts[mask]
    est_risk_oof[mask] /= oof_counts[mask]
    rsf_risk_test = rsf_risk_test_acc / n_folds
    est_risk_test = est_risk_test_acc / n_folds

    # Sanity check: ties
    print(f"\n=== Risk Score Diagnostics ===")
    print(f"  RSF OOF: unique={len(np.unique(rsf_risk_oof))}/{n_train}")
    print(f"  EST OOF: unique={len(np.unique(est_risk_oof))}/{n_train}")
    print(f"  RSF test: unique={len(np.unique(rsf_risk_test))}/{n_test}")
    print(f"  EST test: unique={len(np.unique(est_risk_test))}/{n_test}")

    # ---- Rank-ensemble ----
    # Use ranks instead of raw values to avoid scale domination
    rsf_rank_test = rankdata(rsf_risk_test)
    est_rank_test = rankdata(est_risk_test)
    risk_blend_rank = (rsf_rank_test + est_rank_test) / 2.0

    # Break ties using standardized raw risk average
    rsf_std = (rsf_risk_test - rsf_risk_test.mean()) / (rsf_risk_test.std() + 1e-12)
    est_std = (est_risk_test - est_risk_test.mean()) / (est_risk_test.std() + 1e-12)
    raw_blend = (rsf_std + est_std) / 2.0
    # Add tiny tiebreaker: scale raw_blend to be << 0.5 (rank spacing)
    tiebreaker = raw_blend * 1e-6
    risk_blend_final = risk_blend_rank + tiebreaker

    print(f"\n  Rank-blend test: unique={len(np.unique(risk_blend_rank))}/{n_test}")
    print(f"  After tiebreak:  unique={len(np.unique(risk_blend_final))}/{n_test}")

    # ---- Quantile alignment to 0.96624 reference ----
    print(f"\n=== Quantile Alignment ===")
    ref = pd.read_csv(REF_PATH)
    ref_p12 = np.sort(ref["prob_12h"].values)

    # Map risk ranks to percentiles, then interpolate onto reference distribution
    pct = rankdata(risk_blend_final) / (n_test + 1)  # (0, 1) percentiles
    ref_quantiles = np.linspace(0, 1, len(ref_p12))
    p12_new = np.interp(pct, ref_quantiles, ref_p12)

    # Monotonic protection: p12 must be < p24
    p24_ref = ref["prob_24h"].values
    p12_new = np.minimum(p12_new, p24_ref - 1e-6)

    # ---- Build submission ----
    sub = ref.copy()
    sub["prob_12h"] = p12_new
    # 24/48/72h locked to reference values (already in ref)

    sub.to_csv(OUT_PATH, index=False)
    print(f"  Saved: {OUT_PATH}")

    # ---- Acceptance diagnostics ----
    print(f"\n=== Acceptance Diagnostics ===")

    # 1. Five-number summary comparison
    print("\n  1) p12 Five-Number Summary:")
    for label, vals in [("new", p12_new), ("ref_96624", ref["prob_12h"].values)]:
        print(f"     {label:12s}: min={vals.min():.6f} Q1={np.percentile(vals,25):.6f} "
              f"med={np.median(vals):.6f} Q3={np.percentile(vals,75):.6f} max={vals.max():.6f}")

    # 2. Monotonicity violations
    n_violations = (p12_new >= p24_ref).sum()
    print(f"\n  2) p12 >= p24 violations: {n_violations}")

    # 3. Unique values
    print(f"\n  3) unique(p12_new) = {len(np.unique(p12_new))} (target: ~95)")

    # 4. Spearman correlation with reference
    sr, pval = spearmanr(p12_new, ref["prob_12h"].values)
    print(f"\n  4) Spearman(p12_new, p12_ref): rho={sr:.4f}  p={pval:.2e}")

    # Bonus: rank changes
    ref_ranks = rankdata(ref["prob_12h"].values)
    new_ranks = rankdata(p12_new)
    max_rank_change = np.max(np.abs(ref_ranks - new_ranks))
    mean_rank_change = np.mean(np.abs(ref_ranks - new_ranks))
    print(f"     Mean rank change: {mean_rank_change:.1f}  Max: {max_rank_change:.0f}")

    print(f"\n  Preview:\n{sub.head()}")


if __name__ == "__main__":
    main()
