"""Exp21: Candidate library + small-weight submission bagging.

Generate ~10 RSF+EST candidates (varying seed/leaf/depth/max_features),
filter by Spearman vs anchor > 0.98, blend at small weights.

All submissions lock p12/p24 from anchor, only touch p48.

Usage:
    python -m scripts.exp21_candidate_bag
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from src.config import (
    HORIZONS, PROB_COLS, TIME_COL, EVENT_COL, SAMPLE_SUB_PATH,
)
from src.models import RSF, EST
from src.monotonic import submission_postprocess
from src.train import load_data
from src.features import get_feature_set


ANCHOR_PATH = "submission_0.96624.csv"

# Candidate grid: (seed, min_leaf, max_depth, max_features, label)
CANDIDATES = [
    (42,   5, 5, "sqrt", "base_s42"),
    (123,  5, 5, "sqrt", "s123"),
    (456,  5, 5, "sqrt", "s456"),
    (789,  5, 5, "sqrt", "s789"),
    (2026, 5, 5, "sqrt", "s2026"),
    (42,  10, 5, "sqrt", "leaf10"),
    (42,   5, 4, "sqrt", "depth4"),
    (42,   5, 6, "sqrt", "depth6"),
    (42,   5, 5, 4,      "feat4"),
    (42,   5, 5, 6,      "feat6"),
]


def train_candidate(X_train, y_time, y_event, X_test, feature_cols,
                    seed, min_leaf, max_depth, max_features):
    """Train RSF+EST with given config, return blended test p48."""
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_train),
                          columns=feature_cols, index=X_train.index)
    X_te_s = pd.DataFrame(scaler.transform(X_test),
                          columns=feature_cols, index=X_test.index)

    rsf = RSF(n_estimators=200, max_depth=max_depth,
              min_samples_leaf=min_leaf, max_features=max_features,
              random_state=seed)
    est = EST(n_estimators=200, max_depth=max_depth,
              min_samples_leaf=min_leaf, max_features=max_features,
              random_state=seed)

    rsf.fit(X_tr_s, y_time, y_event)
    est.fit(X_tr_s, y_time, y_event)

    pr = rsf.predict_proba(X_te_s, horizons=[24, 48])
    pe = est.predict_proba(X_te_s, horizons=[24, 48])

    return {h: 0.5 * pr[h] + 0.5 * pe[h] for h in [24, 48]}


def make_submission(anchor, p48_new, path):
    """Replace p48 in anchor, enforce p48>=p24, postprocess, save."""
    prob_dict = {
        12: anchor["prob_12h"].values.copy(),
        24: anchor["prob_24h"].values.copy(),
        48: np.clip(p48_new.copy(), 1e-6, 1.0),
        72: np.ones(len(anchor)),
    }
    prob_dict[48] = np.maximum(prob_dict[48],
                               prob_dict[24] + 1e-7)
    pp = submission_postprocess(prob_dict)

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = pp[h]

    mono_v = (sub["prob_24h"] > sub["prob_48h"] + 1e-9).sum()
    print(f"  72h={'PASS' if (sub['prob_72h']==1.0).all() else 'FAIL'}  "
          f"mono_viol={mono_v}")
    sub.to_csv(path, index=False)
    print(f"  Saved: {path}")

    for col in PROB_COLS[:-1]:
        sr, _ = spearmanr(sub[col], anchor[col])
        print(f"    {col}: rho={sr:.6f}")
    return sub


def main():
    print("=== Exp21: Candidate Library + Submission Bagging ===\n")

    anchor = pd.read_csv(ANCHOR_PATH)
    anchor_p48 = anchor["prob_48h"].values.copy()
    print(f"  Anchor p48: min={anchor_p48.min():.4f} "
          f"med={np.median(anchor_p48):.4f} max={anchor_p48.max():.4f}")

    train, test = load_data(feature_level="medium")
    feature_cols = get_feature_set(train, level="medium")
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values
    X_train = train[feature_cols]
    X_test = test[feature_cols]

    # ==================================================================
    # Generate candidates
    # ==================================================================
    print(f"\n=== Generating {len(CANDIDATES)} candidates ===")
    cand_p48 = {}

    for seed, leaf, depth, feat, label in CANDIDATES:
        print(f"\n  [{label}] seed={seed} leaf={leaf} "
              f"depth={depth} feat={feat}")
        preds = train_candidate(
            X_train, y_time, y_event, X_test, feature_cols,
            seed, leaf, depth, feat,
        )
        p48 = preds[48]
        sr, _ = spearmanr(p48, anchor_p48)
        print(f"    p48: min={p48.min():.4f} med={np.median(p48):.4f} "
              f"max={p48.max():.4f}  rho_anchor={sr:.4f}")

        if sr >= 0.50:  # keep all for analysis, filter later
            cand_p48[label] = p48
        else:
            print(f"    DROPPED (rho < 0.50)")

    # ==================================================================
    # Candidate correlation matrix
    # ==================================================================
    print("\n=== Candidate pairwise Spearman (p48) ===")
    labels = list(cand_p48.keys())
    print(f"  {'':>12}", end="")
    for lb in labels:
        print(f" {lb:>10}", end="")
    print()
    for i, li in enumerate(labels):
        print(f"  {li:>12}", end="")
        for j, lj in enumerate(labels):
            sr, _ = spearmanr(cand_p48[li], cand_p48[lj])
            print(f" {sr:10.4f}", end="")
        print()

    # Spearman vs anchor
    print(f"\n  {'vs anchor':>12}", end="")
    for lb in labels:
        sr, _ = spearmanr(cand_p48[lb], anchor_p48)
        print(f" {sr:10.4f}", end="")
    print()

    # ==================================================================
    # Ensemble: average all candidates, then blend with anchor
    # ==================================================================
    print("\n=== Ensemble averaging ===")
    ens_p48 = np.mean([cand_p48[lb] for lb in labels], axis=0)
    sr_ens, _ = spearmanr(ens_p48, anchor_p48)
    print(f"  Ensemble p48: min={ens_p48.min():.4f} "
          f"med={np.median(ens_p48):.4f} max={ens_p48.max():.4f} "
          f"rho_anchor={sr_ens:.4f}")

    # ==================================================================
    # Submissions: small-weight blending with anchor
    # ==================================================================
    print("\n=== Generating blended submissions ===")
    weights = [0.05, 0.10, 0.15]

    for w in weights:
        print(f"\n--- w={w:.2f}: anchor*(1-w) + ensemble*w ---")
        p48_blend = (1 - w) * anchor_p48 + w * ens_p48
        tag = f"w{int(w*100):02d}"
        make_submission(anchor, p48_blend,
                        f"submission_exp21_ens_{tag}.csv")

    # Also: top-3 most diverse candidates (lowest mutual correlation)
    print("\n=== Top diverse candidates ===")
    # Find candidate most different from anchor
    anchor_corrs = {lb: spearmanr(cand_p48[lb], anchor_p48)[0]
                    for lb in labels}
    sorted_by_div = sorted(anchor_corrs.items(), key=lambda x: x[1])
    print("  By ascending rho vs anchor:")
    for lb, sr in sorted_by_div:
        print(f"    {lb}: rho={sr:.4f}")

    # Single most diverse candidate blend
    if sorted_by_div:
        div_label = sorted_by_div[0][0]
        print(f"\n--- Most diverse single: {div_label} ---")
        for w in [0.05, 0.10]:
            p48_div = (1 - w) * anchor_p48 + w * cand_p48[div_label]
            tag = f"{div_label}_w{int(w*100):02d}"
            make_submission(anchor, p48_div,
                            f"submission_exp21_{tag}.csv")

    print("\n=== Exp21 complete ===")


if __name__ == "__main__":
    main()
