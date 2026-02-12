"""Main training script: RSF single model + postprocessing + submission."""

import warnings

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
from src.models import RSF, EST, GBSA
from src.monotonic import enforce_monotonicity, submission_postprocess

warnings.filterwarnings("ignore")


FEATURE_LEVEL = "medium"


def load_data():
    """Load and prepare train/test data."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if FEATURE_LEVEL in ("v96624", "v96624_plus"):
        train = add_engineered(train)
        test = add_engineered(test)
    else:
        train = add_engineered(remove_redundant(train))
        test = add_engineered(remove_redundant(test))

    return train, test



def _strat_labels(y_time, y_event):
    """Build 2-class stratification labels: event(1) vs censored(0)."""
    return y_event.astype(int)


def run_cv(train, feature_cols, rsf_kwargs=None):
    """Run repeated stratified K-fold CV with RSF, collect OOF predictions."""
    X = train[feature_cols]
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    n = len(train)
    oof_preds = {h: np.zeros(n) for h in HORIZONS}
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
        yt_tr = y_time[tr_idx]
        ye_tr = y_event[tr_idx]

        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=feature_cols, index=X_tr.index)
        X_va_s = pd.DataFrame(scaler.transform(X_va), columns=feature_cols, index=X_va.index)

        rsf = RSF(**(rsf_kwargs or {}))
        rsf.fit(X_tr_s, yt_tr, ye_tr)
        preds = rsf.predict_proba(X_va_s)

        for h in HORIZONS:
            oof_preds[h][va_idx] += preds[h]
        oof_counts[va_idx] += 1

        if fold_idx % N_SPLITS == 0:
            rep = fold_idx // N_SPLITS
            print(f"  Repeat {rep}/{N_REPEATS} done")

    for h in HORIZONS:
        mask = oof_counts > 0
        oof_preds[h][mask] /= oof_counts[mask]

    return oof_preds



def print_oof_scores(oof_preds, y_time, y_event):
    """Print RSF OOF scores using competition hybrid metric."""
    print("\n=== OOF Scores (RSF) ===")
    score, details = hybrid_score(y_time, y_event, oof_preds)
    print(f"  Hybrid={score:.4f}  CI={details['c_index']:.4f}  WBrier={details['weighted_brier']:.4f}")
    pp = submission_postprocess(oof_preds)
    sc2, det2 = hybrid_score(y_time, y_event, pp)
    print(f"  (postprocessed)  Hybrid={sc2:.4f}  CI={det2['c_index']:.4f}  WBrier={det2['weighted_brier']:.4f}")


def main():
    print("=== Loading data ===")
    train, test = load_data()
    feature_cols = get_feature_set(train, level=FEATURE_LEVEL)
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    # --- Step 1: CV (evaluation only) ---
    print("\n=== Running CV ===")
    oof_preds = run_cv(train, feature_cols)
    print_oof_scores(oof_preds, y_time, y_event)

    # --- Step 2: Full retrain -- Selective Horizon Blending ---
    # 12h: RSF+EST only (protect CI ranking)
    # 24h/48h: RSF+EST+GBSA (boost WBrier via model diversity)
    # 72h: hardcoded 1.0 (postprocess handles this)
    RETRAIN_SEEDS = [42, 123, 456, 789, 2026]
    print(f"\n=== Full retrain (RSF+EST+GBSA, {len(RETRAIN_SEEDS)} seeds x 3 models) ===")
    X_train = train[feature_cols]
    X_test = test[feature_cols]

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    rsf_est_preds = []  # RSF+EST predictions (used for all horizons)
    gbsa_preds = []     # GBSA predictions (used for 24h/48h only)
    for seed in RETRAIN_SEEDS:
        rsf = RSF(n_estimators=200, random_state=seed)
        rsf.fit(X_train_s, y_time, y_event)
        rsf_est_preds.append(rsf.predict_proba(X_test_s))
        print(f"  RSF seed {seed} done")

        est = EST(n_estimators=200, random_state=seed)
        est.fit(X_train_s, y_time, y_event)
        rsf_est_preds.append(est.predict_proba(X_test_s))
        print(f"  EST seed {seed} done")

        gbsa = GBSA(n_estimators=300, random_state=seed)
        gbsa.fit(X_train_s, y_time, y_event)
        gbsa_preds.append(gbsa.predict_proba(X_test_s))
        print(f"  GBSA seed {seed} done")

    # Selective blending per horizon
    test_preds = {}
    for h in HORIZONS:
        if h == 12:
            # 12h: RSF+EST only -- protect CI
            test_preds[h] = np.mean([p[h] for p in rsf_est_preds], axis=0)
        elif h in (24, 48):
            # 24h/48h: RSF+EST+GBSA -- boost WBrier diversity
            all_h = [p[h] for p in rsf_est_preds] + [p[h] for p in gbsa_preds]
            test_preds[h] = np.mean(all_h, axis=0)
        else:
            # 72h: postprocess will set to 1.0
            test_preds[h] = np.mean([p[h] for p in rsf_est_preds], axis=0)
    print(f"  {len(rsf_est_preds)} RSF+EST + {len(gbsa_preds)} GBSA models trained")
    print(f"  12h: {len(rsf_est_preds)} models | 24h/48h: {len(rsf_est_preds)+len(gbsa_preds)} models")

    # --- Step 3: Postprocess + submission ---
    print("\n=== Generating submission ===")
    test_preds = submission_postprocess(test_preds)
    test_preds = enforce_monotonicity(test_preds)

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = test_preds[h]

    # --- Validation checks ---
    # 72h all-ones check
    h72_col = PROB_COLS[HORIZONS.index(72)]
    h72_ok = (sub[h72_col] == 1.0).all()

    # Monotonicity check (12h <= 24h <= 48h <= 72h)
    mono_ok = True
    for i in range(len(sub)):
        vals = [sub[col].iloc[i] for col in PROB_COLS]
        for j in range(1, len(vals)):
            if vals[j] < vals[j - 1] - 1e-9:
                mono_ok = False
                break

    print(f"  72h all-ones check: {'PASS' if h72_ok else 'FAIL'}")
    print(f"  Monotonicity check: {'PASS' if mono_ok else 'FAIL'}")
    print(f"  Shape: {sub.shape}")

    # Probability distribution diagnostics
    print("\n=== Probability Distribution (vs 0.96624 target) ===")
    for col in PROB_COLS:
        vals = sub[col]
        print(f"  {col}: min={vals.min():.4f} median={vals.median():.4f} max={vals.max():.4f}")
    print("  Target 12h: min~0.036 median~0.15 max~0.99")

    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n  Submission saved to {SUBMISSION_PATH}")
    print(f"  Preview:\n{sub.head()}")

    # --- Spearman diagnostic vs 0.96624 reference ---
    ref_path = "submission 0.96624.csv"
    try:
        ref = pd.read_csv(ref_path)
        from scipy.stats import spearmanr
        print("\n=== Spearman Rank Correlation vs 0.96624 ===")
        for col in PROB_COLS[:-1]:  # skip 72h (all 1.0)
            sr, pval = spearmanr(sub[col], ref[col])
            print(f"  {col}: rho={sr:.4f}  p={pval:.2e}")
    except FileNotFoundError:
        print(f"\n  [WARN] Reference file '{ref_path}' not found, skipping Spearman diagnostic")


if __name__ == "__main__":
    main()
