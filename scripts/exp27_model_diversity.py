"""Exp27: Test model diversity — RSF, XGBoostAFT, CoxPH as anchor candidates."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from src.config import (
    TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH, TIME_COL, EVENT_COL,
    HORIZONS, PROB_COLS, RANDOM_STATE, FEATURES_V96624_BASE, FEATURES_V96624_ENGINEERED,
)
from src.features import add_engineered
from src.models import RSF, EST, CoxPH, XGBoostAFT
from src.evaluation import hybrid_score, c_index
from src.train import _strat_labels
from src.monotonic import enforce_monotonicity

FEATURES = list(FEATURES_V96624_BASE) + list(FEATURES_V96624_ENGINEERED)
N_SPLITS, N_REPEATS = 5, 10
SEEDS = [42, 123, 456, 789, 2026]


def run_oof(train, model_fn, feature_cols, n_splits=N_SPLITS, n_repeats=N_REPEATS):
    """Run OOF for a single model type."""
    X = train[feature_cols]
    y_time, y_event = train[TIME_COL].values, train[EVENT_COL].values
    n = len(train)
    oof = {h: np.zeros(n) for h in HORIZONS}
    counts = np.zeros(n)

    strat = _strat_labels(y_time, y_event)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)

    fold_idx = 0
    for tr_idx, va_idx in rskf.split(X, strat):
        fold_idx += 1
        scaler = StandardScaler()
        X_tr = pd.DataFrame(scaler.fit_transform(X.iloc[tr_idx]), columns=feature_cols, index=X.iloc[tr_idx].index)
        X_va = pd.DataFrame(scaler.transform(X.iloc[va_idx]), columns=feature_cols, index=X.iloc[va_idx].index)

        model = model_fn()
        model.fit(X_tr, y_time[tr_idx], y_event[tr_idx])
        preds = model.predict_proba(X_va)
        for h in HORIZONS:
            oof[h][va_idx] += preds[h]
        counts[va_idx] += 1

        if fold_idx % n_splits == 0:
            print(f"    Repeat {fold_idx // n_splits}/{n_repeats}")

    mask = counts > 0
    for h in HORIZONS:
        oof[h][mask] /= counts[mask]
    return oof


def eval_blend(oofs, weights, y_time, y_event, label=""):
    """Evaluate weighted blend of multiple OOFs."""
    blend = {h: sum(w * oofs[name][h] for name, w in weights.items()) for h in HORIZONS}
    score, det = hybrid_score(y_time, y_event, blend)
    ci = det['c_index']
    wb = det['weighted_brier']
    print(f"  {label:30s}  Hybrid={score:.4f}  CI={ci:.4f}  WBrier={wb:.4f}")
    return score, blend


def retrain_predict(train, test, model_fn, feature_cols, seeds=SEEDS):
    """Full retrain with multi-seed averaging."""
    X_train, X_test = train[feature_cols], test[feature_cols]
    y_time, y_event = train[TIME_COL].values, train[EVENT_COL].values
    preds_all = {h: [] for h in HORIZONS}

    for seed in seeds:
        scaler = StandardScaler()
        X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
        X_te = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
        try:
            model = model_fn(random_state=seed)
        except TypeError:
            model = model_fn()
        model.fit(X_tr, y_time, y_event)
        preds = model.predict_proba(X_te)
        for h in HORIZONS:
            preds_all[h].append(preds[h])
        print(f"    Seed {seed} done")

    return {h: np.mean(preds_all[h], axis=0) for h in HORIZONS}


def save_submission(test_preds, path):
    """Save submission with monotonicity enforcement."""
    test_preds = enforce_monotonicity(test_preds)
    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = test_preds[h]
    sub.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # Spearman vs anchor
    try:
        ref = pd.read_csv("submissions/submission_0.96624.csv")
        from scipy.stats import spearmanr
        for col in PROB_COLS[:-1]:
            sr, _ = spearmanr(sub[col], ref[col])
            print(f"    {col}: rho={sr:.4f}")
    except FileNotFoundError:
        pass


def main():
    print("=== Loading data ===")
    train = add_engineered(pd.read_csv(TRAIN_PATH))
    test = add_engineered(pd.read_csv(TEST_PATH))
    feature_cols = [c for c in FEATURES if c in train.columns]
    y_time, y_event = train[TIME_COL].values, train[EVENT_COL].values
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    # --- OOF for each model ---
    oofs = {}

    print("\n=== RSF OOF ===")
    oofs["RSF"] = run_oof(train, lambda: RSF(random_state=RANDOM_STATE), feature_cols)

    print("\n=== XGBoostAFT OOF ===")
    oofs["XGB"] = run_oof(train, lambda: XGBoostAFT(random_state=RANDOM_STATE), feature_cols)

    print("\n=== CoxPH OOF ===")
    oofs["Cox"] = run_oof(train, lambda: CoxPH(features=feature_cols), feature_cols)

    # --- Individual scores ---
    print("\n=== Individual Model Scores ===")
    for name in oofs:
        eval_blend(oofs, {name: 1.0}, y_time, y_event, label=name)

    # --- Ensemble weight search ---
    print("\n=== 3-Model Weight Search ===")
    best_score, best_w = -1, None
    for wr in range(0, 11, 2):
        for wx in range(0, 11 - wr, 2):
            wc = 10 - wr - wx
            w = {"RSF": wr/10, "XGB": wx/10, "Cox": wc/10}
            score, _ = eval_blend(oofs, w, y_time, y_event,
                                  label=f"RSF={wr/10:.1f} XGB={wx/10:.1f} Cox={wc/10:.1f}")
            if score > best_score:
                best_score, best_w = score, dict(w)

    print(f"\n  Best: {best_w} → Hybrid={best_score:.4f}")

    # --- Generate submissions for top configs ---
    configs = [
        ("RSF_only", {"RSF": 1.0}),
        ("XGB_only", {"XGB": 1.0}),
        ("Cox_only", {"Cox": 1.0}),
        ("best_blend", best_w),
    ]

    # Retrain all models
    print("\n=== Full Retrain ===")
    test_preds = {}

    print("  RSF:")
    test_preds["RSF"] = retrain_predict(train, test, lambda **kw: RSF(**kw), feature_cols)

    print("  XGBoostAFT:")
    test_preds["XGB"] = retrain_predict(train, test, lambda **kw: XGBoostAFT(**kw), feature_cols)

    print("  CoxPH (single, no seed):")
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(train[feature_cols]), columns=feature_cols, index=train.index)
    X_te = pd.DataFrame(scaler.transform(test[feature_cols]), columns=feature_cols, index=test.index)
    cox = CoxPH(features=feature_cols)
    cox.fit(X_tr, y_time, y_event)
    test_preds["Cox"] = cox.predict_proba(X_te)
    print("    Done")

    # Save submissions
    print("\n=== Saving Submissions ===")
    for name, w in configs:
        blend = {h: sum(w.get(m, 0) * test_preds[m][h] for m in test_preds if m in w)
                 for h in HORIZONS}
        save_submission(blend, f"submissions/submission_exp27_{name}.csv")


if __name__ == "__main__":
    main()
