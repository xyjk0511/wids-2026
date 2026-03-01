"""Exp34: GBSA 50-Model Ensemble (复现 0.97092 核心)

Quick validation: 2 configs × 5 seeds = 10 models
Full version: 5 configs × 10 seeds = 50 models

Target: OOF hybrid > 0.965 (quick gate), > 0.970 (full gate)

Ablation studies (P5-18):
- dropout_rate: 0.0 vs 0.1
- seeds: 3 vs 5 vs 10
- feature_set: v96624 vs medium
"""

import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import (
    TRAIN_PATH, TEST_PATH, SUBMISSION_PATH,
    ID_COL, TIME_COL, EVENT_COL, HORIZONS, PROB_COLS,
    N_SPLITS, N_REPEATS, RANDOM_STATE,
)
from src.features import remove_redundant, add_engineered, get_feature_set
from src.evaluation import hybrid_score
from src.models import GBSA

warnings.filterwarnings("ignore")


# ============================================================
# GBSA Configs (from 0.97092 analysis)
# ============================================================

# Quick validation: 2 configs × 5 seeds = 10 models
QUICK_CONFIGS = [
    {'lr': 0.01, 'ss': 0.7,  'msl': 12, 'n': 1200},
    {'lr': 0.01, 'ss': 0.85, 'msl': 15, 'n': 1200},
]

# Full version: 5 configs × 10 seeds = 50 models
FULL_CONFIGS = [
    {'lr': 0.01, 'ss': 0.7,  'msl': 12, 'n': 1200},
    {'lr': 0.01, 'ss': 0.85, 'msl': 15, 'n': 1200},
    {'lr': 0.01, 'ss': 0.6,  'msl': 12, 'n': 1200},
    {'lr': 0.005,'ss': 0.85, 'msl': 12, 'n': 2000},
    {'lr': 0.01, 'ss': 0.85, 'msl': 20, 'n': 1400},
]

QUICK_SEEDS = [42, 43, 44, 45, 46]
FULL_SEEDS = list(range(42, 52))  # 42-51

# Ablation seeds (P5-18)
ABLATION_SEEDS_3 = [42, 43, 44]
ABLATION_SEEDS_5 = [42, 43, 44, 45, 46]
ABLATION_SEEDS_10 = list(range(42, 52))

FEATURE_LEVEL = "medium"


def load_data(feature_level=None):
    """Load and prepare train/test data."""
    feature_level = feature_level or FEATURE_LEVEL
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if feature_level == "v96624":
        train = add_engineered(train)
        test = add_engineered(test)
    else:  # medium
        train = add_engineered(remove_redundant(train))
        test = add_engineered(remove_redundant(test))

    return train, test


def _strat_labels(y_time, y_event):
    """Build 2-class stratification labels: event(1) vs censored(0)."""
    return y_event.astype(int)


def run_gbsa_ensemble(train, feature_cols, configs, seeds, mode="quick", dropout_rate=0.0):
    """Run GBSA multi-config ensemble with OOF predictions.

    Args:
        train: Training dataframe
        feature_cols: List of feature column names
        configs: List of GBSA config dicts
        seeds: List of random seeds
        mode: "quick" or "full"
        dropout_rate: GBSA dropout rate (0.0 for 0.97092, 0.1 for ablation)

    Returns:
        oof_preds: {horizon: oof_array} averaged across all models
        test_preds: {horizon: test_array} averaged across all models
    """
    X = train[feature_cols]
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    n = len(train)
    oof_preds = {h: np.zeros(n) for h in HORIZONS}
    oof_counts = np.zeros(n)

    # Load test data
    test = pd.read_csv(TEST_PATH)
    test = add_engineered(remove_redundant(test))
    X_test = test[feature_cols]
    test_preds = {h: np.zeros(len(test)) for h in HORIZONS}

    strat = _strat_labels(y_time, y_event)
    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    total_models = len(configs) * len(seeds)
    model_idx = 0

    print(f"\n{'='*60}")
    print(f"GBSA Ensemble - {mode.upper()} mode")
    print(f"Configs: {len(configs)}, Seeds: {len(seeds)}, Total: {total_models} models")
    print(f"Dropout rate: {dropout_rate}")
    print(f"{'='*60}\n")

    for config_idx, cfg in enumerate(configs, 1):
        print(f"Config {config_idx}/{len(configs)}: lr={cfg['lr']}, ss={cfg['ss']}, msl={cfg['msl']}, n={cfg['n']}")

        for seed_idx, seed in enumerate(seeds, 1):
            model_idx += 1
            print(f"  Model {model_idx}/{total_models} (seed={seed})...", end=" ", flush=True)

            # CV loop for this model
            for fold_idx, (tr_idx, va_idx) in enumerate(rskf.split(X, strat), 1):
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                yt_tr = y_time[tr_idx]
                ye_tr = y_event[tr_idx]

                scaler = StandardScaler()
                X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=feature_cols, index=X_tr.index)
                X_va_s = pd.DataFrame(scaler.transform(X_va), columns=feature_cols, index=X_va.index)
                X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

                # CRITICAL: dropout_rate configurable (P5-18 ablation)
                gbsa = GBSA(
                    n_estimators=cfg['n'],
                    max_depth=3,
                    learning_rate=cfg['lr'],
                    subsample=cfg['ss'],
                    dropout_rate=dropout_rate,  # 0.0 for 0.97092, 0.1 for ablation
                    random_state=seed,
                )
                gbsa.model.min_samples_leaf = cfg['msl']

                gbsa.fit(X_tr_s, yt_tr, ye_tr)
                preds_va = gbsa.predict_proba(X_va_s)
                preds_test = gbsa.predict_proba(X_test_s)

                for h in HORIZONS:
                    oof_preds[h][va_idx] += preds_va[h]
                    test_preds[h] += preds_test[h]

                oof_counts[va_idx] += 1

            print("done")

    # Average OOF predictions
    for h in HORIZONS:
        mask = oof_counts > 0
        oof_preds[h][mask] /= oof_counts[mask]

    # Average test predictions (total_models * N_SPLITS * N_REPEATS folds)
    n_folds = N_SPLITS * N_REPEATS
    for h in HORIZONS:
        test_preds[h] /= (total_models * n_folds)

    return oof_preds, test_preds


def print_oof_scores(oof_preds, y_time, y_event, mode="quick"):
    """Print OOF scores."""
    print(f"\n{'='*60}")
    print(f"OOF Scores - {mode.upper()} mode")
    print(f"{'='*60}")
    score, details = hybrid_score(y_time, y_event, oof_preds)
    print(f"  Hybrid = {score:.5f}")
    print(f"  CI     = {details['c_index']:.5f}")
    print(f"  WBrier = {details['weighted_brier']:.5f}")
    print(f"{'='*60}\n")

    # Gate check
    if mode == "quick":
        gate = 0.965
        status = "✓ PASS" if score >= gate else "✗ FAIL"
        print(f"Quick Gate (OOF >= {gate}): {status}")
        if score < gate:
            print("  → Stop-loss triggered. Check implementation.")
    else:
        gate = 0.970
        status = "✓ PASS" if score >= gate else "✗ FAIL"
        print(f"Full Gate (OOF >= {gate}): {status}")
        if score < gate:
            print(f"  → Below target. Gap: {gate - score:.5f}")

    print()
    return score


def save_submission(test_preds, test, mode="quick", ablation=None):
    """Save submission file."""
    sub = pd.DataFrame({ID_COL: test[ID_COL]})
    for i, h in enumerate(HORIZONS):
        sub[PROB_COLS[i]] = np.clip(test_preds[h], 0.0, 1.0)

    suffix = f"_exp34_{mode}"
    if ablation:
        suffix += f"_{ablation}"
    filename = SUBMISSION_PATH.replace(".csv", f"{suffix}.csv")
    sub.to_csv(filename, index=False)
    print(f"Submission saved: {filename}\n")
    return filename


def append_to_experiments_md(mode, oof_score, lb_score, config_summary, ablation=None):
    """Append experiment results to experiments.md (P5-19)."""
    exp_file = Path("experiments.md")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n## Exp34 GBSA Ensemble - {mode.upper()}"
    if ablation:
        entry += f" (Ablation: {ablation})"
    entry += f"\n**Date**: {timestamp}\n\n"
    entry += f"- **OOF Hybrid**: {oof_score:.5f}\n"
    if lb_score:
        entry += f"- **LB Score**: {lb_score:.5f}\n"
    else:
        entry += "- **LB Score**: (pending submission)\n"
    entry += f"- **Config**: {config_summary}\n"
    entry += f"- **Gate**: {'✓ PASS' if (mode == 'quick' and oof_score >= 0.965) or (mode == 'full' and oof_score >= 0.970) else '✗ FAIL'}\n"
    entry += "\n"

    with open(exp_file, "a", encoding="utf-8") as f:
        f.write(entry)

    print(f"Results appended to {exp_file}\n")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                        help="quick: 2 configs × 5 seeds, full: 5 configs × 10 seeds")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Exp34: GBSA Ensemble - {args.mode.upper()} mode")
    print(f"{'='*60}\n")

    # Load data
    train, test = load_data()
    feature_cols = get_feature_set(FEATURE_LEVEL)

    # Select configs and seeds
    configs = QUICK_CONFIGS if args.mode == "quick" else FULL_CONFIGS
    seeds = QUICK_SEEDS if args.mode == "quick" else FULL_SEEDS

    # Run ensemble
    oof_preds, test_preds = run_gbsa_ensemble(train, feature_cols, configs, seeds, mode=args.mode)

    # Evaluate
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values
    oof_score = print_oof_scores(oof_preds, y_time, y_event, mode=args.mode)

    # Save submission
    save_submission(test_preds, test, mode=args.mode)

    # Final recommendation
    print(f"{'='*60}")
    if args.mode == "quick":
        if oof_score >= 0.965:
            print("✓ Quick validation PASSED")
            print("  → Proceed to full mode: python scripts/exp34_gbsa_ensemble.py --mode full")
        else:
            print("✗ Quick validation FAILED")
            print("  → Check implementation, version, or configs")
    else:
        if oof_score >= 0.970:
            print("✓ Full validation PASSED")
            print("  → Submit to Kaggle for LB verification (target: LB > 0.968)")
        else:
            print("⚠ Full validation below target")
            print(f"  → Gap: {0.970 - oof_score:.5f}")
            print("  → Consider: (1) verify configs, (2) check feature engineering, (3) adjust gate")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
