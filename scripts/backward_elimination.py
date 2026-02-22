"""Backward elimination: iteratively remove least important features."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")

from src.train import load_data, run_cv
from src.features import get_feature_set
from src.stacking import train_horizon_heads
from src.evaluation import hybrid_score
from src.config import TIME_COL, EVENT_COL, RANDOM_STATE

N_REPEATS_FAST = 3
N_SPLITS = 5
DROP_THRESHOLD = 0.0005  # stop if removal hurts more than this
MIN_FEATURES = 10


def run_pipeline(train, feature_cols, y_time, y_event):
    """Run base CV + XGB head, return hybrid score."""
    oof = run_cv(
        train, feature_cols,
        n_repeats=N_REPEATS_FAST, n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
    )
    head_oof, _ = train_horizon_heads(
        X_features=train[feature_cols],
        base_oof=oof, y_time=y_time, y_event=y_event,
        n_splits=N_SPLITS, n_repeats=min(N_REPEATS_FAST, 3),
        random_state=RANDOM_STATE + 1000,
        head_model="xgb", use_orig_features=True,
        calibration_mode="auto",
    )
    prob_dict = {h: head_oof[h] for h in [12, 24, 48]}
    prob_dict[72] = np.ones(len(train))
    score, details = hybrid_score(y_time, y_event, prob_dict)
    return score, details


def main():
    print("=== Loading data ===")
    train, _ = load_data(feature_level="v96624_plus")
    all_features = get_feature_set(train, level="v96624_plus")
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    current_features = list(all_features)
    print(f"Starting features ({len(current_features)}): {current_features}\n")

    # Baseline score
    print("=== Baseline (all features) ===")
    baseline_score, baseline_det = run_pipeline(train, current_features, y_time, y_event)
    print(f"  Hybrid={baseline_score:.4f}  CI={baseline_det['c_index']:.4f}  WBrier={baseline_det['weighted_brier']:.4f}\n")

    history = [("ALL", len(current_features), baseline_score)]
    prev_score = baseline_score

    iteration = 0
    while len(current_features) > MIN_FEATURES:
        iteration += 1
        print(f"=== Iteration {iteration}: testing removal of {len(current_features)} features ===")

        scores = {}
        for feat in current_features:
            subset = [f for f in current_features if f != feat]
            score, _ = run_pipeline(train, subset, y_time, y_event)
            drop = prev_score - score
            tag = "HELPS" if drop < 0 else f"drop={drop:.4f}"
            print(f"  Remove {feat:35s} → Hybrid={score:.4f}  ({tag})")
            scores[feat] = score

        # Find least important feature (highest score when removed)
        best_feat = max(scores, key=scores.get)
        best_score = scores[best_feat]
        delta = prev_score - best_score

        if delta > DROP_THRESHOLD:
            print(f"\n  STOP: removing '{best_feat}' hurts by {delta:.4f} > threshold {DROP_THRESHOLD}")
            break

        current_features.remove(best_feat)
        print(f"\n  DROPPED: '{best_feat}' (score {best_score:.4f}, delta={delta:+.4f})")
        print(f"  Remaining: {len(current_features)} features\n")
        history.append((best_feat, len(current_features), best_score))
        prev_score = best_score

    # Summary
    print("\n" + "=" * 60)
    print("ELIMINATION HISTORY:")
    print(f"{'Step':>4}  {'Removed':35s}  {'#Feat':>5}  {'Hybrid':>8}")
    for i, (feat, n, sc) in enumerate(history):
        print(f"{i:4d}  {feat:35s}  {n:5d}  {sc:.4f}")

    print(f"\nFinal feature set ({len(current_features)}):")
    for f in current_features:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
