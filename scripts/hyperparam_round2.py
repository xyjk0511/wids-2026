"""Round 2: Fine-tune min_samples_leaf + n_estimators on best Round 1 config."""

import sys
import time
sys.path.insert(0, ".")

from src.train import load_data, run_cv, get_feature_set
from src.config import TIME_COL, EVENT_COL
from src.evaluation import hybrid_score
from src.monotonic import submission_postprocess

FEATURE_LEVEL = "medium"

# Best from Round 1
FIXED = {"max_features": 12, "max_depth": 5}

GRID = [
    {"min_samples_leaf": 3,  "min_samples_split": 6,  "n_estimators": 1000},
    {"min_samples_leaf": 3,  "min_samples_split": 6,  "n_estimators": 2000},
    {"min_samples_leaf": 5,  "min_samples_split": 10, "n_estimators": 1000},
    {"min_samples_leaf": 5,  "min_samples_split": 10, "n_estimators": 2000},
    {"min_samples_leaf": 8,  "min_samples_split": 16, "n_estimators": 1000},
    {"min_samples_leaf": 8,  "min_samples_split": 16, "n_estimators": 2000},
]


def main():
    print("=== Round 2: min_samples_leaf x n_estimators (mf=12, md=5) ===\n")
    train, _ = load_data()
    feature_cols = get_feature_set(train, level=FEATURE_LEVEL)
    print(f"Features ({len(feature_cols)})")

    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    results = []
    for i, params in enumerate(GRID):
        rsf_kwargs = {**FIXED, **params}
        label = f"leaf={params['min_samples_leaf']}, n={params['n_estimators']}"
        print(f"\n--- [{i+1}/{len(GRID)}] {label} ---")

        t0 = time.time()
        oof = run_cv(
            train,
            feature_cols,
            rsf_kwargs=rsf_kwargs,
            include_boosting=False,
            strat_mode="event_time",
        )
        oof_preds = oof["RSF"]
        elapsed = time.time() - t0

        raw_score, raw_det = hybrid_score(y_time, y_event, oof_preds)
        pp = submission_postprocess(oof_preds)
        pp_score, pp_det = hybrid_score(y_time, y_event, pp)

        results.append({
            "params": label,
            "raw_hybrid": raw_score,
            "raw_ci": raw_det["c_index"],
            "raw_wbrier": raw_det["weighted_brier"],
            "pp_hybrid": pp_score,
            "pp_ci": pp_det["c_index"],
            "pp_wbrier": pp_det["weighted_brier"],
            "time": elapsed,
        })
        print(f"  Raw:  Hybrid={raw_score:.4f} CI={raw_det['c_index']:.4f} WBrier={raw_det['weighted_brier']:.4f}")
        print(f"  Post: Hybrid={pp_score:.4f} CI={pp_det['c_index']:.4f} WBrier={pp_det['weighted_brier']:.4f}")
        print(f"  Time: {elapsed:.1f}s")

    print("\n\n=== RESULTS SORTED BY PP HYBRID (desc) ===")
    results.sort(key=lambda r: r["pp_hybrid"], reverse=True)
    print(f"{'Rank':<5} {'Params':<22} {'Raw Hybrid':<12} {'PP Hybrid':<12} {'CI':<10} {'WBrier':<10} {'Time':<8}")
    print("-" * 79)
    for rank, r in enumerate(results, 1):
        print(f"{rank:<5} {r['params']:<22} {r['raw_hybrid']:<12.4f} {r['pp_hybrid']:<12.4f} "
              f"{r['pp_ci']:<10.4f} {r['pp_wbrier']:<10.4f} {r['time']:<8.1f}")

    best = results[0]
    print(f"\nBest: {best['params']} -> PP Hybrid={best['pp_hybrid']:.4f}")
    print("Round 1 best (mf=12, md=5, leaf=5, n=1000): PP Hybrid=0.9724")
    print("Original baseline (mf=sqrt, md=5, leaf=5, n=1000): PP Hybrid=0.9721")


if __name__ == "__main__":
    main()
