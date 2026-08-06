"""Submission-like CV: fold-wise full-retrain style prediction on validation folds."""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, ".")

from src.config import EVENT_COL, HORIZONS, TIME_COL
from src.evaluation import hybrid_score
from src.features import get_feature_set
from src.monotonic import submission_postprocess
from src.stacking import _train_predict_base
from src.train import _strat_labels, load_data, run_cv


def _diag(pred, floor_12):
    p12 = np.asarray(pred[12], dtype=float)
    d = {
        "share_floor_12": float(np.mean(p12 <= floor_12 + 1e-12)),
        "n_unique_12": int(np.unique(np.round(p12, 12)).size),
        "p10_12": float(np.percentile(p12, 10)),
        "p50_12": float(np.median(p12)),
        "p90_12": float(np.percentile(p12, 90)),
    }
    return d


def _print_eval(name, pred, y_time, y_event, floor_12):
    sc, det = hybrid_score(y_time, y_event, pred)
    d = _diag(pred, floor_12=floor_12)
    print(
        f"{name:24s} Hybrid={sc:.6f} CI={det['c_index']:.6f} WBrier={det['weighted_brier']:.6f} "
        f"B24={det['brier_24h']:.6f} B48={det['brier_48h']:.6f} B72={det['brier_72h']:.6f} "
        f"floor12={d['share_floor_12']:.3f} unique12={d['n_unique_12']:3d} "
        f"p10/p50/p90={d['p10_12']:.4f}/{d['p50_12']:.4f}/{d['p90_12']:.4f}"
    )
    return sc


def _normalize_weights(w):
    w = np.asarray(w, dtype=float)
    s = w.sum()
    if s <= 0:
        raise ValueError("At least one blend weight must be > 0.")
    return w / s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-level", default="medium")
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--n-repeats", type=int, default=1)
    p.add_argument("--cv-seed", type=int, default=42)
    p.add_argument("--strat-mode", choices=["event", "event_time"], default="event_time")
    p.add_argument("--seeds", default="42,123,456,789,2026")
    p.add_argument("--w-rsf", type=float, default=1.0)
    p.add_argument("--w-est", type=float, default=0.0)
    p.add_argument("--w-xgbcox", type=float, default=0.0)
    p.add_argument("--floor-12", type=float, default=1e-6)
    p.add_argument("--floor-24-48", type=float, default=1e-6)
    p.add_argument("--cap12-eps", type=float, default=1e-6)
    p.add_argument("--use-projection", action="store_true")
    p.add_argument("--compare-standard-cv", action="store_true")
    args = p.parse_args()

    retrain_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    w_rsf, w_est, w_xgb = _normalize_weights([args.w_rsf, args.w_est, args.w_xgbcox])

    train, _ = load_data(feature_level=args.feature_level)
    feature_cols = get_feature_set(train, level=args.feature_level)
    X = train[feature_cols].reset_index(drop=True)
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values
    n = len(train)

    print(
        f"[SubmissionLikeCV] rows={n} features={len(feature_cols)} splits={args.n_splits} repeats={args.n_repeats} "
        f"seeds={retrain_seeds} blend=(RSF={w_rsf:.2f},EST={w_est:.2f},XGBCox={w_xgb:.2f})"
    )

    strat = _strat_labels(y_time, y_event, mode=args.strat_mode, n_splits=args.n_splits)
    cv = RepeatedStratifiedKFold(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.cv_seed,
    )

    oof_raw = {h: np.zeros(n, dtype=float) for h in HORIZONS}
    oof_cnt = np.zeros(n, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, strat), start=1):
        X_tr = X.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        yt_tr, ye_tr = y_time[tr_idx], y_event[tr_idx]

        seed_preds = []
        for seed in retrain_seeds:
            base = _train_predict_base(X_tr, yt_tr, ye_tr, X_va, seed=seed)
            pred = {
                12: w_rsf * base["RSF"][12] + w_est * base["EST"][12] + w_xgb * base["XGBCox"][12],
                24: w_rsf * base["RSF"][24] + w_est * base["EST"][24] + w_xgb * base["XGBCox"][24],
                48: w_rsf * base["RSF"][48] + w_est * base["EST"][48] + w_xgb * base["XGBCox"][48],
                72: np.ones(len(va_idx), dtype=float),
            }
            seed_preds.append(pred)

        fold_pred = {
            h: np.mean([sp[h] for sp in seed_preds], axis=0)
            for h in HORIZONS
        }

        for h in HORIZONS:
            oof_raw[h][va_idx] += fold_pred[h]
        oof_cnt[va_idx] += 1.0

        if fold % args.n_splits == 0:
            print(f"  Repeat {fold // args.n_splits}/{args.n_repeats} done")

    mask = oof_cnt > 0
    for h in HORIZONS:
        oof_raw[h][mask] /= oof_cnt[mask]

    oof_post = submission_postprocess(
        oof_raw,
        floor_12=args.floor_12,
        floor_24_48=args.floor_24_48,
        use_projection=args.use_projection,
        cap_12_by_24_eps=args.cap12_eps,
    )

    print("\n=== Submission-like OOF ===")
    _print_eval("submission_like_raw", oof_raw, y_time, y_event, floor_12=args.floor_12)
    _print_eval("submission_like_post", oof_post, y_time, y_event, floor_12=args.floor_12)

    if args.compare_standard_cv:
        print("\n=== Standard CV Baseline (for gap diagnosis) ===")
        std_oof = run_cv(
            train,
            feature_cols,
            min_samples_leaf=5,
            include_boosting=False,
            strat_mode=args.strat_mode,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            random_state=args.cv_seed,
        )
        std_blend = {
            h: w_rsf * std_oof["RSF"][h] + w_est * std_oof["EST"][h]
            for h in HORIZONS
        }
        std_post = submission_postprocess(
            std_blend,
            floor_12=args.floor_12,
            floor_24_48=args.floor_24_48,
            use_projection=args.use_projection,
            cap_12_by_24_eps=args.cap12_eps,
        )
        s1 = _print_eval("standard_cv_raw", std_blend, y_time, y_event, floor_12=args.floor_12)
        s2 = _print_eval("standard_cv_post", std_post, y_time, y_event, floor_12=args.floor_12)

        t1, _ = hybrid_score(y_time, y_event, oof_post)
        print(f"\nDelta(post): submission_like - standard_cv = {t1 - s2:+.6f}")


if __name__ == "__main__":
    main()
