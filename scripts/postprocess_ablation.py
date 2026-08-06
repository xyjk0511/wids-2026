"""Ablation for postprocess strategies with detailed diagnostics (A/B)."""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from src.config import EVENT_COL, HORIZONS, TIME_COL
from src.evaluation import hybrid_score
from src.features import get_feature_set
from src.monotonic import submission_postprocess, submission_postprocess_full_mono
from src.train import load_data, run_cv


def _best_rsf_weight(oof, y_time, y_event):
    best_w, best_score = 1.0, -1e9
    for w in np.linspace(0.5, 1.0, 11):
        pred = {h: w * oof["RSF"][h] + (1.0 - w) * oof["EST"][h] for h in HORIZONS}
        sc, _ = hybrid_score(y_time, y_event, pred)
        if sc > best_score:
            best_score = sc
            best_w = float(w)
    return best_w, best_score


def _diag_12(pred, floor_ref_12):
    p12 = np.asarray(pred[12], dtype=float)
    n = len(p12)
    n_unique = int(np.unique(np.round(p12, 12)).size)
    ties = int(n - n_unique)
    share_floor = float(np.mean(p12 <= (floor_ref_12 + 1e-12)))
    return {
        "share_floor_12": share_floor,
        "n_unique_12": n_unique,
        "ties_12": ties,
        "p10_12": float(np.percentile(p12, 10)),
        "median_12": float(np.median(p12)),
        "p90_12": float(np.percentile(p12, 90)),
    }


def _projection_push_diag(pred_ref, pred_proj):
    out = {}
    for h in [24, 48]:
        d = np.asarray(pred_proj[h], dtype=float) - np.asarray(pred_ref[h], dtype=float)
        out[f"share_up_{h}"] = float(np.mean(d > 1e-12))
        out[f"mean_up_{h}"] = float(np.mean(np.clip(d, 0.0, None)))
        out[f"mean_abs_{h}"] = float(np.mean(np.abs(d)))
    return out


def _print_row(name, pred, y_time, y_event, floor_ref_12, push=None):
    sc, det = hybrid_score(y_time, y_event, pred)
    d12 = _diag_12(pred, floor_ref_12=floor_ref_12)
    row = (
        f"{name:18s} Hybrid={sc:.6f} CI={det['c_index']:.6f} WBrier={det['weighted_brier']:.6f} "
        f"B24={det['brier_24h']:.6f} B48={det['brier_48h']:.6f} "
        f"floor12={d12['share_floor_12']:.3f} unique12={d12['n_unique_12']:3d} ties12={d12['ties_12']:3d} "
        f"p10/p50/p90={d12['p10_12']:.4f}/{d12['median_12']:.4f}/{d12['p90_12']:.4f}"
    )
    if push is not None:
        row += (
            f" | push24(up={push['share_up_24']:.3f},mean+={push['mean_up_24']:.6f},abs={push['mean_abs_24']:.6f})"
            f" push48(up={push['share_up_48']:.3f},mean+={push['mean_up_48']:.6f},abs={push['mean_abs_48']:.6f})"
        )
    print(row)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-level", default="medium")
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--n-repeats", type=int, default=1)
    p.add_argument("--cv-seed", type=int, default=42)
    p.add_argument("--strat-mode", choices=["event", "event_time"], default="event_time")
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument("--with-boosting", action="store_true")
    p.add_argument("--floor-12", type=float, default=1e-6)
    p.add_argument("--floor-24-48", type=float, default=1e-6)
    p.add_argument("--cap12-eps", type=float, default=1e-6)
    args = p.parse_args()

    train, _ = load_data(feature_level=args.feature_level)
    feature_cols = get_feature_set(train, level=args.feature_level)
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    print(
        f"[CV] rows={len(train)} features={len(feature_cols)} "
        f"splits={args.n_splits} repeats={args.n_repeats} strat={args.strat_mode}"
    )
    oof = run_cv(
        train,
        feature_cols,
        min_samples_leaf=args.min_samples_leaf,
        include_boosting=args.with_boosting,
        strat_mode=args.strat_mode,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.cv_seed,
    )
    w_rsf, raw_best = _best_rsf_weight(oof, y_time, y_event)
    base = {h: w_rsf * oof["RSF"][h] + (1.0 - w_rsf) * oof["EST"][h] for h in HORIZONS}
    print(f"[Blend] RSF={w_rsf:.2f} EST={1-w_rsf:.2f} raw_best={raw_best:.6f}")

    v_split_old = submission_postprocess(
        base,
        floor_12=args.floor_12,
        floor_24_48=args.floor_24_48,
        use_projection=False,
        cap_12_by_24_eps=None,
    )
    v_split_new = submission_postprocess(
        base,
        floor_12=args.floor_12,
        floor_24_48=args.floor_24_48,
        use_projection=True,
        cap_12_by_24_eps=None,
    )
    v_split_cap = submission_postprocess(
        base,
        floor_12=args.floor_12,
        floor_24_48=args.floor_24_48,
        use_projection=True,
        cap_12_by_24_eps=args.cap12_eps,
    )

    v_full_old = submission_postprocess_full_mono(
        base,
        floor=args.floor_24_48,
        use_projection=False,
        cap_12_by_24_eps=None,
    )
    v_full_new = submission_postprocess_full_mono(
        base,
        floor=args.floor_24_48,
        use_projection=True,
        cap_12_by_24_eps=None,
    )

    print("\n=== A/B Diagnostics ===")
    _print_row("raw", base, y_time, y_event, floor_ref_12=args.floor_12)
    _print_row("split_old", v_split_old, y_time, y_event, floor_ref_12=args.floor_12)
    _print_row(
        "split_new",
        v_split_new,
        y_time,
        y_event,
        floor_ref_12=args.floor_12,
        push=_projection_push_diag(v_split_old, v_split_new),
    )
    _print_row(
        "split_new_cap12",
        v_split_cap,
        y_time,
        y_event,
        floor_ref_12=args.floor_12,
        push=_projection_push_diag(v_split_old, v_split_cap),
    )
    _print_row("full_old", v_full_old, y_time, y_event, floor_ref_12=args.floor_24_48)
    _print_row(
        "full_new",
        v_full_new,
        y_time,
        y_event,
        floor_ref_12=args.floor_24_48,
        push=_projection_push_diag(v_full_old, v_full_new),
    )


if __name__ == "__main__":
    main()
