"""Direction-1 experiment: risk-driven sigmoid head for 12h ranking."""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")

from src.config import EVENT_COL, HORIZONS, TIME_COL
from src.evaluation import c_index, hybrid_score
from src.features import get_feature_set
from src.models import EST, RSF
from src.monotonic import submission_postprocess
from src.train import _strat_labels, load_data


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def _diag12(pred, floor_ref=0.0):
    p12 = np.asarray(pred[12], dtype=float)
    n_unique = int(np.unique(np.round(p12, 12)).size)
    return {
        "share_floor_12": float(np.mean(p12 <= floor_ref + 1e-12)),
        "n_unique_12": n_unique,
        "ties_12": int(len(p12) - n_unique),
        "p10_12": float(np.percentile(p12, 10)),
        "p50_12": float(np.median(p12)),
        "p90_12": float(np.percentile(p12, 90)),
    }


def _print_eval(name, pred, y_time, y_event, risk=None, floor_ref=0.0):
    sc, det = hybrid_score(y_time, y_event, pred)
    d = _diag12(pred, floor_ref=floor_ref)
    msg = (
        f"{name:18s} Hybrid={sc:.6f} CI={det['c_index']:.6f} WBrier={det['weighted_brier']:.6f} "
        f"B24={det['brier_24h']:.6f} B48={det['brier_48h']:.6f} "
        f"floor12={d['share_floor_12']:.3f} unique12={d['n_unique_12']:3d} ties12={d['ties_12']:3d} "
        f"p10/p50/p90={d['p10_12']:.4f}/{d['p50_12']:.4f}/{d['p90_12']:.4f}"
    )
    if risk is not None:
        rho, _ = spearmanr(np.asarray(risk, dtype=float), np.asarray(pred[12], dtype=float))
        msg += f" rho(risk,p12)={rho:.4f}"
    print(msg)
    return sc, det


def _collect_oof_with_risk(
    train: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int,
    n_repeats: int,
    random_state: int,
    strat_mode: str,
    min_samples_leaf: int,
):
    X = train[feature_cols].reset_index(drop=True)
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values
    n = len(train)

    oof = {
        "RSF": {h: np.zeros(n, dtype=float) for h in HORIZONS},
        "EST": {h: np.zeros(n, dtype=float) for h in HORIZONS},
    }
    oof_risk = {
        "RSF": np.zeros(n, dtype=float),
        "EST": np.zeros(n, dtype=float),
    }
    counts = np.zeros(n, dtype=float)

    strat = _strat_labels(y_time, y_event, mode=strat_mode, n_splits=n_splits)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, strat), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        yt_tr, ye_tr = y_time[tr_idx], y_event[tr_idx]

        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(
            scaler.fit_transform(X_tr), columns=feature_cols, index=X_tr.index
        )
        X_va_s = pd.DataFrame(
            scaler.transform(X_va), columns=feature_cols, index=X_va.index
        )

        rsf = RSF(min_samples_leaf=min_samples_leaf, random_state=42)
        est = EST(min_samples_leaf=min_samples_leaf, random_state=42)
        rsf.fit(X_tr_s, yt_tr, ye_tr)
        est.fit(X_tr_s, yt_tr, ye_tr)

        pred_rsf = rsf.predict_proba(X_va_s)
        pred_est = est.predict_proba(X_va_s)
        risk_rsf = rsf.model.predict(X_va_s[rsf._cols].values)
        risk_est = est.model.predict(X_va_s[est._cols].values)

        for h in HORIZONS:
            oof["RSF"][h][va_idx] += pred_rsf[h]
            oof["EST"][h][va_idx] += pred_est[h]
        oof_risk["RSF"][va_idx] += np.asarray(risk_rsf, dtype=float)
        oof_risk["EST"][va_idx] += np.asarray(risk_est, dtype=float)
        counts[va_idx] += 1.0

        if fold % n_splits == 0:
            print(f"  Repeat {fold // n_splits}/{n_repeats} done")

    mask = counts > 0
    for name in ["RSF", "EST"]:
        for h in HORIZONS:
            oof[name][h][mask] /= counts[mask]
        oof_risk[name][mask] /= counts[mask]
    return oof, oof_risk, y_time, y_event


def _apply_risk_head(p24, risk, a, b, eps=1e-6):
    """p12 = p24 * sigmoid(a * (z - b)), then enforce p12 <= p24 - eps."""
    z = np.asarray(risk, dtype=float)
    p24 = np.asarray(p24, dtype=float)
    raw = p24 * _sigmoid(float(a) * (z - float(b)))
    capped = np.minimum(raw, p24 - float(eps))
    return np.clip(capped, 0.0, 0.99)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-level", default="medium")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument("--cv-seed", type=int, default=42)
    p.add_argument("--strat-mode", choices=["event", "event_time"], default="event_time")
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument("--floor-12", type=float, default=0.0)
    p.add_argument("--floor-24-48", type=float, default=1e-6)
    p.add_argument("--eps-cap", type=float, default=1e-6)
    p.add_argument("--max-a", type=float, default=12.0)
    p.add_argument("--lambda-floor", type=float, default=0.0)
    p.add_argument("--lambda-ties", type=float, default=0.0)
    args = p.parse_args()

    train, _ = load_data(feature_level=args.feature_level)
    feature_cols = get_feature_set(train, level=args.feature_level)
    print(
        f"[RiskHead12] rows={len(train)} features={len(feature_cols)} "
        f"splits={args.n_splits} repeats={args.n_repeats} strat={args.strat_mode}"
    )

    oof, oof_risk, y_time, y_event = _collect_oof_with_risk(
        train=train,
        feature_cols=feature_cols,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.cv_seed,
        strat_mode=args.strat_mode,
        min_samples_leaf=args.min_samples_leaf,
    )

    # Blend weight search (RSF/EST) on raw predictions.
    best_w, best_raw = 1.0, -1e9
    for w in np.linspace(0.5, 1.0, 11):
        pred = {h: w * oof["RSF"][h] + (1.0 - w) * oof["EST"][h] for h in HORIZONS}
        sc, _ = hybrid_score(y_time, y_event, pred)
        if sc > best_raw:
            best_raw = sc
            best_w = float(w)

    base = {h: best_w * oof["RSF"][h] + (1.0 - best_w) * oof["EST"][h] for h in HORIZONS}
    risk = best_w * oof_risk["RSF"] + (1.0 - best_w) * oof_risk["EST"]
    risk = (risk - risk.mean()) / (risk.std() + 1e-12)
    print(f"[Blend] RSF={best_w:.2f} EST={1-best_w:.2f} raw_best={best_raw:.6f}")

    # Baseline postprocess.
    base_post = submission_postprocess(
        base,
        floor_12=args.floor_12,
        floor_24_48=args.floor_24_48,
        use_projection=True,
        cap_12_by_24_eps=None,
    )

    # Grid-search (a, b) to maximize CI of 12h.
    b_grid = np.quantile(risk, [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    a_grid_full = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    a_grid = a_grid_full[a_grid_full <= float(args.max_a) + 1e-12]
    if a_grid.size == 0:
        raise ValueError("No valid a_grid values after --max-a filtering.")

    n = len(y_time)
    best_ci = -1.0
    best_obj = -1e18
    best_ab = None
    best_head = None
    best_diag = None
    for a in a_grid:
        for b in b_grid:
            p12 = _apply_risk_head(base[24], risk, a=a, b=b, eps=args.eps_cap)
            pred = dict(base)
            pred[12] = p12
            pred_post = submission_postprocess(
                pred,
                floor_12=args.floor_12,
                floor_24_48=args.floor_24_48,
                use_projection=True,
                cap_12_by_24_eps=None,
            )
            ci = c_index(y_time, y_event, pred_post[12])
            d = _diag12(pred_post, floor_ref=args.floor_12)
            share_floor = d["share_floor_12"]
            frac_ties = 1.0 - (d["n_unique_12"] / max(n, 1))
            obj = ci - float(args.lambda_floor) * share_floor - float(args.lambda_ties) * frac_ties
            if obj > best_obj:
                best_obj = obj
                best_ci = ci
                best_ab = (float(a), float(b))
                best_head = pred_post
                best_diag = d

    print("\n=== Direction-1 Result ===")
    _print_eval("baseline_post", base_post, y_time, y_event, risk=risk, floor_ref=args.floor_12)
    _print_eval("risk_head_post", best_head, y_time, y_event, risk=risk, floor_ref=args.floor_12)
    print(
        f"best_ab: a={best_ab[0]:.3f}, b={best_ab[1]:.6f}, eps_cap={args.eps_cap:g}, "
        f"objective={best_obj:.6f}, ci={best_ci:.6f}, "
        f"share_floor12={best_diag['share_floor_12']:.3f}, unique12={best_diag['n_unique_12']}"
    )


if __name__ == "__main__":
    main()
