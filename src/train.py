"""Main training script: RSF single model + postprocessing + submission."""

import argparse
from collections import Counter
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
from src.models import RSF, EST, MultiHorizonLGBM, MultiHorizonCatBoost
from src.monotonic import (
    enforce_monotonicity, submission_postprocess, submission_postprocess_full_mono,
)
from src.stacking import train_horizon_heads, predict_horizon_heads

warnings.filterwarnings("ignore")


FEATURE_LEVEL = "medium"
DEFAULT_STRAT_MODE = "event_time"
POST_FLOOR = 1e-6


def load_data(feature_level=FEATURE_LEVEL):
    """Load and prepare train/test data."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if feature_level in ("v96624", "v96624_plus"):
        train = add_engineered(train)
        test = add_engineered(test)
    else:
        train = add_engineered(remove_redundant(train))
        test = add_engineered(remove_redundant(test))

    return train, test


def _merge_rare_labels(labels, min_count):
    """Merge rare composite strat labels to keep StratifiedKFold valid."""
    labels = np.asarray(labels, dtype=int).copy()
    changed = True
    while changed:
        changed = False
        counts = Counter(labels.tolist())
        for lab, cnt in sorted(counts.items(), key=lambda x: x[1]):
            if cnt == 0 or cnt >= min_count:
                continue
            is_event = lab >= 10
            base = 10 if is_event else 0
            b = lab - base

            candidates = []
            for bb in range(4):
                cand = base + bb
                c_cnt = counts.get(cand, 0)
                if cand == lab or c_cnt == 0:
                    continue
                candidates.append((cand, abs(bb - b), -c_cnt))
            if not candidates:
                continue

            target = sorted(candidates, key=lambda x: (x[1], x[2]))[0][0]
            labels[labels == lab] = target
            changed = True
    return labels


def _strat_labels(y_time, y_event, mode=DEFAULT_STRAT_MODE, n_splits=N_SPLITS):
    """Build stratification labels.

    mode='event' keeps legacy 2-class split.
    mode='event_time' uses event/censor + horizon-aware time bins.
    """
    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event, dtype=int)

    if mode == "event":
        return y_event.astype(int)

    if mode != "event_time":
        raise ValueError(f"Unknown strat mode: {mode}")

    # bins: <=12, (12,24], (24,48], >48
    tbin = np.digitize(y_time, bins=[12.0, 24.0, 48.0], right=True)
    labels = np.where(y_event == 1, 10 + tbin, tbin).astype(int)
    labels = _merge_rare_labels(labels, min_count=n_splits)
    return labels


def _format_label_counts(labels):
    c = Counter(np.asarray(labels, dtype=int).tolist())
    return " ".join(f"{k}:{v}" for k, v in sorted(c.items(), key=lambda x: x[0]))


def run_cv(
    train,
    feature_cols,
    min_samples_leaf=5,
    include_boosting=False,
    rsf_kwargs=None,
    strat_mode=DEFAULT_STRAT_MODE,
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
    collect_risk=False,
):
    """Run repeated stratified K-fold CV with RSF+EST, optionally LGBM+CatBoost.

    If collect_risk=True, also collects predict_risk() OOF for RSF and EST.
    """
    X = train[feature_cols]
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    n = len(train)
    model_names = ["RSF", "EST"]
    if include_boosting:
        model_names += ["LGBM", "CatBoost"]
    oof = {name: {h: np.zeros(n) for h in HORIZONS} for name in model_names}
    oof_counts = np.zeros(n)
    risk_oof = None
    if collect_risk:
        risk_oof = {name: np.zeros(n) for name in ["RSF", "EST"]}

    strat = _strat_labels(y_time, y_event, mode=strat_mode, n_splits=n_splits)
    print(f"  Strat mode={strat_mode} labels: {_format_label_counts(strat)}")
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats,
        random_state=random_state,
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

        rsf_cfg = {"min_samples_leaf": min_samples_leaf}
        if rsf_kwargs:
            rsf_cfg.update(rsf_kwargs)

        est_cfg = dict(rsf_cfg)
        models = {
            "RSF": RSF(**rsf_cfg),
            "EST": EST(**est_cfg),
        }
        if include_boosting:
            models["LGBM"] = MultiHorizonLGBM()
            models["CatBoost"] = MultiHorizonCatBoost()

        for name, model in models.items():
            model.fit(X_tr_s, yt_tr, ye_tr)
            preds = model.predict_proba(X_va_s)
            for h in HORIZONS:
                oof[name][h][va_idx] += preds[h]
            if collect_risk and name in ("RSF", "EST"):
                risk_oof[name][va_idx] += model.predict_risk(X_va_s)

        oof_counts[va_idx] += 1

        if fold_idx % n_splits == 0:
            rep = fold_idx // n_splits
            print(f"  Repeat {rep}/{n_repeats} done")

    # Average over repeats
    mask = oof_counts > 0
    for name in model_names:
        for h in HORIZONS:
            oof[name][h][mask] /= oof_counts[mask]
    if collect_risk:
        for name in ["RSF", "EST"]:
            risk_oof[name][mask] /= oof_counts[mask]
        return oof, risk_oof

    return oof


def run_cv_decoupled(
    train,
    feature_cols,
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
):
    """Decoupled per-horizon strategy:
    12h: RSF+EST predict_risk() → rankdata → fold-internal Platt
    24h/48h: 4-model per-horizon weight grid search (Brier-optimal)
    72h: 1.0
    """
    from scipy.stats import rankdata
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from src.evaluation import horizon_brier_score, c_index
    from src.labels import build_horizon_labels

    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values
    n = len(train)

    # Run 4-model CV with risk score collection
    print("\n=== Decoupled: Running 4-model CV with risk scores ===")
    oof, risk_oof = run_cv(
        train, feature_cols, include_boosting=True,
        n_splits=n_splits, n_repeats=n_repeats,
        random_state=random_state, collect_risk=True,
    )

    # --- 12h: RSF+EST risk → rank → fold-internal Platt ---
    risk_12h = 0.5 * risk_oof["RSF"] + 0.5 * risk_oof["EST"]
    ranked = rankdata(risk_12h) / (len(risk_12h) + 1)

    labels_12, elig_12 = build_horizon_labels(y_time, y_event, 12)
    platt_oof = np.full(n, 0.5)
    strat = _strat_labels(y_time, y_event)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state + 999)

    for tr_idx, va_idx in skf.split(train[feature_cols], strat):
        tr_elig = elig_12[tr_idx]
        if tr_elig.sum() < 5:
            continue
        platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        platt.fit(ranked[tr_idx[tr_elig]].reshape(-1, 1),
                  labels_12[tr_idx[tr_elig]])
        platt_oof[va_idx] = platt.predict_proba(
            ranked[va_idx].reshape(-1, 1))[:, 1]

    ci_platt = c_index(y_time, y_event, platt_oof)
    # Also try RSF proba directly (often better CI)
    rsf_12h = oof["RSF"][12]
    ci_rsf = c_index(y_time, y_event, rsf_12h)
    # Also try 4-model CI-optimal weight search for 12h
    best_ci = -1.0
    best_12h_w = None
    for ws in range(0, 11):
        w_surv = ws / 10.0
        for wr in range(0, 11):
            w_rsf = wr / 10.0
            for wl in range(0, 11):
                w_lgbm = wl / 10.0
                weights = {
                    "RSF": w_surv * w_rsf,
                    "EST": w_surv * (1.0 - w_rsf),
                    "LGBM": (1.0 - w_surv) * w_lgbm,
                    "CatBoost": (1.0 - w_surv) * (1.0 - w_lgbm),
                }
                blend = sum(weights[m] * oof[m][12] for m in weights)
                ci_val = c_index(y_time, y_event, blend)
                if ci_val > best_ci:
                    best_ci = ci_val
                    best_12h_w = dict(weights)
    best_12h_blend = sum(best_12h_w[m] * oof[m][12] for m in best_12h_w)

    # Pick best 12h strategy
    candidates = [
        ("Platt(risk)", ci_platt, platt_oof),
        ("RSF_proba", ci_rsf, rsf_12h),
        ("WeightSearch", best_ci, best_12h_blend),
    ]
    best_name, best_ci_val, best_12h = max(candidates, key=lambda x: x[1])
    for name, ci_val, _ in candidates:
        tag = " <-- BEST" if name == best_name else ""
        print(f"  12h CI: {name}={ci_val:.6f}{tag}")
    use_12h_weights = best_name == "WeightSearch"

    # --- 24h/48h: per-horizon weight grid search ---
    all_models = ["RSF", "EST", "LGBM", "CatBoost"]
    best_weights = {}
    best_blend = {}

    for h in [24, 48]:
        best_brier = 1e9
        best_w = None
        for ws in range(0, 11):
            w_surv = ws / 10.0
            w_boost = 1.0 - w_surv
            for wr in range(0, 11):
                w_rsf = wr / 10.0
                for wl in range(0, 11):
                    w_lgbm = wl / 10.0
                    weights = {
                        "RSF": w_surv * w_rsf,
                        "EST": w_surv * (1.0 - w_rsf),
                        "LGBM": w_boost * w_lgbm,
                        "CatBoost": w_boost * (1.0 - w_lgbm),
                    }
                    blend_h = sum(weights[m] * oof[m][h] for m in weights)
                    brier = horizon_brier_score(y_time, y_event, blend_h, h)
                    if brier < best_brier:
                        best_brier = brier
                        best_w = dict(weights)

        best_weights[h] = best_w
        best_blend[h] = sum(best_w[m] * oof[m][h] for m in best_w)
        w_str = "  ".join(f"{m}={best_w[m]:.2f}" for m in all_models)
        print(f"  {h}h: Brier={best_brier:.6f}  {w_str}")

    # Store 12h weights if weight-search won
    if use_12h_weights:
        best_weights[12] = best_12h_w

    # Assemble decoupled predictions
    decoupled = {12: best_12h, 24: best_blend[24],
                 48: best_blend[48], 72: np.ones(n)}

    # Comparison
    print("\n=== Decoupled vs Baseline ===")
    surv_blend = {h: 0.5 * oof["RSF"][h] + 0.5 * oof["EST"][h]
                  for h in HORIZONS}
    _print_score("Baseline(RSF+EST)", surv_blend, y_time, y_event)
    _print_score("Decoupled", decoupled, y_time, y_event)

    return decoupled, oof, best_weights, risk_oof, best_name


def leaf_search(
    train,
    feature_cols,
    strat_mode=DEFAULT_STRAT_MODE,
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
):
    """Grid search min_samples_leaf for RSF/EST (Experiment A)."""
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values
    candidates = [5, 10, 15, 20, 25, 30]
    best_leaf, best_score = 5, -1.0

    print("\n=== Experiment A: min_samples_leaf Grid Search ===")
    for leaf in candidates:
        print(f"\n--- min_samples_leaf={leaf} ---")
        oof = run_cv(
            train,
            feature_cols,
            min_samples_leaf=leaf,
            strat_mode=strat_mode,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        blended = {h: 0.5 * oof["RSF"][h] + 0.5 * oof["EST"][h] for h in HORIZONS}
        score, det = hybrid_score(y_time, y_event, blended)
        p12 = blended[12]
        print(f"  Hybrid={score:.4f}  CI={det['c_index']:.4f}  WBrier={det['weighted_brier']:.4f}")
        print(f"  12h dist: min={p12.min():.4f} p25={np.percentile(p12,25):.4f} "
              f"median={np.median(p12):.4f} p75={np.percentile(p12,75):.4f} max={p12.max():.4f}")
        n_zero = (p12 < 0.001).sum()
        print(f"  12h near-zero (<0.001): {n_zero}/{len(p12)} ({n_zero/len(p12)*100:.1f}%)")
        if score > best_score:
            best_score = score
            best_leaf = leaf

    print(f"\n  Best min_samples_leaf={best_leaf} (Hybrid={best_score:.4f})")
    return best_leaf


def per_horizon_weight_search(oof, y_time, y_event):
    """Search optimal per-horizon weights across all available models (Experiment B).

    For each horizon, grid-search weight combinations to minimize Brier score
    while preserving CI (12h only uses survival models for ranking).
    """
    from src.evaluation import horizon_brier_score, c_index

    surv_models = [n for n in ["RSF", "EST"] if n in oof]
    boost_models = [n for n in ["LGBM", "CatBoost"] if n in oof]
    all_models = surv_models + boost_models

    if not boost_models:
        print("  [WARN] No boosting models in OOF, skipping per-horizon search")
        return None, None

    print("\n=== Per-Horizon Weight Search ===")
    best_weights = {}
    best_blend = {}

    for h in HORIZONS:
        if h == 72:
            # 72h is always 1.0 after postprocess, weights don't matter
            best_weights[h] = {n: 1.0 / len(all_models) for n in all_models}
            best_blend[h] = np.mean([oof[n][h] for n in all_models], axis=0)
            continue

        # 12h: optimize CI (ranking); 24h/48h: optimize Brier (calibration)
        use_ci = (h == 12)
        best_metric = -1.0 if use_ci else 1e9
        best_w_h = None

        # Grid search: pairs of (survival_blend, boost_blend) with 0.1 step
        for w_surv_int in range(0, 11):
            w_surv = w_surv_int / 10.0
            w_boost = 1.0 - w_surv

            for w_rsf_int in range(0, 11):
                w_rsf = w_rsf_int / 10.0
                w_est = 1.0 - w_rsf

                for w_lgbm_int in range(0, 11):
                    w_lgbm = w_lgbm_int / 10.0
                    w_cat = 1.0 - w_lgbm

                    weights = {}
                    if "RSF" in oof:
                        weights["RSF"] = w_surv * w_rsf
                    if "EST" in oof:
                        weights["EST"] = w_surv * w_est
                    if "LGBM" in oof:
                        weights["LGBM"] = w_boost * w_lgbm
                    if "CatBoost" in oof:
                        weights["CatBoost"] = w_boost * w_cat

                    blend_h = sum(weights[n] * oof[n][h] for n in weights)

                    if use_ci:
                        metric = c_index(y_time, y_event, blend_h)
                        if metric > best_metric:
                            best_metric = metric
                            best_w_h = dict(weights)
                    else:
                        metric = horizon_brier_score(y_time, y_event, blend_h, h)
                        if metric < best_metric:
                            best_metric = metric
                            best_w_h = dict(weights)

        best_weights[h] = best_w_h
        best_blend[h] = sum(best_w_h[n] * oof[n][h] for n in best_w_h)
        w_str = "  ".join(f"{n}={best_w_h[n]:.1f}" for n in all_models)
        metric_name = "CI" if use_ci else "Brier"
        print(f"  {h}h: {metric_name}={best_metric:.6f}  {w_str}")

    # Evaluate full blend
    score, det = hybrid_score(y_time, y_event, best_blend)
    print(f"\n  Per-horizon blend: Hybrid={score:.4f}  CI={det['c_index']:.4f}  WBrier={det['weighted_brier']:.4f}")

    return best_weights, best_blend


def _print_score(label, prob_dict, y_time, y_event):
    """Print hybrid/CI/WBrier for a single prob_dict."""
    score, det = hybrid_score(y_time, y_event, prob_dict)
    brier_parts = "  ".join(f"B{h}h={det.get(f'brier_{h}h', 0):.4f}" for h in [24, 48, 72])
    print(f"  {label:20s}  Hybrid={score:.4f}  CI={det['c_index']:.4f}  WBrier={det['weighted_brier']:.4f}  ({brier_parts})")
    return score, det


def print_oof_scores(oof_per_model, y_time, y_event):
    """Print per-model OOF scores and search optimal RSF:EST weight ratio."""
    print("\n=== OOF Scores (per-model) ===")
    for name in oof_per_model:
        _print_score(name, oof_per_model[name], y_time, y_event)

    # Weight search: RSF weight from 0.50 to 1.00, step 0.05
    print("\n=== Weight Search (RSF:EST) ===")
    best_w, best_score = 1.0, -1.0
    for w_int in range(50, 105, 5):
        w_rsf = w_int / 100.0
        w_est = 1.0 - w_rsf
        blended = {h: w_rsf * oof_per_model["RSF"][h] + w_est * oof_per_model["EST"][h]
                   for h in HORIZONS}
        score, _ = hybrid_score(y_time, y_event, blended)
        tag = ""
        if score > best_score:
            best_score = score
            best_w = w_rsf
            tag = " <-- best"
        print(f"  RSF={w_rsf:.2f} EST={w_est:.2f}  Hybrid={score:.4f}{tag}")

    print(f"\n  Optimal: RSF={best_w:.2f} EST={1-best_w:.2f} (Hybrid={best_score:.4f})")

    # Show detailed scores for optimal blend
    optimal_blend = {h: best_w * oof_per_model["RSF"][h] + (1-best_w) * oof_per_model["EST"][h]
                     for h in HORIZONS}
    _print_score(f"Blend(RSF={best_w:.2f})", optimal_blend, y_time, y_event)

    pp = submission_postprocess(optimal_blend)
    _print_score("Blend (postproc)", pp, y_time, y_event)

    return best_w


def _validate_and_save(
    test_preds,
    sub_path=SUBMISSION_PATH,
    postprocess_fn=None,
    floor_for_diag=POST_FLOOR,
):
    """Postprocess, validate, save submission, and print diagnostics."""
    pp_fn = postprocess_fn or submission_postprocess
    test_preds = pp_fn(test_preds)
    test_preds = enforce_monotonicity(test_preds)

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = test_preds[h]

    # Validation checks
    h72_ok = (sub[PROB_COLS[HORIZONS.index(72)]] == 1.0).all()
    mono_ok = all(
        all(sub[PROB_COLS[j]].iloc[i] >= sub[PROB_COLS[j-1]].iloc[i] - 1e-9
            for j in range(1, len(PROB_COLS)))
        for i in range(len(sub))
    )
    print(f"  72h all-ones: {'PASS' if h72_ok else 'FAIL'}  Monotonicity: {'PASS' if mono_ok else 'FAIL'}  Shape: {sub.shape}")

    # Probability distribution
    print("\n=== Probability Distribution (vs 0.96624 target) ===")
    for col in PROB_COLS:
        vals = sub[col]
        print(f"  {col}: min={vals.min():.4f} median={vals.median():.4f} max={vals.max():.4f}")
    print("  Target 12h: min~0.036 median~0.15 max~0.99")

    # Floor diagnostics
    print("\n=== Floor Diagnostics ===")
    for h in [12, 24, 48]:
        col = f"prob_{h}h"
        vals = sub[col]
        n_exact = np.isclose(vals, floor_for_diag, atol=1e-12).sum()
        n_near_zero = (vals <= 1e-5).sum()
        print(f"  {col}: near_floor({floor_for_diag:g})={n_exact}/{len(vals)} "
              f"near_zero={n_near_zero}/{len(vals)} "
              f"p10={vals.quantile(.1):.4f} p25={vals.quantile(.25):.4f} "
              f"median={vals.median():.4f} p75={vals.quantile(.75):.4f}")

    sub.to_csv(sub_path, index=False)
    print(f"\n  Submission saved to {sub_path}")
    print(f"  Preview:\n{sub.head()}")

    # Spearman vs reference
    ref_path = "submission 0.96624.csv"
    try:
        ref = pd.read_csv(ref_path)
        from scipy.stats import spearmanr
        print("\n=== Spearman Rank Correlation vs 0.96624 ===")
        for col in PROB_COLS[:-1]:
            sr, pval = spearmanr(sub[col], ref[col])
            print(f"  {col}: rho={sr:.4f}  p={pval:.2e}")
    except FileNotFoundError:
        print(f"\n  [WARN] Reference '{ref_path}' not found, skipping Spearman")

    return sub


def _build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-level", default=FEATURE_LEVEL)
    p.add_argument("--strat-mode", choices=["event", "event_time"], default=DEFAULT_STRAT_MODE)
    p.add_argument("--cv-seed", type=int, default=RANDOM_STATE)
    p.add_argument("--n-splits", type=int, default=N_SPLITS)
    p.add_argument("--n-repeats", type=int, default=N_REPEATS)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument("--run-leaf-search", action="store_true")
    p.add_argument("--with-boosting", action="store_true")
    p.add_argument("--max-rsf-weight", type=float, default=1.0)
    p.add_argument("--head-model", choices=["xgb", "logit"], default="xgb")
    p.add_argument(
        "--base-feature-mode",
        choices=["raw", "rank_logit", "raw_rank_logit"],
        default="raw",
    )
    p.add_argument(
        "--calibration-mode",
        choices=["auto", "iso24_48", "none"],
        default="auto",
    )
    p.add_argument(
        "--head-base-only",
        action="store_true",
        help="Use only base-model derived features for horizon heads.",
    )
    p.add_argument(
        "--disable-head",
        action="store_true",
        help="Bypass horizon heads and submit survival baseline blend.",
    )
    p.add_argument("--no-boosting", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--decoupled",
        action="store_true",
        help="Decoupled per-horizon strategy: risk-Platt for 12h, weight-search for 24h/48h.",
    )
    return p


def _run_decoupled_path(args, train, test, feature_cols, y_time, y_event, post_floor):
    """Decoupled per-horizon pipeline: Steps 2-4 of P1 plan."""
    from functools import partial
    from scipy.stats import rankdata
    from sklearn.linear_model import LogisticRegression
    from src.labels import build_horizon_labels

    # Step 2: OOF decoupled evaluation
    decoupled, oof, best_weights, risk_oof, best_name = run_cv_decoupled(
        train, feature_cols,
        n_splits=args.n_splits, n_repeats=args.n_repeats,
        random_state=args.cv_seed,
    )
    print(f"  12h strategy selected: {best_name}")

    # Step 3: Full retrain 4 models for test prediction
    RETRAIN_SEEDS = [42, 123, 456, 789, 2026]
    X_train = train[feature_cols]
    X_test = test[feature_cols]

    print(f"\n=== Full retrain 4 models ({len(RETRAIN_SEEDS)} seeds) ===")
    test_preds_all = {m: {h: [] for h in HORIZONS} for m in ["RSF", "EST", "LGBM", "CatBoost"]}
    test_risk_all = {m: [] for m in ["RSF", "EST"]}

    for seed in RETRAIN_SEEDS:
        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(scaler.fit_transform(X_train),
                              columns=feature_cols, index=X_train.index)
        X_te_s = pd.DataFrame(scaler.transform(X_test),
                              columns=feature_cols, index=X_test.index)

        rsf = RSF(min_samples_leaf=5, random_state=seed)
        est = EST(min_samples_leaf=5, random_state=seed)
        lgbm = MultiHorizonLGBM(random_state=seed)
        cat = MultiHorizonCatBoost(random_state=seed)

        for name, model in [("RSF", rsf), ("EST", est),
                            ("LGBM", lgbm), ("CatBoost", cat)]:
            model.fit(X_tr_s, y_time, y_event)
            preds = model.predict_proba(X_te_s)
            for h in HORIZONS:
                test_preds_all[name][h].append(preds[h])
            if name in ("RSF", "EST"):
                test_risk_all[name].append(model.predict_risk(X_te_s))

        print(f"  Seed {seed}: 4 models done")

    # Average across seeds
    for m in test_preds_all:
        for h in HORIZONS:
            test_preds_all[m][h] = np.mean(test_preds_all[m][h], axis=0)
    for m in test_risk_all:
        test_risk_all[m] = np.mean(test_risk_all[m], axis=0)

    # 12h test prediction: strategy depends on OOF winner
    if best_name == "Platt(risk)":
        train_risk = 0.5 * risk_oof["RSF"] + 0.5 * risk_oof["EST"]
        train_ranked = rankdata(train_risk) / (len(train_risk) + 1)
        labels_12, elig_12 = build_horizon_labels(y_time, y_event, 12)
        platt_full = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        platt_full.fit(train_ranked[elig_12].reshape(-1, 1), labels_12[elig_12])
        test_risk = 0.5 * test_risk_all["RSF"] + 0.5 * test_risk_all["EST"]
        combined = np.concatenate([train_risk, test_risk])
        combined_ranks = rankdata(combined) / (len(combined) + 1)
        test_ranked = combined_ranks[len(train_risk):]
        test_12h = platt_full.predict_proba(test_ranked.reshape(-1, 1))[:, 1]
    elif best_name == "WeightSearch":
        w = best_weights[12]
        test_12h = sum(w[m] * test_preds_all[m][12] for m in w)
    else:  # RSF_proba
        test_12h = test_preds_all["RSF"][12]
    print(f"  12h test strategy: {best_name}")

    # 24h/48h test: weighted blend using OOF-optimal weights
    test_preds = {12: test_12h, 72: np.ones(len(test))}
    for h in [24, 48]:
        w = best_weights[h]
        test_preds[h] = sum(w[m] * test_preds_all[m][h] for m in w)

    # Generate submission
    print("\n=== Generating decoupled submission ===")
    postprocess_fn = partial(submission_postprocess, floor=post_floor)
    _validate_and_save(test_preds, postprocess_fn=postprocess_fn,
                       floor_for_diag=post_floor)


def main():
    args = _build_argparser().parse_args()
    post_floor = POST_FLOOR

    print("=== Loading data ===")
    train, test = load_data(feature_level=args.feature_level)
    feature_cols = get_feature_set(train, level=args.feature_level)
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(
        f"  CV config: strat={args.strat_mode} seed={args.cv_seed} "
        f"n_splits={args.n_splits} n_repeats={args.n_repeats}"
    )
    print(
        f"  Head config: model={args.head_model} base_features={args.base_feature_mode} "
        f"use_orig_features={not args.head_base_only} calib={args.calibration_mode} "
        f"disable_head={args.disable_head}"
    )

    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    # ================================================================
    # Decoupled path: separate strategy per horizon
    # ================================================================
    if args.decoupled:
        _run_decoupled_path(args, train, test, feature_cols, y_time, y_event, post_floor)
        return

    # ================================================================
    # Experiment A: min_samples_leaf grid search (RSF+EST only)
    # ================================================================
    if args.run_leaf_search:
        best_leaf = leaf_search(
            train,
            feature_cols,
            strat_mode=args.strat_mode,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            random_state=args.cv_seed,
        )
    else:
        best_leaf = args.min_samples_leaf
        print(f"\n=== Experiment A skipped (using fixed min_samples_leaf={best_leaf}) ===")

    # ================================================================
    # Experiment B: Per-horizon blend with all 4 models
    # Uses best_leaf from Experiment A (= Experiment C if A improved)
    # ================================================================
    include_boosting = args.with_boosting and not args.no_boosting
    print(
        f"\n=== Running CV (RSF+EST{' +LGBM+CatBoost' if include_boosting else ''}, "
        f"min_samples_leaf={best_leaf}) ==="
    )
    oof = run_cv(
        train,
        feature_cols,
        min_samples_leaf=best_leaf,
        include_boosting=include_boosting,
        strat_mode=args.strat_mode,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.cv_seed,
    )

    # Print per-model scores
    print("\n=== OOF Scores (per-model) ===")
    for name in oof:
        _print_score(name, oof[name], y_time, y_event)

    # RSF:EST weight search (for comparison with previous baseline)
    print("\n=== RSF:EST Weight Search (survival-only baseline) ===")
    best_surv_w, best_surv_score = 1.0, -1.0
    for w_int in range(50, 105, 5):
        w_rsf = w_int / 100.0
        blended = {h: w_rsf * oof["RSF"][h] + (1-w_rsf) * oof["EST"][h] for h in HORIZONS}
        score, _ = hybrid_score(y_time, y_event, blended)
        tag = " <-- best" if score > best_surv_score else ""
        if score > best_surv_score:
            best_surv_score = score
            best_surv_w = w_rsf
        print(f"  RSF={w_rsf:.2f} EST={1-w_rsf:.2f}  Hybrid={score:.4f}{tag}")

    surv_blend = {h: best_surv_w * oof["RSF"][h] + (1-best_surv_w) * oof["EST"][h] for h in HORIZONS}
    _print_score(f"SurvBlend(RSF={best_surv_w:.2f})", surv_blend, y_time, y_event)
    pp_surv = submission_postprocess(surv_blend)
    _print_score("SurvBlend (postproc)", pp_surv, y_time, y_event)

    from functools import partial
    heads = None
    if args.disable_head:
        print("\n=== Phase 2 skipped: --disable-head enabled (survival baseline path) ===")
        postprocess_fn = partial(submission_postprocess, floor=post_floor)
    else:
        # ================================================================
        # Phase 2: Horizon heads (stacking, replaces per_horizon_weight_search)
        # ================================================================
        print("\n=== Phase 2: Training Horizon Heads (stacking) ===")
        head_oof, heads = train_horizon_heads(
            X_features=train[feature_cols],
            base_oof=oof,
            y_time=y_time,
            y_event=y_event,
            n_splits=args.n_splits,
            n_repeats=min(args.n_repeats, 3),
            n_inner_splits=3,
            random_state=args.cv_seed + 1000,
            head_model=args.head_model,
            base_feature_mode=args.base_feature_mode,
            use_orig_features=not args.head_base_only,
            calibration_mode=args.calibration_mode,
        )

        # Evaluate head OOF
        head_prob_dict = {h: head_oof[h] for h in [12, 24, 48]}
        head_prob_dict[72] = np.ones(len(train))
        _print_score("Head OOF (raw)", head_prob_dict, y_time, y_event)

        # ================================================================
        # Phase 3: A/B postprocess comparison (on OOF, decision locked here)
        # ================================================================
        print("\n=== A/B Postprocess Comparison (OOF) ===")

        pp_a = submission_postprocess(head_prob_dict, floor=post_floor)
        pp_b = submission_postprocess_full_mono(head_prob_dict, floor=post_floor)

        score_a, det_a = hybrid_score(y_time, y_event, pp_a)
        score_b, det_b = hybrid_score(y_time, y_event, pp_b)
        _print_score("PostProc-A (split)", pp_a, y_time, y_event)
        _print_score("PostProc-B (mono)", pp_b, y_time, y_event)

        use_full_mono = score_b > score_a
        postprocess_fn = partial(
            submission_postprocess_full_mono if use_full_mono else submission_postprocess,
            floor=post_floor,
        )
        print(f"  Selected: {'B (full mono)' if use_full_mono else 'A (split chain)'}")

        # Floor diagnostics on OOF
        chosen_pp = pp_b if use_full_mono else pp_a
        print("\n=== OOF Floor Diagnostics ===")
        for h in [12, 24, 48]:
            vals = chosen_pp[h]
            n_exact = np.isclose(vals, post_floor, atol=1e-12).sum()
            n_near_zero = (vals <= 1e-5).sum()
            print(f"  {h}h: near_floor({post_floor:g})={n_exact}/{len(vals)} "
                  f"near_zero={n_near_zero}/{len(vals)} "
                  f"p10={np.percentile(vals, 10):.4f} p25={np.percentile(vals, 25):.4f} "
                  f"median={np.median(vals):.4f} p75={np.percentile(vals, 75):.4f}")

        # Spearman between head OOF and surv blend OOF
        from scipy.stats import spearmanr
        print("\n=== Spearman: Head OOF vs SurvBlend OOF ===")
        for h in [12, 24, 48]:
            sr, _ = spearmanr(head_oof[h], surv_blend[h])
            print(f"  {h}h: rho={sr:.4f}")

    # ================================================================
    # Full retrain: base models + heads for test prediction
    # ================================================================
    from src.stacking import _train_predict_base, HEAD_HORIZONS

    RETRAIN_SEEDS = [42, 123, 456, 789, 2026]
    X_train = train[feature_cols]
    X_test = test[feature_cols]

    print(f"\n=== Full retrain base models ({len(RETRAIN_SEEDS)} seeds) ===")
    base_test_all = []
    for seed in RETRAIN_SEEDS:
        bp = _train_predict_base(X_train, y_time, y_event, X_test, seed=seed)
        base_test_all.append(bp)
        print(f"  Seed {seed}: RSF+EST+XGBCox done")

    # Average base predictions across seeds
    base_names = list(base_test_all[0].keys())
    base_test_preds = {}
    for name in base_names:
        base_test_preds[name] = {
            h: np.mean([bp[name][h] for bp in base_test_all], axis=0)
            for h in HEAD_HORIZONS
        }

    if args.disable_head:
        print("\n=== Building test predictions from survival baseline ===")
        test_preds = {
            h: best_surv_w * base_test_preds["RSF"][h] + (1 - best_surv_w) * base_test_preds["EST"][h]
            for h in [12, 24, 48]
        }
        test_preds[72] = np.ones(len(test))
    else:
        # Predict test with heads
        print("\n=== Predicting test with horizon heads ===")
        test_head_preds = predict_horizon_heads(heads, X_test, base_test_preds)
        test_preds = {h: test_head_preds[h] for h in [12, 24, 48]}
        test_preds[72] = np.ones(len(test))

    # Generate submission
    print("\n=== Generating submission ===")
    _validate_and_save(test_preds, postprocess_fn=postprocess_fn, floor_for_diag=post_floor)


if __name__ == "__main__":
    main()
