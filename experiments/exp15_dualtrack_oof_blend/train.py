"""Exp15: OOF-learned dual-track ensemble with EST-capped rank track.

Track-Rank:
  RSF + EST (EST weight constrained by est_cap), tuned on 12h C-index.

Track-Calib:
  RSF + LogNormalAFT, per-horizon weight search
  (12h optimized by C-index; 24h/48h by Brier; 72h fixed).

Dual-Track:
  Per-horizon alpha blend of Track-Rank and Track-Calib learned on OOF.

Usage:
  python -m experiments.exp15_dualtrack_oof_blend.train
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import (
    EVENT_COL,
    HORIZONS,
    ID_COL,
    N_REPEATS,
    N_SPLITS,
    PROB_COLS,
    RANDOM_STATE,
    SAMPLE_SUB_PATH,
    TEST_PATH,
    TIME_COL,
    TRAIN_PATH,
)
from src.evaluation import c_index, horizon_brier_score, hybrid_score
from src.features import add_engineered, get_feature_set, remove_redundant
from src.models import EST, LogNormalAFT, RSF
from src.monotonic import enforce_monotonicity, submission_postprocess

warnings.filterwarnings("ignore")

FEATURE_LEVEL = "medium"
DEFAULT_STRAT_MODE = "event_time"
DEFAULT_RETRAIN_SEEDS = [42, 123, 456, 789, 2026]
EXP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXP_DIR / "submission.csv"
DEFAULT_REPORT = EXP_DIR / "RESULTS_2026-02-13.md"


def load_data(feature_level: str = FEATURE_LEVEL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test and apply production-consistent feature preprocessing."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if feature_level in ("v96624", "v96624_plus"):
        train = add_engineered(train)
        test = add_engineered(test)
    else:
        train = add_engineered(remove_redundant(train))
        test = add_engineered(remove_redundant(test))
    return train, test


def _merge_rare_labels(labels: np.ndarray, min_count: int) -> np.ndarray:
    """Merge rare event-time labels so repeated stratified folds remain valid."""
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


def build_strat_labels(
    y_time: np.ndarray,
    y_event: np.ndarray,
    mode: str = DEFAULT_STRAT_MODE,
    n_splits: int = N_SPLITS,
) -> np.ndarray:
    """Build stratification labels."""
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


def _format_label_counts(labels: np.ndarray) -> str:
    c = Counter(np.asarray(labels, dtype=int).tolist())
    return " ".join(f"{k}:{v}" for k, v in sorted(c.items(), key=lambda x: x[0]))


def grid_values(low: float, high: float, step: float) -> list[float]:
    """Stable inclusive float grid."""
    if step <= 0:
        raise ValueError("step must be > 0")
    n_steps = int(round((high - low) / step))
    vals = [low + i * step for i in range(max(n_steps, 0) + 1)]
    if vals and vals[-1] < high - 1e-12:
        vals.append(high)
    return [float(np.clip(v, low, high)) for v in vals]


def blend_two(prob_a: dict[int, np.ndarray], prob_b: dict[int, np.ndarray], w_a: float) -> dict[int, np.ndarray]:
    """Blend two probability dictionaries with weight on A."""
    return {h: w_a * prob_a[h] + (1.0 - w_a) * prob_b[h] for h in HORIZONS}


def objective_for_horizon(
    horizon: int,
    pred_h: np.ndarray,
    y_time: np.ndarray,
    y_event: np.ndarray,
) -> float:
    """Unified scalar objective: higher is better."""
    if horizon == 12:
        return float(c_index(y_time, y_event, pred_h))
    if horizon in (24, 48):
        return float(-horizon_brier_score(y_time, y_event, pred_h, horizon))
    # 72h is hardcoded to 1.0 in final postprocess; do not optimize.
    return 0.0


def search_weight_1d(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    y_time: np.ndarray,
    y_event: np.ndarray,
    horizon: int,
    low: float,
    high: float,
    step: float,
) -> tuple[float, float]:
    """Search weight w for pred = w*A + (1-w)*B maximizing horizon objective."""
    best_w = low
    best_obj = -np.inf
    for w in grid_values(low, high, step):
        pred = w * pred_a + (1.0 - w) * pred_b
        obj = objective_for_horizon(horizon, pred, y_time, y_event)
        if obj > best_obj:
            best_obj = obj
            best_w = w
    return best_w, best_obj


def run_oof_cv(
    train: pd.DataFrame,
    feature_cols: list[str],
    min_samples_leaf: int,
    n_estimators: int,
    strat_mode: str,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> tuple[dict[str, dict[int, np.ndarray]], np.ndarray, np.ndarray]:
    """Run repeated CV and collect OOF for RSF/EST/LogNormalAFT."""
    X = train[feature_cols]
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values
    n = len(train)

    model_names = ["RSF", "EST", "LogNormal"]
    oof = {name: {h: np.zeros(n) for h in HORIZONS} for name in model_names}
    oof_counts = np.zeros(n, dtype=float)

    strat = build_strat_labels(y_time, y_event, mode=strat_mode, n_splits=n_splits)
    print(f"  Strat mode={strat_mode} labels: {_format_label_counts(strat)}")

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    fold_idx = 0
    for tr_idx, va_idx in rskf.split(X, strat):
        fold_idx += 1
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        yt_tr = y_time[tr_idx]
        ye_tr = y_event[tr_idx]

        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(
            scaler.fit_transform(X_tr),
            columns=feature_cols,
            index=X_tr.index,
        )
        X_va_s = pd.DataFrame(
            scaler.transform(X_va),
            columns=feature_cols,
            index=X_va.index,
        )

        fold_seed = random_state + fold_idx
        rsf = RSF(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=fold_seed,
        )
        est = EST(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=fold_seed,
        )
        lnaft = LogNormalAFT()

        rsf.fit(X_tr_s, yt_tr, ye_tr)
        est.fit(X_tr_s, yt_tr, ye_tr)
        # AFT has its own internal scaling; use raw fold features.
        lnaft.fit(X_tr, yt_tr, ye_tr)

        rsf_p = rsf.predict_proba(X_va_s)
        est_p = est.predict_proba(X_va_s)
        lnaft_p = lnaft.predict_proba(X_va)

        for h in HORIZONS:
            oof["RSF"][h][va_idx] += rsf_p[h]
            oof["EST"][h][va_idx] += est_p[h]
            oof["LogNormal"][h][va_idx] += lnaft_p[h]
        oof_counts[va_idx] += 1.0

        if fold_idx % n_splits == 0:
            rep = fold_idx // n_splits
            print(f"  Repeat {rep}/{n_repeats} done")

    mask = oof_counts > 0
    for name in model_names:
        for h in HORIZONS:
            oof[name][h][mask] /= oof_counts[mask]
    return oof, y_time, y_event


def print_score(label: str, prob_dict: dict[int, np.ndarray], y_time: np.ndarray, y_event: np.ndarray) -> tuple[float, dict]:
    """Print Hybrid/CI/WBrier details."""
    score, det = hybrid_score(y_time, y_event, prob_dict)
    print(
        f"  {label:26s} Hybrid={score:.4f} "
        f"CI={det['c_index']:.4f} WBrier={det['weighted_brier']:.4f} "
        f"(B24={det['brier_24h']:.4f} B48={det['brier_48h']:.4f} B72={det['brier_72h']:.4f})"
    )
    return score, det


def learn_dualtrack_weights(
    oof: dict[str, dict[int, np.ndarray]],
    y_time: np.ndarray,
    y_event: np.ndarray,
    est_cap: float,
    lognormal_cap: float,
    step: float,
) -> tuple[dict, dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Learn all dual-track weights from OOF."""
    # Track-Rank: RSF + EST (EST constrained), optimize 12h C-index.
    w_rsf_rank, best_obj = search_weight_1d(
        pred_a=oof["RSF"][12],
        pred_b=oof["EST"][12],
        y_time=y_time,
        y_event=y_event,
        horizon=12,
        low=max(1.0 - est_cap, 0.0),
        high=1.0,
        step=step,
    )
    est_weight = 1.0 - w_rsf_rank
    rank_track = blend_two(oof["RSF"], oof["EST"], w_rsf_rank)

    print("\n=== Learned Track-Rank weight (OOF) ===")
    print(f"  RSF={w_rsf_rank:.3f} EST={est_weight:.3f} (objective@12h={best_obj:.6f})")

    # Track-Calib: RSF + LogNormal, per-horizon weights.
    w_rsf_logn = {}
    calib_track = {}
    print("\n=== Learned Track-Calib per-horizon weights (OOF) ===")
    for h in HORIZONS:
        if h == 72:
            w_rsf_logn[h] = 1.0
            calib_track[h] = oof["RSF"][h].copy()
            print(f"  {h}h: RSF=1.000 LogNormal=0.000 (fixed)")
            continue
        w_rsf_h, obj_h = search_weight_1d(
            pred_a=oof["RSF"][h],
            pred_b=oof["LogNormal"][h],
            y_time=y_time,
            y_event=y_event,
            horizon=h,
            low=max(1.0 - lognormal_cap, 0.0),
            high=1.0,
            step=step,
        )
        w_rsf_logn[h] = w_rsf_h
        calib_track[h] = w_rsf_h * oof["RSF"][h] + (1.0 - w_rsf_h) * oof["LogNormal"][h]
        metric_name = "CI" if h == 12 else "Brier(-)"
        print(f"  {h}h: RSF={w_rsf_h:.3f} LogNormal={1.0 - w_rsf_h:.3f} ({metric_name}={obj_h:.6f})")

    # Dual-Track fusion alpha_h: pred = alpha * rank + (1-alpha) * calib
    alpha_rank = {}
    dual_track = {}
    print("\n=== Learned Dual-Track alpha (OOF) ===")
    for h in HORIZONS:
        if h == 72:
            alpha_rank[h] = 0.0
            dual_track[h] = calib_track[h].copy()
            print(f"  {h}h: alpha_rank=0.000 alpha_calib=1.000 (fixed)")
            continue
        alpha_h, obj_h = search_weight_1d(
            pred_a=rank_track[h],
            pred_b=calib_track[h],
            y_time=y_time,
            y_event=y_event,
            horizon=h,
            low=0.0,
            high=1.0,
            step=step,
        )
        alpha_rank[h] = alpha_h
        dual_track[h] = alpha_h * rank_track[h] + (1.0 - alpha_h) * calib_track[h]
        metric_name = "CI" if h == 12 else "Brier(-)"
        print(f"  {h}h: alpha_rank={alpha_h:.3f} alpha_calib={1.0 - alpha_h:.3f} ({metric_name}={obj_h:.6f})")

    params = {
        "w_rsf_rank": float(w_rsf_rank),
        "w_est_rank": float(est_weight),
        "w_rsf_lognormal": {int(h): float(w_rsf_logn[h]) for h in HORIZONS},
        "alpha_rank": {int(h): float(alpha_rank[h]) for h in HORIZONS},
    }
    return params, rank_track, calib_track, dual_track


def parse_seed_list(seed_text: str) -> list[int]:
    """Parse comma-separated seed list."""
    seeds = []
    for x in seed_text.split(","):
        x = x.strip()
        if not x:
            continue
        seeds.append(int(x))
    if not seeds:
        raise ValueError("retrain_seeds cannot be empty")
    return seeds


def full_retrain_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    y_time: np.ndarray,
    y_event: np.ndarray,
    params: dict,
    retrain_seeds: list[int],
    min_samples_leaf: int,
    n_estimators: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], dict[str, dict[int, np.ndarray]]]:
    """Full retrain with learned OOF weights."""
    X_train = train[feature_cols]
    X_test = test[feature_cols]

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )

    rsf_preds = []
    est_preds = []
    print(f"\n=== Full retrain (RSF/EST, {len(retrain_seeds)} seeds) ===")
    for seed in retrain_seeds:
        rsf = RSF(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
        )
        rsf.fit(X_train_s, y_time, y_event)
        rsf_preds.append(rsf.predict_proba(X_test_s))

        est = EST(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
        )
        est.fit(X_train_s, y_time, y_event)
        est_preds.append(est.predict_proba(X_test_s))
        print(f"  Seed {seed}: RSF+EST done")

    avg = {
        "RSF": {h: np.mean([p[h] for p in rsf_preds], axis=0) for h in HORIZONS},
        "EST": {h: np.mean([p[h] for p in est_preds], axis=0) for h in HORIZONS},
    }

    print("\n=== Full retrain (LogNormalAFT) ===")
    lnaft = LogNormalAFT()
    lnaft.fit(X_train, y_time, y_event)
    avg["LogNormal"] = lnaft.predict_proba(X_test)
    print("  LogNormalAFT done")

    w_rsf_rank = params["w_rsf_rank"]
    rank_track = blend_two(avg["RSF"], avg["EST"], w_rsf_rank)

    w_rsf_logn = params["w_rsf_lognormal"]
    calib_track = {}
    for h in HORIZONS:
        w_rsf_h = float(w_rsf_logn[h])
        calib_track[h] = w_rsf_h * avg["RSF"][h] + (1.0 - w_rsf_h) * avg["LogNormal"][h]

    alpha_rank = params["alpha_rank"]
    dual_track = {}
    for h in HORIZONS:
        a = float(alpha_rank[h])
        dual_track[h] = a * rank_track[h] + (1.0 - a) * calib_track[h]

    return rank_track, calib_track, dual_track, avg


def save_submission(prob_dict: dict[int, np.ndarray], output_path: Path) -> pd.DataFrame:
    """Apply postprocess, validate and save submission file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    test_preds = submission_postprocess(prob_dict)
    test_preds = enforce_monotonicity(test_preds)

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = test_preds[h]

    h72_ok = (sub[PROB_COLS[HORIZONS.index(72)]] == 1.0).all()
    mono_ok = all(
        all(sub[PROB_COLS[j]].iloc[i] >= sub[PROB_COLS[j - 1]].iloc[i] - 1e-9 for j in range(1, len(PROB_COLS)))
        for i in range(len(sub))
    )
    print(
        f"\n=== Submission validation ===\n"
        f"  72h all-ones: {'PASS' if h72_ok else 'FAIL'}  "
        f"Monotonicity: {'PASS' if mono_ok else 'FAIL'}  Shape: {sub.shape}"
    )

    for col in PROB_COLS:
        vals = sub[col]
        print(f"  {col}: min={vals.min():.4f} median={vals.median():.4f} max={vals.max():.4f}")

    sub.to_csv(output_path, index=False)
    print(f"\n  Submission saved to {output_path}")
    return sub


def write_report(
    report_path: Path,
    params: dict,
    oof_scores: dict[str, tuple[float, dict]],
    oof_pp_scores: dict[str, tuple[float, dict]],
    output_path: Path,
) -> None:
    """Write compact markdown report for this experiment."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Exp15 Results (OOF-Learned Dual-Track Blend)")
    lines.append("")
    lines.append("## Learned weights")
    lines.append(f"- Track-Rank: RSF={params['w_rsf_rank']:.3f}, EST={params['w_est_rank']:.3f}")
    lines.append("- Track-Calib (RSF vs LogNormalAFT):")
    for h in HORIZONS:
        w = params["w_rsf_lognormal"][h]
        lines.append(f"  - {h}h: RSF={w:.3f}, LogNormal={1.0 - w:.3f}")
    lines.append("- Dual-Track alpha (rank vs calib):")
    for h in HORIZONS:
        a = params["alpha_rank"][h]
        lines.append(f"  - {h}h: rank={a:.3f}, calib={1.0 - a:.3f}")
    lines.append("")
    lines.append("## OOF scores (raw)")
    for name, (score, det) in oof_scores.items():
        lines.append(
            f"- {name}: Hybrid={score:.4f}, CI={det['c_index']:.4f}, "
            f"WBrier={det['weighted_brier']:.4f}"
        )
    lines.append("")
    lines.append("## OOF scores (postprocessed)")
    for name, (score, det) in oof_pp_scores.items():
        lines.append(
            f"- {name}: Hybrid={score:.4f}, CI={det['c_index']:.4f}, "
            f"WBrier={det['weighted_brier']:.4f}"
        )
    lines.append("")
    lines.append("## Output")
    lines.append(f"- Submission: `{output_path}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("- OOF weight learning is optimized horizon-wise to reflect metric structure.")
    lines.append("- EST is explicitly constrained via `est_cap` to avoid OOF over-commitment.")
    lines.append("- 72h is treated as fixed due submission postprocess behavior.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved to {report_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-level", default=FEATURE_LEVEL)
    p.add_argument("--strat-mode", choices=["event", "event_time"], default=DEFAULT_STRAT_MODE)
    p.add_argument("--cv-seed", type=int, default=RANDOM_STATE)
    p.add_argument("--n-splits", type=int, default=N_SPLITS)
    p.add_argument("--n-repeats", type=int, default=N_REPEATS)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument("--est-cap", type=float, default=0.15)
    p.add_argument("--lognormal-cap", type=float, default=0.30)
    p.add_argument("--weight-step", type=float, default=0.05)
    p.add_argument("--retrain-seeds", type=str, default="42,123,456,789,2026")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    retrain_seeds = parse_seed_list(args.retrain_seeds)

    print("=== Exp15: OOF-learned Dual-Track Ensemble ===")
    print(f"  Feature level: {args.feature_level}")
    print(
        f"  CV: strat={args.strat_mode} seed={args.cv_seed} "
        f"splits={args.n_splits} repeats={args.n_repeats}"
    )
    print(
        f"  Caps: est_cap={args.est_cap:.2f} lognormal_cap={args.lognormal_cap:.2f} "
        f"step={args.weight_step:.2f}"
    )

    train, test = load_data(feature_level=args.feature_level)
    feature_cols = get_feature_set(train, level=args.feature_level)
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    oof, y_time, y_event = run_oof_cv(
        train=train,
        feature_cols=feature_cols,
        min_samples_leaf=args.min_samples_leaf,
        n_estimators=args.n_estimators,
        strat_mode=args.strat_mode,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.cv_seed,
    )

    print("\n=== OOF single-model baselines ===")
    oof_scores_raw = {}
    for name in ("RSF", "EST", "LogNormal"):
        oof_scores_raw[name] = print_score(name, oof[name], y_time, y_event)

    params, rank_track_oof, calib_track_oof, dual_track_oof = learn_dualtrack_weights(
        oof=oof,
        y_time=y_time,
        y_event=y_event,
        est_cap=float(np.clip(args.est_cap, 0.0, 1.0)),
        lognormal_cap=float(np.clip(args.lognormal_cap, 0.0, 1.0)),
        step=args.weight_step,
    )

    print("\n=== OOF blended tracks (raw) ===")
    oof_scores_raw["Track-Rank"] = print_score("Track-Rank", rank_track_oof, y_time, y_event)
    oof_scores_raw["Track-Calib"] = print_score("Track-Calib", calib_track_oof, y_time, y_event)
    oof_scores_raw["Dual-Track"] = print_score("Dual-Track", dual_track_oof, y_time, y_event)

    print("\n=== OOF blended tracks (postprocess) ===")
    oof_scores_pp = {}
    for name, pdict in (
        ("RSF", oof["RSF"]),
        ("Track-Rank", rank_track_oof),
        ("Track-Calib", calib_track_oof),
        ("Dual-Track", dual_track_oof),
    ):
        pp = enforce_monotonicity(submission_postprocess(pdict))
        oof_scores_pp[name] = print_score(f"{name} (pp)", pp, y_time, y_event)

    rank_test, calib_test, dual_test, base_test = full_retrain_predict(
        train=train,
        test=test,
        feature_cols=feature_cols,
        y_time=y_time,
        y_event=y_event,
        params=params,
        retrain_seeds=retrain_seeds,
        min_samples_leaf=args.min_samples_leaf,
        n_estimators=args.n_estimators,
    )

    print("\n=== Full-retrain track distribution (test) ===")
    for name, pdict in (
        ("RSF", base_test["RSF"]),
        ("EST", base_test["EST"]),
        ("LogNormal", base_test["LogNormal"]),
        ("Track-Rank", rank_test),
        ("Track-Calib", calib_test),
        ("Dual-Track", dual_test),
    ):
        p12 = pdict[12]
        print(
            f"  {name:12s} 12h: min={p12.min():.4f} p25={np.percentile(p12,25):.4f} "
            f"median={np.median(p12):.4f} p75={np.percentile(p12,75):.4f} max={p12.max():.4f}"
        )

    save_submission(dual_test, args.output)
    write_report(args.report, params, oof_scores_raw, oof_scores_pp, args.output)

    print("\n=== Learned params summary ===")
    print(f"  Track-Rank: RSF={params['w_rsf_rank']:.3f} EST={params['w_est_rank']:.3f}")
    for h in HORIZONS:
        print(
            f"  {h}h: RSF(LogNormal track)={params['w_rsf_lognormal'][h]:.3f} "
            f"alpha_rank={params['alpha_rank'][h]:.3f}"
        )


if __name__ == "__main__":
    main()

