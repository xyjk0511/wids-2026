"""Exp33: Split-conformal anchor calibration (止损式).

Two methods, anchor-based only on 24h/48h:
  A) Median residual shift in probability space
  B) Quantile recalibration (rank-based nonparametric mapping)

Hard gates: OOF Hybrid >= +0.0015, rho >= 0.90, CI no drop.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import (
    TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH, TIME_COL, EVENT_COL,
    HORIZONS, PROB_COLS, FEATURES_V96624_BASE, FEATURES_V96624_ENGINEERED,
)
from src.features import add_engineered
from src.models import RSF, EST
from src.labels import build_horizon_labels
from src.evaluation import hybrid_score
from src.monotonic import submission_postprocess
from src.train import _strat_labels

FEATURES = list(FEATURES_V96624_BASE) + list(FEATURES_V96624_ENGINEERED)
SEEDS = [42, 123, 456, 789, 2026]
ANCHOR_PATH = "submissions/submission_0.96624.csv"
EPS = 1e-7

# Hard gates
GATE_DHYBRID = 0.0015
GATE_SPEARMAN = 0.90


def run_oof(train, feature_cols):
    """5-fold x 5-seed OOF for RSF+EST 50/50 blend."""
    X = train[feature_cols]
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values
    n = len(train)
    oof = {h: np.zeros(n) for h in HORIZONS}
    counts = np.zeros(n)
    strat = _strat_labels(y_time, y_event)

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(X, strat):
            scaler = StandardScaler()
            X_tr = pd.DataFrame(scaler.fit_transform(X.iloc[tr_idx]), columns=feature_cols)
            X_va = pd.DataFrame(scaler.transform(X.iloc[va_idx]), columns=feature_cols)

            rsf = RSF(random_state=seed); rsf.fit(X_tr, y_time[tr_idx], y_event[tr_idx])
            est = EST(random_state=seed); est.fit(X_tr, y_time[tr_idx], y_event[tr_idx])

            for h in HORIZONS:
                oof[h][va_idx] += 0.5 * rsf.predict_proba(X_va)[h] + 0.5 * est.predict_proba(X_va)[h]
            counts[va_idx] += 1

    for h in HORIZONS:
        oof[h] /= counts

    return oof, y_time, y_event


# ── Method A: Median residual shift ──────────────────────────────

def conformal_median_shift(oof_h, y_time, y_event, horizon, anchor_h, lambdas):
    """Compute median(p_oof - y_true) on eligible, shift anchor."""
    labels, eligible = build_horizon_labels(y_time, y_event, horizon)
    p_e = oof_h[eligible]
    y_e = labels[eligible].astype(float)
    residuals = p_e - y_e
    med_resid = np.median(residuals)
    print(f"    h={horizon}: n_elig={eligible.sum()} median_resid={med_resid:.5f} "
          f"mean_resid={np.mean(residuals):.5f}")

    results = {}
    for lam in lambdas:
        shifted = anchor_h - med_resid * lam
        results[lam] = np.clip(shifted, EPS, 1.0 - EPS)
    return results


# ── Method B: Quantile recalibration ─────────────────────────────

def conformal_quantile_recal(oof_h, y_time, y_event, horizon, anchor_h, alphas):
    """Map anchor through OOF empirical quantile function."""
    labels, eligible = build_horizon_labels(y_time, y_event, horizon)
    p_e = oof_h[eligible]
    y_e = labels[eligible].astype(float)

    # Compute calibrated probabilities via binned reliability
    n_bins = 10
    sorted_idx = np.argsort(p_e)
    bin_size = len(p_e) // n_bins
    cal_map_x, cal_map_y = [], []
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(p_e)
        bin_pred = p_e[sorted_idx[start:end]]
        bin_true = y_e[sorted_idx[start:end]]
        cal_map_x.append(np.mean(bin_pred))
        cal_map_y.append(np.mean(bin_true))

    cal_map_x = np.array(cal_map_x)
    cal_map_y = np.array(cal_map_y)
    print(f"    h={horizon}: cal_map pred={cal_map_x.round(3)} true={cal_map_y.round(3)}")

    # Interpolate anchor through calibration map
    recal = np.interp(anchor_h, cal_map_x, cal_map_y, left=cal_map_y[0], right=cal_map_y[-1])

    results = {}
    for alpha in alphas:
        blended = anchor_h + alpha * (recal - anchor_h)
        results[alpha] = np.clip(blended, EPS, 1.0 - EPS)
    return results


# ── Gate check ───────────────────────────────────────────────────

def check_gate(oof_new, oof_base, y_time, y_event, label):
    """Check OOF gates: dHybrid >= +0.0015, CI no drop."""
    score_new, det_new = hybrid_score(y_time, y_event, oof_new)
    score_base, det_base = hybrid_score(y_time, y_event, oof_base)
    dhybrid = score_new - score_base
    dci = det_new["c_index"] - det_base["c_index"]
    passes = dhybrid >= GATE_DHYBRID and dci >= -0.001

    print(f"  [{label}] Hybrid={score_new:.5f} dHybrid={dhybrid:+.5f} "
          f"CI={det_new['c_index']:.4f} dCI={dci:+.4f} PASS={passes}")
    return passes, score_new, dhybrid


def main():
    print("=== Loading data ===")
    train = add_engineered(pd.read_csv(TRAIN_PATH))
    anchor = pd.read_csv(ANCHOR_PATH)
    feature_cols = [c for c in FEATURES if c in train.columns]

    print(f"\n=== Generating OOF (RSF+EST 50/50, 5-fold x 5-seed) ===")
    oof, y_time, y_event = run_oof(train, feature_cols)

    score_base, det_base = hybrid_score(y_time, y_event, oof)
    print(f"  Baseline: Hybrid={score_base:.5f} CI={det_base['c_index']:.4f} "
          f"WBrier={det_base['weighted_brier']:.5f}")

    # ── Method A: Median residual shift ──
    print("\n=== Method A: Median Residual Shift ===")
    lambdas_a = [0.3, 0.5, 0.7, 1.0]
    best_a = None

    for lam in lambdas_a:
        oof_a = dict(oof)
        for h in [24, 48]:
            shifts = conformal_median_shift(oof[h], y_time, y_event, h, oof[h], [lam])
            oof_a[h] = shifts[lam]
        passes, score, dh = check_gate(oof_a, oof, y_time, y_event, f"A_lam{lam}")
        if passes and (best_a is None or score > best_a[1]):
            best_a = (lam, score, dh)

    # ── Method B: Quantile recalibration ──
    print("\n=== Method B: Quantile Recalibration ===")
    alphas_b = [0.1, 0.2, 0.3, 0.5]
    best_b = None

    for alpha in alphas_b:
        oof_b = dict(oof)
        for h in [24, 48]:
            recals = conformal_quantile_recal(oof[h], y_time, y_event, h, oof[h], [alpha])
            oof_b[h] = recals[alpha]
        passes, score, dh = check_gate(oof_b, oof, y_time, y_event, f"B_a{alpha}")
        if passes and (best_b is None or score > best_b[1]):
            best_b = (alpha, score, dh)

    # ── Select best ──
    print("\n=== Results ===")
    candidates = []
    if best_a:
        candidates.append(("A", best_a[0], best_a[1], best_a[2]))
        print(f"  Best A: lam={best_a[0]} hybrid={best_a[1]:.5f} dH={best_a[2]:+.5f}")
    if best_b:
        candidates.append(("B", best_b[0], best_b[1], best_b[2]))
        print(f"  Best B: alpha={best_b[0]} hybrid={best_b[1]:.5f} dH={best_b[2]:+.5f}")

    if not candidates:
        print("\n  NO candidates pass gate. Phase 3 止损 — 不生成提交.")
        print("  建议: 转 Phase 4 (更强锚点获取)")
        return

    best = max(candidates, key=lambda x: x[2])
    method, param, score, dh = best
    print(f"\n  Winner: Method {method} param={param} hybrid={score:.5f} dH={dh:+.5f}")

    # ── Generate submission ──
    print("\n=== Generating submission ===")
    sub_preds = {h: anchor[f"prob_{h}h"].values.copy() for h in HORIZONS}

    for h in [24, 48]:
        if method == "A":
            shifts = conformal_median_shift(oof[h], y_time, y_event, h, sub_preds[h], [param])
            sub_preds[h] = shifts[param]
        else:
            recals = conformal_quantile_recal(oof[h], y_time, y_event, h, sub_preds[h], [param])
            sub_preds[h] = recals[param]

    pp = submission_postprocess(sub_preds)
    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = pp[h]

    out = f"submissions/submission_exp33_{method}_{param}.csv"
    sub.to_csv(out, index=False)
    print(f"  Saved: {out}")

    # Spearman vs anchor
    for h in [24, 48]:
        r, _ = spearmanr(sub[f"prob_{h}h"], anchor[f"prob_{h}h"])
        print(f"  rho_{h}h vs anchor: {r:.4f}")


if __name__ == "__main__":
    main()
