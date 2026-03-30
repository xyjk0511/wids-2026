"""Exp18: odds_scale calibration (48h priority, stepwise decoupling).

Strategy: only touch 48h first (1 param), confirm projection doesn't eat gains,
then optionally add 24h. Subgroup calibration as third step, gated by diagnostic.

Usage:
    python -m scripts.exp18_subgroup_odds_scale
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import (
    HORIZONS, PROB_COLS, TIME_COL, EVENT_COL, SAMPLE_SUB_PATH,
    N_SPLITS, N_REPEATS, RANDOM_STATE,
)
from src.labels import build_horizon_labels
from src.calibration import odds_scale, fit_odds_scale_brier
from src.evaluation import (
    hybrid_score, horizon_brier_score, weighted_brier_score,
)
from src.monotonic import submission_postprocess
from src.train import load_data, run_cv, _strat_labels, _print_score


# ---------------------------------------------------------------------------
# Step 0: Brier contribution diagnostic
# ---------------------------------------------------------------------------
def brier_contribution_diagnostic(blend, y_time, y_event, subgroup,
                                  horizon=48, n_boot=2000):
    """Per-subgroup Brier contribution with bootstrap significance test."""
    print(f"\n=== Step 0: Brier Contribution Diagnostic ({horizon}h) ===")

    labels, eligible = build_horizon_labels(y_time, y_event, horizon)
    probs = np.asarray(blend[horizon])
    subgroup = np.asarray(subgroup)
    sq_err = (probs - labels) ** 2

    groups = sorted(np.unique(subgroup[eligible]).astype(int))
    group_errs = {}

    for g in groups:
        mask = eligible & (subgroup == g)
        n_elig = mask.sum()
        pos_rate = labels[mask].mean()
        mean_b = sq_err[mask].mean()
        group_errs[g] = sq_err[mask]
        print(f"  group{g}(low_temp={g}) n_eligible={n_elig} "
              f"pos_rate={pos_rate:.2f} mean_brier={mean_b:.4f}")

    if len(groups) < 2:
        return False

    err0, err1 = group_errs[groups[0]], group_errs[groups[1]]
    obs_diff = err0.mean() - err1.mean()
    rng = np.random.default_rng(42)
    boot_diffs = np.empty(n_boot)
    for b in range(n_boot):
        s0 = rng.choice(err0, size=len(err0), replace=True)
        s1 = rng.choice(err1, size=len(err1), replace=True)
        boot_diffs[b] = s0.mean() - s1.mean()

    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    significant = not (ci_lo <= 0 <= ci_hi)
    tag = "SIGNIFICANT" if significant else "NOT significant"
    print(f"  diff={obs_diff:+.4f}  95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]  {tag}")

    if abs(obs_diff) < 0.002 and not significant:
        print("  Decision: diff < 0.002 and CI crosses 0 -> skip subgroup cal")
    return significant


# ---------------------------------------------------------------------------
# Cross-fitted odds_scale (Steps 1 / 1b / 2)
# ---------------------------------------------------------------------------
def cross_fitted_odds_scale(
    blend, y_time, y_event, strat, horizons,
    subgroup=None, shrink_denom=100,
    n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE,
):
    """Cross-fitted odds_scale. Returns (calibrated_oof, fold_scales)."""
    n = len(y_time)
    fold_scales = {}
    global_fold_scales = {}
    for h in horizons:
        fold_scales[h] = {} if subgroup is not None else []
        global_fold_scales[h] = []

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state,
    )

    oof_cal = {h: np.zeros(n) for h in horizons}
    oof_counts = np.zeros(n)

    for tr_idx, va_idx in rskf.split(np.zeros(n), strat):
        for h in horizons:
            labels_full, eligible_full = build_horizon_labels(
                y_time, y_event, h,
            )
            tr_elig = eligible_full.copy()
            tr_elig[va_idx] = False

            if subgroup is None:
                scale = (fit_odds_scale_brier(
                    blend[h][tr_elig], labels_full[tr_elig],
                ) if tr_elig.sum() >= 10 else 1.0)
                fold_scales[h].append(scale)
                global_fold_scales[h].append(scale)
                oof_cal[h][va_idx] += odds_scale(blend[h][va_idx], scale)
            else:
                sub_arr = np.asarray(subgroup)
                groups = sorted(np.unique(sub_arr).astype(int))
                scale_global = (fit_odds_scale_brier(
                    blend[h][tr_elig], labels_full[tr_elig],
                ) if tr_elig.sum() >= 10 else 1.0)
                global_fold_scales[h].append(scale_global)

                for g in groups:
                    g_tr = tr_elig & (sub_arr == g)
                    n_g = g_tr.sum()
                    scale_sub = (fit_odds_scale_brier(
                        blend[h][g_tr], labels_full[g_tr],
                    ) if n_g >= 5 else scale_global)

                    shrink = min(1.0, n_g / shrink_denom)
                    log_s = (shrink * np.log(scale_sub)
                             + (1 - shrink) * np.log(scale_global))
                    scale_final = np.exp(log_s)

                    fold_scales[h].setdefault(g, []).append(scale_final)

                    va_g = np.zeros(n, dtype=bool)
                    va_g[va_idx] = True
                    va_g &= (sub_arr == g)
                    oof_cal[h][va_g] += odds_scale(
                        blend[h][va_g], scale_final,
                    )
        oof_counts[va_idx] += 1

    mask = oof_counts > 0
    calibrated = {h: blend[h].copy() for h in HORIZONS}
    for h in horizons:
        oof_cal[h][mask] /= oof_counts[mask]
        calibrated[h] = oof_cal[h]

    return calibrated, fold_scales


# ---------------------------------------------------------------------------
# Monitoring & helpers
# ---------------------------------------------------------------------------
def print_coupling_monitor(label, pre_dict, post_dict, y_time, y_event):
    """Print B24/B48/WBrier before vs after projection."""
    b24_pre = horizon_brier_score(y_time, y_event, pre_dict[24], 24)
    b48_pre = horizon_brier_score(y_time, y_event, pre_dict[48], 48)
    wb_pre = weighted_brier_score(y_time, y_event, pre_dict)
    b24_post = horizon_brier_score(y_time, y_event, post_dict[24], 24)
    b48_post = horizon_brier_score(y_time, y_event, post_dict[48], 48)
    wb_post = weighted_brier_score(y_time, y_event, post_dict)

    print(f"  {label}")
    print(f"    B24: {b24_pre:.6f} -> {b24_post:.6f} "
          f"(delta={b24_post - b24_pre:+.6f})")
    print(f"    B48: {b48_pre:.6f} -> {b48_post:.6f} "
          f"(delta={b48_post - b48_pre:+.6f})")
    print(f"    WBrier: {wb_pre:.6f} -> {wb_post:.6f} "
          f"(delta={wb_post - wb_pre:+.6f})")
    if b24_post > b24_pre + 0.0005:
        print(f"    WARNING: B24 degraded by "
              f"{b24_post - b24_pre:+.6f} (> 0.0005 threshold)")
    return wb_pre, wb_post


def run_step(step_label, blend, y_time, y_event, strat, horizons,
             subgroup=None, shrink_denom=100):
    """Run one calibration step with full monitoring."""
    print(f"\n=== {step_label} ===")

    pre_pp = submission_postprocess(blend)
    _print_score("Before (raw)", blend, y_time, y_event)
    _print_score("Before (proj)", pre_pp, y_time, y_event)

    calibrated, fold_scales = cross_fitted_odds_scale(
        blend, y_time, y_event, strat,
        horizons=horizons, subgroup=subgroup, shrink_denom=shrink_denom,
    )

    cal_dict = {h: calibrated.get(h, blend[h]) for h in HORIZONS}
    cal_pp = submission_postprocess(cal_dict)

    _print_score("After cal (raw)", cal_dict, y_time, y_event)
    _print_score("After cal (proj)", cal_pp, y_time, y_event)

    wb_pre, wb_post = print_coupling_monitor(
        "Projection coupling", blend, cal_pp, y_time, y_event,
    )

    for h in horizons:
        if subgroup is not None and isinstance(fold_scales[h], dict):
            for g, scales in fold_scales[h].items():
                gm = np.exp(np.mean(np.log(scales)))
                print(f"  Scale {h}h group={g}: geo_mean={gm:.4f} "
                      f"min={min(scales):.4f} max={max(scales):.4f} "
                      f"n_folds={len(scales)}")
        else:
            scales = fold_scales[h]
            gm = np.exp(np.mean(np.log(scales)))
            print(f"  Scale {h}h: geo_mean={gm:.4f} "
                  f"min={min(scales):.4f} max={max(scales):.4f} "
                  f"n_folds={len(scales)}")

    return cal_dict, cal_pp, fold_scales, wb_pre, wb_post


# ---------------------------------------------------------------------------
# Step 3: Apply calibration to test
# ---------------------------------------------------------------------------
def apply_calibration_to_test(test_blend, fold_scales, horizons,
                              subgroup_test=None):
    """Apply geometric-mean scale from CV folds to test predictions."""
    result = {h: test_blend[h].copy() for h in HORIZONS}

    for h in horizons:
        if subgroup_test is not None and isinstance(fold_scales[h], dict):
            sub_arr = np.asarray(subgroup_test)
            for g, scales in fold_scales[h].items():
                gm = np.exp(np.mean(np.log(scales)))
                g_mask = sub_arr == g
                result[h][g_mask] = odds_scale(
                    test_blend[h][g_mask], gm,
                )
                print(f"  Test {h}h group={g}: scale={gm:.4f} "
                      f"n={g_mask.sum()}")
        else:
            scales = fold_scales[h]
            gm = np.exp(np.mean(np.log(scales)))
            result[h] = odds_scale(test_blend[h], gm)
            print(f"  Test {h}h: applying geo_mean scale={gm:.4f}")

    return result


def save_submission(test_preds, path, ref_path="submission_0.96624.csv"):
    """Postprocess, save, and compare to reference."""
    pp = submission_postprocess(test_preds)

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    for h, col in zip(HORIZONS, PROB_COLS):
        sub[col] = pp[h]

    h72_ok = (sub["prob_72h"] == 1.0).all()
    print(f"  72h all-ones: {'PASS' if h72_ok else 'FAIL'}  "
          f"Shape: {sub.shape}")

    sub.to_csv(path, index=False)
    print(f"  Saved: {path}")

    try:
        ref = pd.read_csv(ref_path)
        print(f"  Spearman vs {ref_path}:")
        for col in PROB_COLS[:-1]:
            sr, _ = spearmanr(sub[col], ref[col])
            print(f"    {col}: rho={sr:.6f}")
    except FileNotFoundError:
        print(f"  [WARN] Reference '{ref_path}' not found")

    return sub


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== Exp18: odds_scale Calibration ===\n")

    # Load data and run CV
    train, test = load_data(feature_level="medium")
    from src.features import get_feature_set
    feature_cols = get_feature_set(train, level="medium")

    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    print("=== Running CV for OOF predictions ===")
    oof = run_cv(train, feature_cols)

    # RSF:EST 50:50 blend
    blend = {h: 0.5 * oof["RSF"][h] + 0.5 * oof["EST"][h] for h in HORIZONS}
    _print_score("OOF Blend (raw)", blend, y_time, y_event)

    pp_blend = submission_postprocess(blend)
    _print_score("OOF Blend (proj)", pp_blend, y_time, y_event)

    strat = _strat_labels(y_time, y_event)
    low_temp = train["low_temporal_resolution_0_5h"].values

    # Step 0
    step0_sig = brier_contribution_diagnostic(
        blend, y_time, y_event, low_temp, horizon=48,
    )

    # Step 1: Global 48h-only
    cal1, pp1, scales1, _, _ = run_step(
        "Step 1: Global 48h-only odds_scale",
        blend, y_time, y_event, strat, horizons=[48],
    )

    # Step 1b: Global 24+48
    cal1b, pp1b, scales1b, _, _ = run_step(
        "Step 1b: Global 24+48 odds_scale",
        blend, y_time, y_event, strat, horizons=[24, 48],
    )

    # Step 2: Subgroup 48h (gated by Step 0)
    if step0_sig:
        cal2, pp2, scales2, _, _ = run_step(
            "Step 2: Subgroup 48h odds_scale",
            blend, y_time, y_event, strat,
            horizons=[48], subgroup=low_temp, shrink_denom=100,
        )
    else:
        print("\n=== Step 2: SKIPPED (Step 0 not significant) ===")
        pp2, scales2 = None, None

    # Summary
    print("\n" + "=" * 60)
    print("=== Summary ===")
    print("=" * 60)
    _print_score("Baseline (proj)", pp_blend, y_time, y_event)
    _print_score("Step1: 48h-only", pp1, y_time, y_event)
    _print_score("Step1b: 24+48", pp1b, y_time, y_event)
    if pp2 is not None:
        _print_score("Step2: sub 48h", pp2, y_time, y_event)

    # Step 3: Test submissions
    print("\n=== Step 3: Generating test submissions ===")
    from src.stacking import _train_predict_base, HEAD_HORIZONS

    X_train = train[feature_cols]
    X_test = test[feature_cols]

    RETRAIN_SEEDS = [42, 123, 456, 789, 2026]
    print(f"  Full retrain ({len(RETRAIN_SEEDS)} seeds)...")
    base_all = []
    for seed in RETRAIN_SEEDS:
        bp = _train_predict_base(X_train, y_time, y_event, X_test, seed=seed)
        base_all.append(bp)
        print(f"    Seed {seed} done")

    base_names = list(base_all[0].keys())
    base_avg = {}
    for name in base_names:
        base_avg[name] = {
            h: np.mean([bp[name][h] for bp in base_all], axis=0)
            for h in HEAD_HORIZONS
        }

    # Test blend: RSF:EST 50:50 (matching OOF blend)
    test_blend = {}
    for h in HORIZONS:
        if h in HEAD_HORIZONS and "RSF" in base_avg and "EST" in base_avg:
            test_blend[h] = (0.5 * base_avg["RSF"][h]
                             + 0.5 * base_avg["EST"][h])
        else:
            test_blend[h] = np.ones(len(test))

    low_temp_test = test["low_temporal_resolution_0_5h"].values

    # Submission B: 48h-only
    print("\n--- Submission B: Global 48h-only odds_scale ---")
    test_b = apply_calibration_to_test(test_blend, scales1, horizons=[48])
    save_submission(test_b, "submission_exp18_48only.csv")

    # Submission C: 24+48
    print("\n--- Submission C: Global 24+48 odds_scale ---")
    test_c = apply_calibration_to_test(test_blend, scales1b, horizons=[24, 48])
    save_submission(test_c, "submission_exp18_24_48.csv")

    # Submission D: Subgroup 48h
    if scales2 is not None:
        print("\n--- Submission D: Subgroup 48h odds_scale ---")
        test_d = apply_calibration_to_test(
            test_blend, scales2, horizons=[48],
            subgroup_test=low_temp_test,
        )
        save_submission(test_d, "submission_exp18_sub48.csv")

    print("\n=== Exp18 complete ===")


if __name__ == "__main__":
    main()
